# -*- coding: utf-8 -*-
"""
Centro Logístico de Simulação Meteorológica
Copyright (c) 1995-2026, Instituto de Controle do Espaço Aéreo

unt_janelas

janelas de treino/avaliação (4 entradas + 6 alvos, +10..+60 min) a partir dos CAPPIs
prontos de cappi_products. A tabela sequences do 27.nowcast só tem 1 alvo (+10 min), e
o horizonte do projeto é de 30 a 60 min.

fluxo:
  1. associa_slots(): encadeia os produtos em slots da cadência nominal (600 s). Cada
     produto vai round(Δt / intervalo) slots depois do ANTERIOR, então a deriva dos
     radares (594-622 s por volume) nunca se acumula; os slots intermediários ficam
     vazios (None). É a regra do Timeline do 27.nowcast, reimplementada aqui (este
     projeto não importa código de lá). Dois produtos no mesmo slot: o observado vence.
  2. monta_janelas(): toda sequência de num_entradas + num_alvos slots preenchidos.
     Aceita quando:
       - todos os alvos são OBSERVADOS (as entradas podem ser interpoladas);
       - todo produto tem valid_fraction >= min_valid_ratio x a maior da série;
       - a média de rain_fraction das entradas >= min_rain_fraction.
     quality: 0 = tudo observado, 1 = alguma entrada interpolada.
  3. atribui_split(): treino/validacao/teste por BLOCO de tempo (padrão: o dia UTC;
     hash estável do bloco), para que um mesmo evento nunca caia em dois conjuntos.
     Purge na fronteira: a janela cujos quadros caem em blocos de conjuntos
     DIFERENTES é descartada ("fronteira_split") -- senão os alvos de uma janela de
     treino das 23:50 seriam as entradas de uma janela de teste das 00:30. Blocos
     menores (bloco_split_s) só para testes com pouco dado.
  4. grava_manifesto()/le_manifesto(): CSV (uma janela por linha) + JSON com os
     parâmetros. Os três backends treinam e são avaliados sobre o MESMO manifesto.

2026.out  mlabru  purge na fronteira do split (split_da_janela, "fronteira_split")
2026.out  mlabru  initial version (Linux/Python)
"""
# < imports >----------------------------------------------------------------------------

# python library
import csv
import hashlib
import json
import logging
import math
import pathlib
import typing

# < constants >--------------------------------------------------------------------------

# origens de cappi_products (27.nowcast)
CS_ORIGEM_OBSERVADA = "observed"

# nomes dos conjuntos
CS_SPLIT_TREINO = "treino"
CS_SPLIT_VALIDACAO = "validacao"
CS_SPLIT_TESTE = "teste"
CT_SPLITS = (CS_SPLIT_TREINO, CS_SPLIT_VALIDACAO, CS_SPLIT_TESTE)

# bloco padrão do split: um dia UTC (s)
CI_SEGUNDOS_POR_DIA = 86400

# < logging >----------------------------------------------------------------------------

# logger
M_LOG = logging.getLogger(__name__)
M_LOG.setLevel(logging.INFO)

# < tipos >------------------------------------------------------------------------------

class SProduto(typing.NamedTuple):
    """
    um CAPPI pronto (linha de cappi_products).
    """
    timestamp: int
    origin: str
    valid_fraction: float
    rain_fraction: float

# -------------------------------------------------------------------------------------
class SParametrosJanela(typing.NamedTuple):
    """
    critério de montagem das janelas ([janelas] do config.ini).
    """
    intervalo_s: int = 600
    tolerancia_s: int = 300
    num_entradas: int = 4
    num_alvos: int = 6
    min_valid_ratio: float = 0.95
    min_rain_fraction: float = 0.0
    frac_treino: float = 0.70
    frac_validacao: float = 0.15
    bloco_split_s: int = 86400

# -------------------------------------------------------------------------------------
class SJanela(typing.NamedTuple):
    """
    uma janela: timestamps (epoch UTC) das entradas e dos alvos, em ordem cronológica.
    """
    radar: str
    alcance: int
    preset: str
    entradas: tuple
    alvos: tuple
    quality: int
    split: str

# < associação em slots >----------------------------------------------------------------

def _arredonda_meio_para_cima(ff_valor: float) -> int:
    """
    arredondamento "meio para cima" (2.5 -> 3), sem o arredondamento bancário do round().
    """
    return int(math.floor(ff_valor + 0.5))

# -------------------------------------------------------------------------------------
def _prefere(fprod_atual: SProduto, fprod_novo: SProduto) -> SProduto:
    """
    entre dois produtos no mesmo slot, o observado vence; empate, fica o primeiro.
    """
    lb_novo_observado = CS_ORIGEM_OBSERVADA == fprod_novo.origin
    lb_atual_observado = CS_ORIGEM_OBSERVADA == fprod_atual.origin

    if lb_novo_observado and not lb_atual_observado:
        return fprod_novo

    return fprod_atual

# -------------------------------------------------------------------------------------
def associa_slots(flst_produtos: list, fi_intervalo_s: int=600,
                  fi_tolerancia_s: int=300) -> list:
    """
    encadeia os produtos (em ordem cronológica) em slots da cadência nominal.

    um Δt cujo resíduo |Δt - n·intervalo| passa da tolerância quebra a série: entra um
    slot vazio (None), que nenhuma janela atravessa. Com tolerância = intervalo/2 isso
    nunca acontece.

    :param flst_produtos: SProduto em ordem crescente de timestamp
    :param fi_intervalo_s: cadência nominal (s)
    :param fi_tolerancia_s: resíduo máximo (s), no máximo intervalo/2

    :returns: lista de slots (SProduto ou None = faltante)

    :raises ValueError: tolerância maior que meio intervalo
    """
    if fi_tolerancia_s * 2 > fi_intervalo_s:
        raise ValueError("tolerancia_s não pode passar de intervalo_s / 2")

    llst_slots = []

    for lprod in flst_produtos:
        if not llst_slots:
            llst_slots.append(lprod)
            continue

        li_delta_s = lprod.timestamp - _ultimo_preenchido(llst_slots).timestamp
        li_passos = _arredonda_meio_para_cima(li_delta_s / fi_intervalo_s)

        # mesmo slot: fica o preferido
        if 0 == li_passos:
            llst_slots[-1] = _prefere(llst_slots[-1], lprod)
            continue

        lb_quebra = abs(li_delta_s - li_passos * fi_intervalo_s) > fi_tolerancia_s
        li_vazios = li_passos - 1 + (1 if lb_quebra else 0)

        llst_slots.extend([None] * li_vazios)
        llst_slots.append(lprod)

    return llst_slots

# -------------------------------------------------------------------------------------
def _ultimo_preenchido(flst_slots: list) -> SProduto:
    """
    último slot da lista (sempre preenchido: vazios só entram antes de um produto).
    """
    return flst_slots[-1]

# < janelas >----------------------------------------------------------------------------

def _maior_valid_fraction(flst_produtos: list) -> float:
    """
    maior valid_fraction OBSERVADA da série (referência do filtro relativo).
    """
    lf_maior = 0.0

    for lprod in flst_produtos:
        if CS_ORIGEM_OBSERVADA != lprod.origin:
            continue

        lf_maior = max(lf_maior, lprod.valid_fraction or 0.0)

    return lf_maior

# -------------------------------------------------------------------------------------
def motivo_rejeicao(flst_entradas: list, flst_alvos: list, ff_valid_minimo: float,
                    ff_min_rain_fraction: float) -> typing.Optional[str]:
    """
    critério de aceitação de uma janela (todos os slots já preenchidos).

    :param flst_entradas: SProduto das entradas
    :param flst_alvos: SProduto dos alvos
    :param ff_valid_minimo: valid_fraction mínima (já multiplicada pelo máximo da série)
    :param ff_min_rain_fraction: média mínima de rain_fraction das entradas

    :returns: None se aceita, senão o motivo ("alvo_interpolado", "cobertura", "seca")
    """
    for lprod in flst_alvos:
        if CS_ORIGEM_OBSERVADA != lprod.origin:
            return "alvo_interpolado"

    for lprod in list(flst_entradas) + list(flst_alvos):
        if (lprod.valid_fraction or 0.0) < ff_valid_minimo:
            return "cobertura"

    lf_soma_chuva = 0.0
    for lprod in flst_entradas:
        lf_soma_chuva += lprod.rain_fraction or 0.0

    if lf_soma_chuva / len(flst_entradas) < ff_min_rain_fraction:
        return "seca"

    return None

# -------------------------------------------------------------------------------------
def _quality(flst_entradas: list) -> int:
    """
    0 = todas as entradas observadas, 1 = alguma interpolada (regra do nwcSequences).
    """
    for lprod in flst_entradas:
        if CS_ORIGEM_OBSERVADA != lprod.origin:
            return 1

    return 0

# -------------------------------------------------------------------------------------
def atribui_split(fi_timestamp: int, ff_frac_treino: float, ff_frac_validacao: float,
                  fi_bloco_s: int=CI_SEGUNDOS_POR_DIA) -> str:
    """
    conjunto de uma janela pelo bloco de tempo do timestamp (padrão: o dia UTC). Hash
    estável: o mesmo bloco cai sempre no mesmo conjunto, em qualquer máquina.

    :param fi_timestamp: epoch UTC (a última entrada da janela)
    :param ff_frac_treino: fração dos blocos no treino
    :param ff_frac_validacao: fração dos blocos na validação (o resto é teste)
    :param fi_bloco_s: tamanho do bloco (s)

    :returns: "treino", "validacao" ou "teste"
    """
    li_bloco = fi_timestamp // fi_bloco_s
    ls_hash = hashlib.sha1(f"bloco-{fi_bloco_s}-{li_bloco}".encode()).hexdigest()

    # primeiros 8 dígitos hexadecimais -> [0, 1)
    lf_sorteio = int(ls_hash[:8], 16) / float(0x100000000)

    if lf_sorteio < ff_frac_treino:
        return CS_SPLIT_TREINO

    if lf_sorteio < ff_frac_treino + ff_frac_validacao:
        return CS_SPLIT_VALIDACAO

    return CS_SPLIT_TESTE

# -------------------------------------------------------------------------------------
def split_da_janela(flst_timestamps: list, ff_frac_treino: float, ff_frac_validacao: float,
                    fi_bloco_s: int=CI_SEGUNDOS_POR_DIA) -> typing.Optional[str]:
    """
    o conjunto de uma janela: o de todos os seus quadros (entradas e alvos), se for o
    mesmo; None se a janela atravessa blocos de conjuntos diferentes (purge, para que
    nenhum quadro seja compartilhado entre conjuntos).

    :param flst_timestamps: timestamps de todos os quadros da janela
    """
    lset_splits = {atribui_split(li_timestamp, ff_frac_treino, ff_frac_validacao, fi_bloco_s)
                   for li_timestamp in flst_timestamps}

    if 1 != len(lset_splits):
        return None

    return lset_splits.pop()

# -------------------------------------------------------------------------------------
def monta_janelas(flst_produtos: list, fs_radar: str, fi_alcance: int, fs_preset: str,
                  ft_parametros: SParametrosJanela=SParametrosJanela()) -> tuple:
    """
    todas as janelas aceitas de uma série (radar, alcance, preset).

    :param flst_produtos: SProduto em ordem cronológica
    :param fs_radar: código do radar
    :param fi_alcance: alcance em km
    :param fs_preset: id do preset
    :param ft_parametros: critério ([janelas])

    :returns: (lista de SJanela, dicionário motivo -> rejeições)
    """
    llst_slots = associa_slots(flst_produtos, ft_parametros.intervalo_s,
                               ft_parametros.tolerancia_s)

    lf_valid_minimo = ft_parametros.min_valid_ratio * _maior_valid_fraction(flst_produtos)
    li_tamanho = ft_parametros.num_entradas + ft_parametros.num_alvos

    llst_janelas = []
    ldct_rejeicoes = {}

    for li_inicio in range(0, len(llst_slots) - li_tamanho + 1):
        llst_trecho = llst_slots[li_inicio:li_inicio + li_tamanho]

        # slot faltante: não é janela candidata
        if any(lprod is None for lprod in llst_trecho):
            continue

        llst_entradas = llst_trecho[:ft_parametros.num_entradas]
        llst_alvos = llst_trecho[ft_parametros.num_entradas:]

        ls_motivo = motivo_rejeicao(llst_entradas, llst_alvos, lf_valid_minimo,
                                    ft_parametros.min_rain_fraction)
        if ls_motivo is not None:
            ldct_rejeicoes[ls_motivo] = ldct_rejeicoes.get(ls_motivo, 0) + 1
            continue

        ls_split = split_da_janela([lprod.timestamp for lprod in llst_trecho],
                                   ft_parametros.frac_treino, ft_parametros.frac_validacao,
                                   ft_parametros.bloco_split_s)
        if ls_split is None:
            ldct_rejeicoes["fronteira_split"] = ldct_rejeicoes.get("fronteira_split", 0) + 1
            continue

        llst_janelas.append(SJanela(fs_radar, fi_alcance, fs_preset,
                                    tuple(lprod.timestamp for lprod in llst_entradas),
                                    tuple(lprod.timestamp for lprod in llst_alvos),
                                    _quality(llst_entradas), ls_split))

    return llst_janelas, ldct_rejeicoes

# -------------------------------------------------------------------------------------
def produtos_de_linhas(flst_linhas: list) -> list:
    """
    converte as linhas de unt_catalog.lista_produtos_prontos em SProduto.
    """
    llst_produtos = []

    for ldct_linha in flst_linhas:
        llst_produtos.append(SProduto(int(ldct_linha["timestamp"]), ldct_linha["origin"],
                                      ldct_linha["valid_fraction"],
                                      ldct_linha["rain_fraction"]))

    return llst_produtos

# < manifesto >--------------------------------------------------------------------------

def _colunas(fi_num_entradas: int, fi_num_alvos: int) -> list:
    """
    cabeçalho do CSV: radar, alcance, preset, in_1..in_N, alvo_1..alvo_M, quality, split.
    """
    llst_colunas = ["radar", "alcance", "preset"]
    llst_colunas += [f"in_{li}" for li in range(1, fi_num_entradas + 1)]
    llst_colunas += [f"alvo_{li}" for li in range(1, fi_num_alvos + 1)]
    llst_colunas += ["quality", "split"]

    return llst_colunas

# -------------------------------------------------------------------------------------
def caminho_parametros(fpth_manifesto: typing.Union[str, pathlib.Path]) -> pathlib.Path:
    """
    o JSON de parâmetros fica ao lado do CSV: <nome>.csv -> <nome>.json.
    """
    return pathlib.Path(fpth_manifesto).with_suffix(".json")

# -------------------------------------------------------------------------------------
def grava_manifesto(flst_janelas: list, fpth_manifesto: typing.Union[str, pathlib.Path],
                    fdct_parametros: dict) -> None:
    """
    grava as janelas (CSV) e os parâmetros (JSON ao lado).

    :param flst_janelas: SJanela (todas com o mesmo número de entradas/alvos)
    :param fpth_manifesto: caminho do .csv
    :param fdct_parametros: parâmetros da montagem (catálogo, preset, [janelas], ...)
    """
    lpth_manifesto = pathlib.Path(fpth_manifesto)
    lpth_manifesto.parent.mkdir(parents=True, exist_ok=True)

    li_num_entradas = fdct_parametros["num_entradas"]
    li_num_alvos = fdct_parametros["num_alvos"]

    with lpth_manifesto.open("w", newline="") as larq:
        lescritor = csv.writer(larq)
        lescritor.writerow(_colunas(li_num_entradas, li_num_alvos))

        for ljanela in flst_janelas:
            lescritor.writerow([ljanela.radar, ljanela.alcance, ljanela.preset,
                                *ljanela.entradas, *ljanela.alvos, ljanela.quality,
                                ljanela.split])

    ls_json = json.dumps(fdct_parametros, indent=2, ensure_ascii=False)
    caminho_parametros(lpth_manifesto).write_text(ls_json)

# -------------------------------------------------------------------------------------
def le_manifesto(fpth_manifesto: typing.Union[str, pathlib.Path]) -> tuple:
    """
    lê um manifesto gravado por grava_manifesto.

    :param fpth_manifesto: caminho do .csv

    :returns: (lista de SJanela, dicionário de parâmetros)
    """
    lpth_manifesto = pathlib.Path(fpth_manifesto)
    ldct_parametros = json.loads(caminho_parametros(lpth_manifesto).read_text())

    li_num_entradas = ldct_parametros["num_entradas"]
    li_num_alvos = ldct_parametros["num_alvos"]

    llst_janelas = []

    with lpth_manifesto.open(newline="") as larq:
        for ldct_linha in csv.DictReader(larq):
            lt_entradas = tuple(int(ldct_linha[f"in_{li}"])
                                for li in range(1, li_num_entradas + 1))
            lt_alvos = tuple(int(ldct_linha[f"alvo_{li}"])
                             for li in range(1, li_num_alvos + 1))

            llst_janelas.append(SJanela(ldct_linha["radar"], int(ldct_linha["alcance"]),
                                        ldct_linha["preset"], lt_entradas, lt_alvos,
                                        int(ldct_linha["quality"]), ldct_linha["split"]))

    return llst_janelas, ldct_parametros

# -------------------------------------------------------------------------------------
def filtra_split(flst_janelas: list, fs_split: str) -> list:
    """
    só as janelas de um conjunto ("treino", "validacao" ou "teste").
    """
    return [ljanela for ljanela in flst_janelas if fs_split == ljanela.split]

# -------------------------------------------------------------------------------------
def conta_por_split(flst_janelas: list) -> dict:
    """
    número de janelas em cada conjunto.
    """
    ldct_contagem = {ls_split: 0 for ls_split in CT_SPLITS}

    for ljanela in flst_janelas:
        ldct_contagem[ljanela.split] = ldct_contagem.get(ljanela.split, 0) + 1

    return ldct_contagem

# < the end >----------------------------------------------------------------------------
