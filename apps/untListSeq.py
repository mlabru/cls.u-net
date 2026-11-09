# -*- coding: utf-8 -*-
"""
Centro Logístico de Simulação Meteorológica
Copyright (c) 1995-2026, Instituto de Controle do Espaço Aéreo

untListSeq

ferramenta de diagnóstico: NÃO gera nowcast, só inspeciona o catálogo
SQLite (o mesmo que untMain.py lê) e mostra, para cada par (radar,
alcance), quais sequências estão prontas para os backends. Uma sequência
é válida se:

  1. os 4 CAPPIs (.npy) de entrada existem em --cappi-dir;
  2. o radar tem posição catalogada (tabela radar_sites);
  3. o "target" é mais novo que --max-age (mesmo limite que o
     unt_common.sh usa para descartar sequências velhas, UNT_AGE_MAX_SECS;
     0 = sem limite).

responde "por que as visualizações não estão sendo geradas?" sem rodar o
untMain.py -- ex.: a sequência mais recente expirou, ou falta um CAPPI.

saída: uma tabela no terminal (horários em UTC) e o resumo
"N/M sequência(s) válida(s)". Código de saída 0 se houver alguma válida.

$ python untListSeq.py [--catalog catalog.db] [--cappi-dir ./data/cappi]
                       [-r CGU] [-a 250] [--all] [--max-age 7200]

2026.out  mlabru  portado de 25.oflow (oflListSeq), sem mudança de lógica.

2026.out  mlabru  main() quebrada em funções de uma responsabilidade; as
                  três verificações em funções próprias; linha da tabela
                  como NamedTuple.

2026.out  mlabru  caminho do CAPPI com preset (layout atual do 27.nowcast);
                  colunas preset/quality na tabela; --max-age 0 desliga o
                  filtro de idade (replay de dados antigos, 03.dat/tst).

2026.ago  mlabru  initial version (Linux/Python)
"""
# < imports >----------------------------------------------------------------------------

# python library
import argparse
import datetime
import logging
import sqlite3
import sys
import time
import typing

# model
import work.unt_catalog as oc

# < constants >--------------------------------------------------------------------------

# idade máxima (s) do "target" de uma sequência válida -- mesmo padrão de
# UNT_AGE_MAX_SECS no unt_cfg.sh (2 horas)
CI_IDADE_MAXIMA_PADRAO_SEGUNDOS = 2 * 60 * 60

# formato das linhas da tabela (largura de cada coluna)
CS_FORMATO_LINHA = "{:<8}{:<9}{:<10}{:<5}{:<21}{:<12}{:<8}{}"

# separador horizontal da tabela
CS_SEPARADOR = "-" * 100

# < logging >----------------------------------------------------------------------------

# logger
M_LOG = logging.getLogger(__name__)
M_LOG.setLevel(logging.INFO)

# < tipos >------------------------------------------------------------------------------

class LinhaTabela(typing.NamedTuple):
    """
    uma linha da tabela impressa (já formatada como texto).
    """
    radar: str
    alcance: str
    preset: str
    quality: str
    data_target: str
    idade: str
    valida: str
    motivos: str

# -------------------------------------------------------------------------------------
def arg_parse() -> argparse.Namespace:
    """
    lê os argumentos da linha de comando.

    :returns: argumentos lidos
    """
    lparser = argparse.ArgumentParser(
        description="Lista as sequências catalogadas e indica quais estão "
                    "prontas para gerar nowcast (CAPPI presente, geometria "
                    "catalogada, dentro da janela de validade)")

    lparser.add_argument("-r", "--radar", type=str, default=None,
                         help="código do site de radar (ex: CGU); "
                              "se omitido, todos os catalogados")
    lparser.add_argument("-a", "--alcance", type=int, default=None,
                         help="alcance em km (ex: 250); se omitido, todos os catalogados")
    lparser.add_argument("--catalog", type=str, default=oc.caminho_catalogo_padrao(),
                         help="caminho do catálogo SQLite")
    lparser.add_argument("--cappi-dir", type=str, default=oc.diretorio_cappi_padrao(),
                         help="diretório dos CAPPI (.npy)")
    lparser.add_argument("--max-age", type=int, default=CI_IDADE_MAXIMA_PADRAO_SEGUNDOS,
                         help="idade máxima (s) do \"target\" para a sequência ser "
                              "considerada válida (0 = sem limite)")
    lparser.add_argument("--all", action="store_true",
                         help="lista todas as sequências catalogadas do par, "
                              "não só a mais recente")

    return lparser.parse_args()

# -------------------------------------------------------------------------------------
def formata_duracao(fi_segundos: int) -> str:
    """
    formata uma duração como "HHhMMm" ou, com dias, "DdHHhMMm".

    durações negativas (target no futuro, relógios dessincronizados) são
    mostradas com sinal, sem "correção": -3600 -> "-01h00m".

    :param fi_segundos: duração em segundos

    :returns: texto, ex.: 3900 -> "01h05m"; 176400 -> "2d01h00m"
    """
    ls_sinal = ""

    if fi_segundos < 0:
        ls_sinal = "-"

    li_segundos = abs(fi_segundos)

    li_dias, li_resto = divmod(li_segundos, 86400)
    li_horas, li_resto = divmod(li_resto, 3600)
    li_minutos = li_resto // 60

    if li_dias > 0:
        return f"{ls_sinal}{li_dias}d{li_horas:02d}h{li_minutos:02d}m"

    return f"{ls_sinal}{li_horas:02d}h{li_minutos:02d}m"

# < verificações de uma sequência >------------------------------------------------------

def _verifica_cappis_presentes(fdct_sequencia: dict,
                               fs_diretorio_cappi: str) -> typing.Optional[str]:
    """
    :returns: o motivo de invalidez ("CAPPI ausente: ..."), ou None se todos
              os CAPPIs de entrada existem
    """
    llst_arquivos_faltando = []

    for li_timestamp in oc.lista_timestamps_de_entrada(fdct_sequencia):
        lpth_cappi = oc.monta_caminho_cappi(fs_diretorio_cappi, fdct_sequencia["radar"],
                                            fdct_sequencia["alcance"], li_timestamp,
                                            fdct_sequencia.get("preset"))
        if not lpth_cappi.exists():
            llst_arquivos_faltando.append(lpth_cappi.name)

    if not llst_arquivos_faltando:
        return None

    return f"CAPPI ausente: {', '.join(llst_arquivos_faltando)}"

# -------------------------------------------------------------------------------------
def _verifica_posicao_radar(fconexao: sqlite3.Connection,
                            fdct_sequencia: dict) -> typing.Optional[str]:
    """
    :returns: o motivo de invalidez, ou None se o radar está em radar_sites
    """
    if oc.busca_posicao_radar(fconexao, fdct_sequencia["radar"]) is not None:
        return None

    return "radar sem geometria catalogada (radar_sites)"

# -------------------------------------------------------------------------------------
def _verifica_idade(fdct_sequencia: dict, fi_idade_maxima: int,
                    fi_agora: int) -> typing.Optional[str]:
    """
    :returns: o motivo de invalidez ("expirada ..."), ou None se o target é
              recente o bastante (ou se fi_idade_maxima é 0 = sem limite)
    """
    if fi_idade_maxima <= 0:
        return None

    li_idade = fi_agora - fdct_sequencia["target"]

    if li_idade <= fi_idade_maxima:
        return None

    return (f"expirada (idade={formata_duracao(li_idade)} > "
            f"{formata_duracao(fi_idade_maxima)})")

# -------------------------------------------------------------------------------------
def avalia_sequencia(fconexao: sqlite3.Connection, fdct_sequencia: dict,
                     fs_diretorio_cappi: str, fi_idade_maxima: int,
                     fi_agora: int) -> tuple:
    """
    aplica as três verificações a uma sequência.

    :param fconexao: conexão com o catálogo
    :param fdct_sequencia: a sequência (linha de sequences)
    :param fs_diretorio_cappi: raiz dos CAPPIs
    :param fi_idade_maxima: idade máxima do target (s); 0 = sem limite
    :param fi_agora: instante de referência (epoch UTC)

    :returns: (valida, lista_de_motivos) -- lista vazia quando válida
    """
    llst_motivos = []

    for ls_motivo in (_verifica_cappis_presentes(fdct_sequencia, fs_diretorio_cappi),
                      _verifica_posicao_radar(fconexao, fdct_sequencia),
                      _verifica_idade(fdct_sequencia, fi_idade_maxima, fi_agora)):
        if ls_motivo is not None:
            llst_motivos.append(ls_motivo)

    lv_valida = (0 == len(llst_motivos))

    return lv_valida, llst_motivos

# < passos de main() >-------------------------------------------------------------------

def filtra_pares(flst_pares: list, fs_radar: typing.Optional[str],
                 fi_alcance: typing.Optional[int]) -> list:
    """
    mantém só os pares (radar, alcance) pedidos em -r/-a (None = qualquer).

    :returns: pares filtrados
    """
    llst_filtrados = []

    for ls_radar, li_alcance in flst_pares:
        if fs_radar is not None and fs_radar != ls_radar:
            continue

        if fi_alcance is not None and fi_alcance != li_alcance:
            continue

        llst_filtrados.append((ls_radar, li_alcance))

    return llst_filtrados

# -------------------------------------------------------------------------------------
def sequencias_do_par(fconexao: sqlite3.Connection, fs_radar: str, fi_alcance: int,
                      fv_todas: bool) -> list:
    """
    :param fv_todas: True = todas as sequências do par (--all); False = só a
                     mais recente

    :returns: lista de sequências (pode ser vazia)
    """
    if fv_todas:
        return oc.lista_sequencias(fconexao, fs_radar, fi_alcance)

    ldct_mais_recente = oc.busca_sequencia_mais_recente(fconexao, fs_radar, fi_alcance)

    if ldct_mais_recente is None:
        return []

    return [ldct_mais_recente]

# -------------------------------------------------------------------------------------
def monta_linha_tabela(fdct_sequencia: dict, fv_valida: bool, flst_motivos: list,
                       fi_agora: int) -> LinhaTabela:
    """
    converte o resultado da avaliação em textos para a tabela.

    :returns: LinhaTabela
    """
    # horário do target em UTC, como nos nomes dos arquivos e no catálogo
    ldt_target = datetime.datetime.fromtimestamp(fdct_sequencia["target"],
                                                 tz=datetime.timezone.utc)

    ls_preset = fdct_sequencia.get("preset") or "-"

    if fdct_sequencia.get("quality") is None:
        ls_quality = "-"
    else:
        ls_quality = str(fdct_sequencia["quality"])

    if fv_valida:
        ls_valida = "sim"
    else:
        ls_valida = "não"

    if flst_motivos:
        ls_motivos = "; ".join(flst_motivos)
    else:
        ls_motivos = "-"

    return LinhaTabela(radar=fdct_sequencia["radar"],
                       alcance=str(fdct_sequencia["alcance"]),
                       preset=ls_preset,
                       quality=ls_quality,
                       data_target=ldt_target.strftime("%Y-%m-%d %H:%MZ"),
                       idade=formata_duracao(fi_agora - fdct_sequencia["target"]),
                       valida=ls_valida,
                       motivos=ls_motivos)

# -------------------------------------------------------------------------------------
def imprime_tabela(flst_linhas: list, fi_num_validas: int, fi_idade_maxima: int) -> None:
    """
    imprime a tabela de sequências e o resumo no terminal (stdout).
    """
    print(CS_FORMATO_LINHA.format("radar", "alcance", "preset", "qlt", "target", "idade",
                                  "válida", "motivo"))
    print(CS_SEPARADOR)

    for lt_linha in flst_linhas:
        print(CS_FORMATO_LINHA.format(*lt_linha))

    print(CS_SEPARADOR)

    if fi_idade_maxima > 0:
        ls_janela = formata_duracao(fi_idade_maxima)
    else:
        ls_janela = "sem limite"

    print(f"{fi_num_validas}/{len(flst_linhas)} sequência(s) válida(s) "
          f"(janela de validade: {ls_janela}).")

# < main >-------------------------------------------------------------------------------

def main() -> int:
    """
    lista as sequências e diz quais estão prontas.

    :returns: 0 se há ao menos uma sequência válida, 1 caso contrário
    """
    largs = arg_parse()

    lconexao = oc.abre_catalogo_somente_leitura(largs.catalog)

    llst_pares = filtra_pares(oc.lista_pares_radar_alcance(lconexao),
                              largs.radar, largs.alcance)
    if not llst_pares:
        M_LOG.error("[x] nenhum par (radar, alcance) catalogado com os filtros informados.")
        return 1

    # um único "agora" para a varredura inteira: todos os pares são julgados
    # pelo mesmo critério de idade
    li_agora = int(time.time())

    llst_linhas = []
    li_num_validas = 0

    for ls_radar, li_alcance in llst_pares:
        for ldct_sequencia in sequencias_do_par(lconexao, ls_radar, li_alcance, largs.all):
            lv_valida, llst_motivos = avalia_sequencia(lconexao, ldct_sequencia,
                                                       largs.cappi_dir, largs.max_age, li_agora)
            if lv_valida:
                li_num_validas += 1

            llst_linhas.append(monta_linha_tabela(ldct_sequencia, lv_valida, llst_motivos,
                                                  li_agora))

    imprime_tabela(llst_linhas, li_num_validas, largs.max_age)

    if 0 == li_num_validas:
        return 1

    return 0

# -------------------------------------------------------------------------------------
# this is the bootstrap process

if "__main__" == __name__:
    # logger
    logging.basicConfig(level=logging.INFO)

    # run
    sys.exit(main())

# < the end >----------------------------------------------------------------------------
