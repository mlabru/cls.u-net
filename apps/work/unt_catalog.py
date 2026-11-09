# -*- coding: utf-8 -*-
"""
Centro Logístico de Simulação Meteorológica
Copyright (c) 1995-2026, Instituto de Controle do Espaço Aéreo

unt_catalog

leitura (SOMENTE leitura) do catálogo SQLite e dos CAPPIs produzidos pelo
pipeline do 27.nowcast (repo irmão, 04.dvl). Este projeto nunca grava
no catálogo e nunca importa código do 27.nowcast: só compartilha o *schema*
das tabelas e a convenção de nomes dos arquivos.

tabelas usadas:
  - sequences:      4 entradas (input_1..input_4, do mais antigo para o
                    mais novo) + 1 target observado, por radar/alcance
                    (runner operacional, untMain);
  - cappi_products: os CAPPIs prontos (status DONE) de cada preset, de onde
                    saem as janelas de treino 4 + 6 (untJanelas); também onde
                    o 27.nowcast gravou cada CAPPI -- só para diagnóstico;
  - radar_sites:    latitude/longitude/altitude de cada radar.

2026.out  mlabru  portado de 25.oflow (ofl_catalog); + preset_ativo_padrao(),
                  lista_produtos_prontos() e lista_pares_com_produtos() para a
                  montagem das janelas de treino.

2026.out  mlabru  nomes descritivos; uma única consulta de sequências
                  (_consulta_sequencias) para listar e buscar a mais recente;
                  lista_timestamps_de_entrada() compartilhada por oflMain e
                  oflListSeq.

2026.out  mlabru  CAPPIs agora em <cappi_dir>/<preset>/<R>_<A>/<R>_<A>_<ts>.npy
                  (layout atual do 27.nowcast, sequences.preset); filtros
                  opcionais de preset/quality nas consultas de sequência;
                  carrega_cappi converte o "fora da cobertura" (-9999) em
                  NaN; o raio do mapa é derivado da grade.

2026.ago  mlabru  initial version (Linux/Python)
"""
# < imports >----------------------------------------------------------------------------

# python library
import os
import pathlib
import sqlite3
import typing

# numPy
import numpy as np

# < constants >--------------------------------------------------------------------------

# valores especiais gravados nos CAPPIs pelo 27.nowcast (cópia do contrato de
# apps/nwc_presets.py -- só o contrato é compartilhado, nunca o código):
#   - fora da cobertura do feixe: vira NaN ao carregar ("sem dado", não
#     "sem chuva");
#   - coberto, sem eco: -32 dBZ, mantido (em mm/h é praticamente 0).
CF_DBZ_FORA_DA_COBERTURA = -9999.0
CF_DBZ_SEM_ECO = -32.0

# resolução das grades dos presets ativos (nwc250v2: ±250 km 500x500;
# nwc400v2: ±300 km 600x600 -- ambos 1 km por pixel)
CF_KM_POR_PIXEL = 1.0

# número de CAPPIs de entrada de cada sequência (colunas input_1..input_4)
CI_NUM_ENTRADAS = 4

# status "pronto" de cappi_products (CI_STATUS_DONE do 27.nowcast)
CI_STATUS_PRONTO = 2

# preset ativo de cada alcance (cópia do contrato nwc_presets.MDCT_ACTIVE do 27.nowcast)
MDCT_PRESET_ATIVO = {250: "nwc250v2", 400: "nwc400v2"}

# espera máxima (s) por um lock do 27.nowcast (que escreve no catálogo
# enquanto lemos) antes de falhar com "database is locked" -- mesmo valor
# de UNT_DB_TIMEOUT_MS no unt_cfg.sh
CF_TIMEOUT_LOCK_SEGUNDOS = 10.0

# < tipos >------------------------------------------------------------------------------

class PosicaoRadar(typing.NamedTuple):
    """
    posição de um radar (tabela radar_sites).
    """
    latitude: float
    longitude: float
    altitude_m: float

# < caminhos padrão (--catalog / --cappi-dir) >------------------------------------------

def caminho_catalogo_padrao() -> str:
    """
    padrão de --catalog: a variável de ambiente UNT_CATALOG_DB (definida pelo
    unt_cfg.sh) ou, sem ela, ./data/dbase/catalog.db.
    """
    return os.environ.get("UNT_CATALOG_DB", "./data/dbase/catalog.db")

# -------------------------------------------------------------------------------------
def diretorio_cappi_padrao() -> str:
    """
    padrão de --cappi-dir: a variável de ambiente UNT_CAPPI_DIR (definida pelo
    unt_cfg.sh) ou, sem ela, ./data/cappi.
    """
    return os.environ.get("UNT_CAPPI_DIR", "./data/cappi")

# < conexão >----------------------------------------------------------------------------

def abre_catalogo_somente_leitura(fs_caminho_catalogo: str) -> sqlite3.Connection:
    """
    abre o catálogo em modo somente leitura (mode=ro): mesmo um bug aqui não
    consegue alterar o catálogo do 27.nowcast.

    as linhas lidas se comportam como dicionários (lrow["radar"]).

    :param fs_caminho_catalogo: caminho do catalog.db

    :returns: conexão SQLite
    """
    lconexao = sqlite3.connect(f"file:{fs_caminho_catalogo}?mode=ro", uri=True,
                               timeout=CF_TIMEOUT_LOCK_SEGUNDOS)
    lconexao.row_factory = sqlite3.Row

    return lconexao

# < sequências >-------------------------------------------------------------------------

def _monta_filtros_opcionais(fs_preset: typing.Optional[str]=None,
                             fi_quality_maxima: typing.Optional[int]=None,
                             fi_target: typing.Optional[int]=None) -> tuple:
    """
    monta o trecho "AND ..." do WHERE com os filtros pedidos.

    cada coluna só aparece no SQL quando o filtro correspondente é usado:
    assim catálogos antigos, sem as colunas preset/quality, continuam
    funcionando quando nenhum filtro é pedido.

    :param fs_preset: só sequências deste preset (ex.: "nwc250v2")
    :param fi_quality_maxima: só sequências com quality <= este valor
                              (0 = só observados, 1 = com interpolado,
                              2 = com previsto)
    :param fi_target: só a sequência com este target (epoch UTC)

    :returns: (trecho_sql, parametros) -- ex.: (" AND preset = ?", ("nwc250v2",))
    """
    ls_trecho_sql = ""
    llst_parametros = []

    if fs_preset is not None:
        ls_trecho_sql += " AND preset = ?"
        llst_parametros.append(fs_preset)

    if fi_quality_maxima is not None:
        ls_trecho_sql += " AND quality <= ?"
        llst_parametros.append(fi_quality_maxima)

    if fi_target is not None:
        ls_trecho_sql += " AND target = ?"
        llst_parametros.append(fi_target)

    return ls_trecho_sql, tuple(llst_parametros)

# -------------------------------------------------------------------------------------
def _consulta_sequencias(fconexao: sqlite3.Connection, fs_radar: str, fi_alcance: int,
                         fs_preset: typing.Optional[str], fi_quality_maxima: typing.Optional[int],
                         fi_target: typing.Optional[int],
                         fi_limite: typing.Optional[int]) -> list:
    """
    consulta única de sequências, da mais recente para a mais antiga.

    :param fconexao: conexão com o catálogo
    :param fs_radar: código do radar (ex.: "CGU")
    :param fi_alcance: alcance em km (ex.: 250)
    :param fs_preset: filtro opcional (ver _monta_filtros_opcionais)
    :param fi_quality_maxima: filtro opcional
    :param fi_target: filtro opcional
    :param fi_limite: no máximo esta quantidade de linhas (None = todas)

    :returns: lista de dicionários (uma sequência por item)
    """
    ls_filtros_sql, lt_parametros_filtros = _monta_filtros_opcionais(fs_preset,
                                                                     fi_quality_maxima,
                                                                     fi_target)
    ls_sql = f"""SELECT *
                 FROM   sequences
                 WHERE  radar   = ?
                 AND    alcance = ?{ls_filtros_sql}
                 ORDER BY target DESC"""

    lt_parametros = (fs_radar, fi_alcance) + lt_parametros_filtros

    if fi_limite is not None:
        ls_sql += " LIMIT ?"
        lt_parametros += (fi_limite, )

    llst_linhas = fconexao.execute(ls_sql, lt_parametros).fetchall()

    return [dict(lrow) for lrow in llst_linhas]

# -------------------------------------------------------------------------------------
def lista_pares_radar_alcance(fconexao: sqlite3.Connection) -> list:
    """
    lista os pares (radar, alcance) que têm ao menos uma sequência,
    ordenados por radar e alcance.

    :param fconexao: conexão com o catálogo

    :returns: lista de tuplas, ex.: [("CGU", 250), ("CGU", 400)]
    """
    llst_linhas = fconexao.execute("""SELECT DISTINCT radar, alcance
                                      FROM   sequences
                                      ORDER BY radar, alcance""").fetchall()

    return [(lrow["radar"], lrow["alcance"]) for lrow in llst_linhas]

# -------------------------------------------------------------------------------------
def lista_sequencias(fconexao: sqlite3.Connection, fs_radar: str, fi_alcance: int,
                     fs_preset: typing.Optional[str]=None,
                     fi_quality_maxima: typing.Optional[int]=None) -> list:
    """
    lista todas as sequências do par (radar, alcance), da mais recente para a
    mais antiga.

    :param fconexao: conexão com o catálogo
    :param fs_radar: código do radar
    :param fi_alcance: alcance em km
    :param fs_preset: filtro opcional (ver _monta_filtros_opcionais)
    :param fi_quality_maxima: filtro opcional

    :returns: lista de dicionários (vazia se não houver nenhuma)
    """
    return _consulta_sequencias(fconexao, fs_radar, fi_alcance, fs_preset,
                                fi_quality_maxima, fi_target=None, fi_limite=None)

# -------------------------------------------------------------------------------------
def busca_sequencia_mais_recente(fconexao: sqlite3.Connection, fs_radar: str,
                                 fi_alcance: int,
                                 fs_preset: typing.Optional[str]=None,
                                 fi_quality_maxima: typing.Optional[int]=None,
                                 fi_target: typing.Optional[int]=None) -> typing.Optional[dict]:
    """
    busca a sequência de maior target do par (radar, alcance).

    com fi_target, busca exatamente a sequência daquele target (útil para
    reprocessar/verificar sequências antigas).

    :param fconexao: conexão com o catálogo
    :param fs_radar: código do radar
    :param fi_alcance: alcance em km
    :param fs_preset: filtro opcional (ver _monta_filtros_opcionais)
    :param fi_quality_maxima: filtro opcional
    :param fi_target: filtro opcional

    :returns: a sequência como dicionário, ou None se não houver nenhuma
    """
    llst_sequencias = _consulta_sequencias(fconexao, fs_radar, fi_alcance, fs_preset,
                                           fi_quality_maxima, fi_target, fi_limite=1)
    if not llst_sequencias:
        return None

    return llst_sequencias[0]

# -------------------------------------------------------------------------------------
def lista_timestamps_de_entrada(fdct_sequencia: dict) -> list:
    """
    devolve os timestamps (epoch UTC) dos CAPPIs de entrada da sequência.

    as colunas input_1..input_4 já estão em ordem cronológica (do mais
    antigo para o mais novo) -- não inverter.

    :param fdct_sequencia: uma linha da tabela sequences

    :returns: lista com CI_NUM_ENTRADAS timestamps
    """
    llst_timestamps = []

    for li_numero_entrada in range(1, CI_NUM_ENTRADAS + 1):
        llst_timestamps.append(fdct_sequencia[f"input_{li_numero_entrada}"])

    return llst_timestamps

# < produtos (janelas de treino) >-------------------------------------------------------

def preset_ativo_padrao(fi_alcance: int) -> typing.Optional[str]:
    """
    preset ativo do 27.nowcast para o alcance (nwc250v2, nwc400v2).

    :param fi_alcance: alcance em km

    :returns: id do preset, ou None para um alcance sem preset conhecido
    """
    return MDCT_PRESET_ATIVO.get(fi_alcance)

# -------------------------------------------------------------------------------------
def lista_produtos_prontos(fconexao: sqlite3.Connection, fs_radar: str, fi_alcance: int,
                           fs_preset: str) -> list:
    """
    lista os CAPPIs prontos (status DONE) do par (radar, alcance) em um preset, em ordem
    cronológica. É a matéria-prima das janelas de treino (unt_janelas).

    :param fconexao: conexão com o catálogo
    :param fs_radar: código do radar
    :param fi_alcance: alcance em km
    :param fs_preset: id do preset (ex.: "nwc250v2")

    :returns: lista de dicionários {"timestamp", "origin", "valid_fraction",
              "rain_fraction"} (vazia se não houver nenhum, ou se o catálogo
              é antigo e não tem a tabela cappi_products)
    """
    ls_sql = """SELECT timestamp, origin, valid_fraction, rain_fraction
                FROM   cappi_products
                WHERE  radar = ? AND alcance = ? AND preset = ? AND status = ?
                ORDER BY timestamp"""

    lt_parametros = (fs_radar, fi_alcance, fs_preset, CI_STATUS_PRONTO)

    try:
        llst_linhas = fconexao.execute(ls_sql, lt_parametros).fetchall()

    # catálogo antigo, sem a tabela cappi_products
    except sqlite3.OperationalError:
        return []

    return [dict(lrow) for lrow in llst_linhas]

# -------------------------------------------------------------------------------------
def lista_pares_com_produtos(fconexao: sqlite3.Connection) -> list:
    """
    lista os pares (radar, alcance) que têm ao menos um CAPPI pronto.

    :param fconexao: conexão com o catálogo

    :returns: lista de tuplas (radar, alcance), ordenada
    """
    try:
        llst_linhas = fconexao.execute("""SELECT DISTINCT radar, alcance
                                          FROM   cappi_products
                                          WHERE  status = ?
                                          ORDER BY radar, alcance""",
                                       (CI_STATUS_PRONTO, )).fetchall()

    # catálogo antigo, sem a tabela cappi_products
    except sqlite3.OperationalError:
        return []

    return [(lrow["radar"], lrow["alcance"]) for lrow in llst_linhas]

# < radar >------------------------------------------------------------------------------

def busca_posicao_radar(fconexao: sqlite3.Connection,
                        fs_radar: str) -> typing.Optional[PosicaoRadar]:
    """
    busca latitude, longitude e altitude do radar.

    :param fconexao: conexão com o catálogo
    :param fs_radar: código do radar

    :returns: PosicaoRadar, ou None se o 27.nowcast ainda não gravou este radar
    """
    lrow = fconexao.execute("""SELECT latitude, longitude, altitude
                               FROM   radar_sites
                               WHERE  radar = ?""", (fs_radar, )).fetchone()
    if lrow is None:
        return None

    return PosicaoRadar(lrow["latitude"], lrow["longitude"], lrow["altitude"])

# < arquivos CAPPI >---------------------------------------------------------------------

def monta_caminho_cappi(fs_diretorio_cappi: str, fs_radar: str, fi_alcance: int,
                        fi_timestamp: int,
                        fs_preset: typing.Optional[str]=None) -> pathlib.Path:
    """
    monta o caminho do CAPPI (.npy) pela convenção de nomes do 27.nowcast:

        com preset: <diretorio>/<preset>/<R>_<A>/<R>_<A>_<timestamp>.npy
        sem preset: <diretorio>/<R>_<A>_<timestamp>.npy   (catálogo antigo)

    o caminho é montado a partir de --cappi-dir (e não lido da coluna
    cappi_products.cappi_path) porque o caminho gravado pelo 27.nowcast pode
    ser de outra máquina/ponto de montagem.

    :param fs_diretorio_cappi: raiz dos CAPPIs (--cappi-dir/UNT_CAPPI_DIR)
    :param fs_radar: código do radar
    :param fi_alcance: alcance em km
    :param fi_timestamp: epoch UTC do volume
    :param fs_preset: sequences.preset (None/vazio = layout antigo, plano)

    :returns: caminho do arquivo (pode não existir)
    """
    ls_nome_arquivo = f"{fs_radar}_{fi_alcance}_{fi_timestamp}.npy"

    if not fs_preset:
        return pathlib.Path(fs_diretorio_cappi) / ls_nome_arquivo

    return (pathlib.Path(fs_diretorio_cappi) / fs_preset / f"{fs_radar}_{fi_alcance}" /
            ls_nome_arquivo)

# -------------------------------------------------------------------------------------
def busca_caminho_cappi_registrado(fconexao: sqlite3.Connection, fs_radar: str,
                                   fi_alcance: int, fi_timestamp: int,
                                   fs_preset: typing.Optional[str]=None) -> typing.Optional[str]:
    """
    devolve onde o 27.nowcast diz ter gravado o CAPPI (cappi_products.cappi_path).

    SÓ para diagnóstico: a leitura usa sempre monta_caminho_cappi(). Se os
    dois diferem na raiz, --cappi-dir/UNT_CAPPI_DIR está apontando para o
    lugar errado.

    :param fconexao: conexão com o catálogo
    :param fs_radar: código do radar
    :param fi_alcance: alcance em km
    :param fi_timestamp: epoch UTC do volume
    :param fs_preset: filtro opcional de preset

    :returns: o caminho registrado, ou None (sem linha, coluna nula ou
              catálogo antigo sem a tabela cappi_products)
    """
    ls_filtro_preset_sql, lt_parametros_preset = _monta_filtros_opcionais(fs_preset=fs_preset)

    ls_sql = f"""SELECT cappi_path
                 FROM   cappi_products
                 WHERE  radar = ? AND alcance = ? AND timestamp = ?{ls_filtro_preset_sql}"""

    lt_parametros = (fs_radar, fi_alcance, fi_timestamp) + lt_parametros_preset

    try:
        lrow = fconexao.execute(ls_sql, lt_parametros).fetchone()

    # catálogo antigo, sem a tabela cappi_products
    except sqlite3.OperationalError:
        return None

    if lrow is None:
        return None

    return lrow[0]

# -------------------------------------------------------------------------------------
def carrega_cappi(fpth_cappi: typing.Union[str, pathlib.Path]) -> np.ndarray:
    """
    carrega um CAPPI (.npy, dBZ) como float32.

    o "fora da cobertura" (-9999) vira NaN; o "sem eco" (-32) é mantido.

    :param fpth_cappi: caminho do .npy

    :returns: array 2D em dBZ
    """
    lnp_cappi_dbz = np.load(str(fpth_cappi)).astype(np.float32)
    lnp_cappi_dbz[lnp_cappi_dbz <= CF_DBZ_FORA_DA_COBERTURA] = np.nan

    return lnp_cappi_dbz

# -------------------------------------------------------------------------------------
def carrega_pilha_cappi(flst_caminhos: list) -> np.ndarray:
    """
    carrega vários CAPPIs e os empilha na ordem recebida.

    :param flst_caminhos: caminhos dos .npy (do mais antigo para o mais novo)

    :returns: array 3D (num_cappis, linhas, colunas) em dBZ
    """
    llst_cappis = []

    for lpth_cappi in flst_caminhos:
        llst_cappis.append(carrega_cappi(lpth_cappi))

    return np.stack(llst_cappis)

# -------------------------------------------------------------------------------------
def calcula_raio_da_grade_km(ft_shape: tuple, ff_km_por_pixel: float=CF_KM_POR_PIXEL) -> float:
    """
    calcula o meio-lado (km) da grade do CAPPI a partir do seu shape:
    500x500 -> 250 km, 600x600 -> 300 km.

    é este valor (e não o alcance nominal do radar) que define a área do
    mapa: o preset nwc400v2 tem grade de ±300 km, não ±400.

    :param ft_shape: shape do array (..., linhas, colunas)
    :param ff_km_por_pixel: resolução da grade

    :returns: raio em km
    """
    li_maior_lado_pixels = max(ft_shape[-2:])

    return li_maior_lado_pixels * ff_km_por_pixel / 2.0

# < the end >----------------------------------------------------------------------------
