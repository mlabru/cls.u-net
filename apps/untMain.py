# -*- coding: utf-8 -*-
"""
Centro Logístico de Simulação Meteorológica
Copyright (c) 1995-2026, Instituto de Controle do Espaço Aéreo

untMain

gera a previsão de curto prazo (nowcast) de UM par radar/alcance, com os
três backends de Deep Learning (pytorch, rainnet, smaat) rodando em
paralelo, a partir de CAPPIs já prontos (catálogo SQLite + .npy, gerados
pelo pipeline do 27.nowcast/04.dvl). Espelho do oflMain do 25.oflow: mesma
entrada, mesmos arquivos de saída e o mesmo verify.json, para que os
backends de optical flow e de Deep Learning sejam comparáveis.

fluxo de main():
  1. busca a sequência (4 CAPPIs de entrada + 1 target) no catálogo;
  2. confere que os 4 .npy existem e os carrega (do mais antigo ao mais novo);
  3. busca a posição do radar e calcula o raio do mapa pela grade;
  4. descarta os backends sem checkpoint (<modelos>/<backend>/atual.pt) e os
     cujo checkpoint não cobre este par radar/alcance (escopo do .json ao lado),
     grava <saida>/referencia.json (horário de referência, passo, ordem dos
     backends -- lido pelo untMakGIF para o GIF montado) e roda cada backend
     num processo próprio (multiprocessing): cada um grava
     <saida>/<backend>/<prefix>_NN.png e <prefix>.npy;
  5. com --verify, compara a previsão com o target observado e grava
     <saida>/verify.json (+ FSS).

os filhos herdam UNT_MODELOS_DIR (--modelos-dir), UNT_DEVICE (--device) e
UNT_THREADS (CPUs / backends, só em cpu). O processo pai nunca inicializa a
CUDA (só importa o torch), então o fork padrão do Linux é seguro.

código de saída: 0 se ao menos um backend concluiu; 1 caso contrário.

$ python untMain.py --radar CGU --alcance 400 [--catalog catalog.db]
                    [--cappi-dir ./data/cappi] [--modelos-dir ./data/modelos]
                    [--device auto] [--saida ./teste] [-p nowcast]
                    [--preset nwc400v2] [--quality-max 0] [--target EPOCH]
                    [--verify]

2026.out  mlabru  backends_treinados descarta o checkpoint cujo escopo (pares
                  radar/alcance do manifesto de treino) não cobre o par da
                  passada; escopo desconhecido (versão 1, sem .json) roda com
                  aviso.

2026.out  mlabru  portado de 25.oflow (oflMain): backends pytorch, rainnet e
                  smaat no lugar de opencv, pysteps e rainymotion;
                  --modelos-dir e --device; backend sem checkpoint é
                  descartado com aviso (backends_treinados).

2026.out  mlabru  grava <saida>/referencia.json para o GIF montado
                  (backends lado a lado) do untMakGIF.

2026.out  mlabru  main() quebrada em funções de uma responsabilidade;
                  registro explícito de backends (MDCT_MODULOS_BACKEND), todos
                  com a mesma entrada gera_nowcast(); tarefas e resultados do
                  Pool como NamedTuple; removido o parâmetro de depuração
                  nunca usado (fv_debug).

2026.out  mlabru  dados reais do 27.nowcast (ambiente de teste 03.dat/tst):
                  caminho do CAPPI com preset (sequences.preset), -9999 ->
                  NaN, passo de tempo derivado da sequência (10 min nas
                  séries reais, não mais 5 fixo), raio do mapa derivado da
                  grade (nwc400v2 é ±300 km, não 400), filtros
                  --preset/--quality-max/--target e verificação opcional
                  (--verify) contra o volume observado "target" da
                  sequência (unt_verify.py, grava <saida>/verify.json).
                  --mask e --shape removidos; --prefix nomeia os arquivos
                  de cada backend (padrão "nowcast").

2026.ago  mlabru  não lê mais arquivos .mvol brutos -- a conversão .mvol ->
                  CAPPI já é feita pelo pipeline do 27.nowcast, que cataloga
                  volumes/sequências prontas num SQLite (catalog.db). A
                  geometria do radar vem da tabela radar_sites. Sequência
                  de 4 CAPPIs de entrada.

2026.ago  mlabru  os três backends rodam em paralelo (um processo por
                  backend, via multiprocessing.Pool). Processos separados
                  evitam problemas de thread-safety do matplotlib/cartopy.
                  executa_backend() roda no processo filho e nunca propaga
                  exceção.

2026.jun  mlabru  passa a chamar os três backends de nowcast (pysteps,
                  rainymotion, opencv), cada um no seu subdiretório de
                  <saida>. Import de cada backend é opcional; falha de UM
                  backend não impede os demais.

2025.nov  mlabru  initial version (Linux/Python)
"""
# < imports >----------------------------------------------------------------------------

# python library
import argparse
import importlib
import logging
import multiprocessing as mp
import os
import pathlib
import sqlite3
import sys
import typing

# matplotlib sem tela (Agg) -- os frames só são salvos em arquivo. Precisa
# vir ANTES de importar os backends (eles importam pyplot). Com DISPLAY
# definido o matplotlib escolhia QtAgg, e o Qt (plugins do cv2) abortava
# dentro dos processos do Pool: o worker morria e o Pool esperava para
# sempre, até o timeout do shell. Um MPLBACKEND explícito ainda prevalece
# (ex.: MPLBACKEND=QtAgg para depurar com janela).
os.environ.setdefault("MPLBACKEND", "Agg")

# numPy
import numpy as np

# model
import work.unt_backend_comum as obc
import work.unt_catalog as oc
import work.unt_inferencia as ui
import work.unt_modelos as um
import work.unt_verify as ov

# < constants >--------------------------------------------------------------------------

# backends de nowcast: nome -> módulo Python. Todo módulo listado aqui
# expõe gera_nowcast() (ver unt_backend_comum). A ordem é a ordem do log.
# Ao adicionar um backend, incluí-lo também em UNT_BACKENDS (unt_cfg.sh).
MDCT_MODULOS_BACKEND = {
    "pytorch": "work.unt_pytorch",
    "rainnet": "work.unt_rainnet",
    "smaat": "work.unt_smaat",
}

# número de passos previstos por backend (6 x 10 min = 1 hora)
CI_NUM_PASSOS_PREVISAO = 6

# dados da execução para o GIF montado (lido pelo untMakGIF)
CS_ARQUIVO_REFERENCIA = "referencia.json"

# < logging >----------------------------------------------------------------------------

# logger
M_LOG = logging.getLogger(__name__)
M_LOG.setLevel(logging.INFO)

# < tipos >------------------------------------------------------------------------------

class TarefaBackend(typing.NamedTuple):
    """
    tudo o que um processo filho precisa para rodar um backend.
    """
    nome_backend: str
    refletividade_dbz: np.ndarray
    radar: obc.GeometriaRadar
    num_passos: int
    passo_minutos: int
    prefixo_saida: str


class ResultadoBackend(typing.NamedTuple):
    """
    o que um processo filho devolve (nunca uma exceção).
    """
    nome_backend: str
    sucesso: bool
    mensagem_erro: typing.Optional[str]

# < backends disponíveis >---------------------------------------------------------------

def _lista_backends_disponiveis() -> list:
    """
    tenta importar cada backend de MDCT_MODULOS_BACKEND. Se a biblioteca de
    um deles (o torch) não estiver instalada nesta máquina, ele é pulado com
    um aviso -- os demais continuam funcionando.

    :returns: nomes dos backends importáveis, na ordem de MDCT_MODULOS_BACKEND
    """
    llst_disponiveis = []

    for ls_nome_backend, ls_modulo in MDCT_MODULOS_BACKEND.items():
        try:
            importlib.import_module(ls_modulo)

        except ImportError as le_erro:
            M_LOG.warning(f"[!] {ls_modulo} indisponível: {le_erro}")
            continue

        llst_disponiveis.append(ls_nome_backend)

    return llst_disponiveis


# calculado uma vez, ao importar este módulo
MLST_BACKENDS_DISPONIVEIS = _lista_backends_disponiveis()

# -------------------------------------------------------------------------------------
def _checkpoint_cobre_par(fs_nome_backend: str, fpth_checkpoint: pathlib.Path,
                          fs_radar: typing.Optional[str], fi_alcance: typing.Optional[int]) -> bool:
    """
    o atual.pt se aplica ao par? Lê só o .json ao lado (sem carregar os pesos).
    Escopo desconhecido roda, com aviso.
    """
    if fs_radar is None or fi_alcance is None:
        return True

    lb_cobre = um.cobre_par(um.le_resumo_checkpoint(fpth_checkpoint), fs_radar, fi_alcance)

    if lb_cobre is None:
        M_LOG.warning(f"[!] {fs_nome_backend}: escopo de {fpth_checkpoint} desconhecido "
                      f"(checkpoint antigo ou sem .json); rodando em {fs_radar}_{fi_alcance} "
                      f"assim mesmo.")
        return True

    if not lb_cobre:
        M_LOG.warning(f"[!] {fs_nome_backend}: {fpth_checkpoint.resolve().name} não foi "
                      f"treinado para {fs_radar}_{fi_alcance}; pulado.")

    return lb_cobre

# -------------------------------------------------------------------------------------
def backends_treinados(fs_modelos_dir: str, fs_radar: typing.Optional[str]=None,
                       fi_alcance: typing.Optional[int]=None) -> list:
    """
    dos backends importáveis, os que têm checkpoint (<modelos>/<backend>/atual.pt)
    aplicável ao par radar/alcance. Um backend ainda não treinado, ou treinado para
    outros pares, é pulado com aviso.

    :param fs_modelos_dir: raiz dos modelos (--modelos-dir)
    :param fs_radar: radar da passada (None = não filtra pelo escopo)
    :param fi_alcance: alcance da passada (km)

    :returns: nomes, na ordem de MDCT_MODULOS_BACKEND
    """
    llst_treinados = []

    for ls_nome_backend in MLST_BACKENDS_DISPONIVEIS:
        lpth_checkpoint = um.caminho_checkpoint_atual(fs_modelos_dir, ls_nome_backend)

        if not lpth_checkpoint.exists():
            M_LOG.warning(f"[!] {ls_nome_backend}: sem checkpoint ({lpth_checkpoint}); "
                          f"treine com untTrain.py -b {ls_nome_backend}.")
            continue

        if not _checkpoint_cobre_par(ls_nome_backend, lpth_checkpoint, fs_radar, fi_alcance):
            continue

        llst_treinados.append(ls_nome_backend)

    return llst_treinados

# -------------------------------------------------------------------------------------
def exporta_ambiente_dos_filhos(fargs: argparse.Namespace, fi_num_backends: int) -> None:
    """
    o que os processos filhos (gera_nowcast) leem do ambiente: raiz dos modelos,
    device e threads do torch em cpu (CPUs divididas entre os backends).
    """
    os.environ["UNT_MODELOS_DIR"] = fargs.modelos_dir
    os.environ["UNT_DEVICE"] = fargs.device
    os.environ["UNT_THREADS"] = str(ui.threads_por_backend(fi_num_backends))

# < execução de um backend (processo filho) >--------------------------------------------

def _grava_previsao(fnp_previsao_dbz: typing.Optional[np.ndarray], fs_prefixo_saida: str) -> None:
    """
    grava a previsão em dBZ em <prefixo>.npy, para a verificação no
    processo pai -- por arquivo, para não mandar arrays grandes de volta
    pelo Pool. None (backend não viu chuva) não grava nada.
    """
    if fnp_previsao_dbz is None:
        return

    np.save(f"{fs_prefixo_saida}.npy", np.asarray(fnp_previsao_dbz, dtype=np.float32))

# -------------------------------------------------------------------------------------
def executa_backend(ft_tarefa: TarefaBackend) -> ResultadoBackend:
    """
    roda um backend (chamada pelo Pool, dentro do processo filho).

    o módulo do backend é importado aqui dentro, e não herdado do processo
    pai, para não depender de como o multiprocessing cria os filhos
    (fork/spawn).

    NUNCA propaga exceção: qualquer erro vira ResultadoBackend(sucesso=False),
    para que o Pool sempre complete todas as tarefas e o pai decida o que
    logar.

    :param ft_tarefa: entrada do backend

    :returns: nome, sucesso e mensagem de erro (None se sucesso)
    """
    ls_nome_backend = ft_tarefa.nome_backend

    try:
        if ls_nome_backend not in MDCT_MODULOS_BACKEND:
            raise ValueError(f"backend desconhecido: {ls_nome_backend}")

        lmodulo_backend = importlib.import_module(MDCT_MODULOS_BACKEND[ls_nome_backend])

        # diretório <saida>/<backend>/
        pathlib.Path(ft_tarefa.prefixo_saida).parent.mkdir(parents=True, exist_ok=True)

        lresultado = lmodulo_backend.gera_nowcast(ft_tarefa.refletividade_dbz, ft_tarefa.radar,
                                                  fi_num_passos=ft_tarefa.num_passos,
                                                  fi_passo_minutos=ft_tarefa.passo_minutos,
                                                  fs_prefixo_saida=ft_tarefa.prefixo_saida)

        _grava_previsao(lresultado.previsao_dbz, ft_tarefa.prefixo_saida)

        return ResultadoBackend(ls_nome_backend, True, None)

    # falha contida: não derruba os demais processos do Pool
    except Exception as le_erro:
        return ResultadoBackend(ls_nome_backend, False, str(le_erro))

# < argumentos >-------------------------------------------------------------------------

def arg_parse() -> argparse.Namespace:
    """
    lê os argumentos da linha de comando.

    --catalog e --cappi-dir têm como padrão as variáveis de ambiente
    UNT_CATALOG_DB e UNT_CAPPI_DIR (ver unt_catalog).

    :returns: argumentos lidos
    """
    lparser = argparse.ArgumentParser(
        description="Gera previsão de curto prazo (nowcast via Deep Learning: "
                    "pytorch, rainnet e smaat) a partir da sequência mais "
                    "recente catalogada para um radar/alcance")

    lparser.add_argument("-r", "--radar", type=str, required=True,
                         help="código do site de radar (ex: CGU)")
    lparser.add_argument("-a", "--alcance", type=int, required=True,
                         help="alcance em km (ex: 400)")
    lparser.add_argument("--catalog", type=str, default=oc.caminho_catalogo_padrao(),
                         help="caminho do catálogo SQLite")
    lparser.add_argument("--cappi-dir", type=str, default=oc.diretorio_cappi_padrao(),
                         help="diretório dos CAPPI (.npy)")
    lparser.add_argument("--modelos-dir", type=str, default=um.diretorio_modelos_padrao(),
                         help="raiz dos checkpoints (<dir>/<backend>/atual.pt)")
    lparser.add_argument("--device", type=str, default=os.environ.get("UNT_DEVICE", "auto"),
                         help="auto, cpu, cuda, cuda:N ou mps")
    lparser.add_argument("-p", "--prefix", type=str, default="nowcast",
                         help="prefixo dos frames (<prefix>_NN.png) e da previsão "
                              "(<prefix>.npy) de cada backend")
    lparser.add_argument("-s", "--saida", type=str, default="./data/cache/nowcast",
                         help="diretório de saída dos plots")
    lparser.add_argument("--preset", type=str, default=None,
                         help="preset do CAPPI (ex: nwc250v2); se omitido, qualquer um")
    lparser.add_argument("--quality-max", type=int, default=None,
                         help="quality máximo da sequência "
                              "(0 = só observados, 1 = com interpolado)")
    lparser.add_argument("--target", type=int, default=None,
                         help="target (epoch UTC) da sequência; se omitido, a mais recente")
    lparser.add_argument("--verify", action="store_true",
                         help="verifica o nowcast contra o volume observado (target) "
                              "e grava verify.json")

    return lparser.parse_args()

# < passos de main() >-------------------------------------------------------------------

def busca_sequencia(fconexao: sqlite3.Connection,
                    fargs: argparse.Namespace) -> typing.Optional[dict]:
    """
    passo 1: busca a sequência a prever (a mais recente, ou a de --target).

    :param fconexao: conexão com o catálogo
    :param fargs: argumentos da linha de comando

    :returns: a sequência, ou None (já logado) se não houver
    """
    ldct_sequencia = oc.busca_sequencia_mais_recente(fconexao, fargs.radar, fargs.alcance,
                                                     fs_preset=fargs.preset,
                                                     fi_quality_maxima=fargs.quality_max,
                                                     fi_target=fargs.target)
    if ldct_sequencia is None:
        M_LOG.error(f"[x] nenhuma sequência catalogada para "
                    f"radar={fargs.radar} alcance={fargs.alcance} "
                    f"preset={fargs.preset} quality<={fargs.quality_max}.")

    return ldct_sequencia

# -------------------------------------------------------------------------------------
def _loga_cappi_ausente(fconexao: sqlite3.Connection, fargs: argparse.Namespace,
                        fi_timestamp: int, fs_preset: typing.Optional[str],
                        fpth_procurado: pathlib.Path) -> None:
    """
    explica no log por que um CAPPI não foi encontrado: o caminho procurado
    e, se o catálogo souber, onde o 27.nowcast diz tê-lo gravado. Raízes
    diferentes indicam --cappi-dir/UNT_CAPPI_DIR apontando para o lugar
    errado.
    """
    M_LOG.error(f"[x] CAPPI ausente para radar={fargs.radar} "
                f"alcance={fargs.alcance} timestamp={fi_timestamp}: "
                f"{fpth_procurado} (--cappi-dir={fargs.cappi_dir}).")

    ls_caminho_registrado = oc.busca_caminho_cappi_registrado(fconexao, fargs.radar,
                                                              fargs.alcance, fi_timestamp,
                                                              fs_preset)
    if not ls_caminho_registrado:
        return

    if pathlib.Path(ls_caminho_registrado).exists():
        ls_situacao = "existe"
    else:
        ls_situacao = "também ausente"

    M_LOG.error(f"[x] o catálogo registra este CAPPI em {ls_caminho_registrado} "
                f"({ls_situacao} nesta máquina).")

# -------------------------------------------------------------------------------------
def resolve_caminhos_cappi(fconexao: sqlite3.Connection, fargs: argparse.Namespace,
                           fdct_sequencia: dict) -> typing.Optional[list]:
    """
    passo 2: monta o caminho de cada CAPPI de entrada e confere que existe.

    :param fconexao: conexão com o catálogo (só para o diagnóstico)
    :param fargs: argumentos da linha de comando
    :param fdct_sequencia: a sequência (passo 1)

    :returns: caminhos do mais antigo ao mais novo, ou None (já logado) se
              faltar algum
    """
    ls_preset = fdct_sequencia.get("preset")

    llst_caminhos = []

    for li_timestamp in oc.lista_timestamps_de_entrada(fdct_sequencia):
        lpth_cappi = oc.monta_caminho_cappi(fargs.cappi_dir, fargs.radar, fargs.alcance,
                                            li_timestamp, ls_preset)
        if not lpth_cappi.exists():
            _loga_cappi_ausente(fconexao, fargs, li_timestamp, ls_preset, lpth_cappi)
            return None

        llst_caminhos.append(lpth_cappi)

    return llst_caminhos

# -------------------------------------------------------------------------------------
def monta_geometria_radar(fconexao: sqlite3.Connection, fs_radar: str,
                          ft_shape_pilha: tuple) -> typing.Optional[obc.GeometriaRadar]:
    """
    passo 3: posição do radar (catálogo) + raio do mapa (pela grade do CAPPI,
    não pelo alcance nominal: o nwc400v2 tem grade de ±300 km).

    :param fconexao: conexão com o catálogo
    :param fs_radar: código do radar
    :param ft_shape_pilha: shape da pilha de CAPPIs

    :returns: GeometriaRadar, ou None (já logado) se o radar não estiver em
              radar_sites
    """
    lt_posicao = oc.busca_posicao_radar(fconexao, fs_radar)

    if lt_posicao is None:
        M_LOG.error(f"[x] geometria não catalogada para radar={fs_radar} "
                    f"(tabela radar_sites).")
        return None

    return obc.GeometriaRadar(latitude=lt_posicao.latitude,
                              longitude=lt_posicao.longitude,
                              altitude_m=lt_posicao.altitude_m,
                              raio_mapa_km=oc.calcula_raio_da_grade_km(ft_shape_pilha))

# -------------------------------------------------------------------------------------
def calcula_passo_minutos(flst_timestamps: list) -> int:
    """
    intervalo (minutos inteiros) entre os dois últimos CAPPIs de entrada --
    é também o passo da previsão. Nas séries do 27.nowcast é ~10 min (o
    radar oscila entre ~586 e ~622 s).

    :param flst_timestamps: timestamps de entrada (epoch UTC), em ordem

    :returns: passo em minutos (no mínimo 1)
    """
    li_segundos = flst_timestamps[-1] - flst_timestamps[-2]

    li_minutos = int(round(li_segundos / 60.0))

    return max(1, li_minutos)

# -------------------------------------------------------------------------------------
def monta_tarefas(fargs: argparse.Namespace, fnp_pilha_dbz: np.ndarray,
                  ft_radar: obc.GeometriaRadar, fi_passo_minutos: int,
                  flst_backends: list) -> list:
    """
    passo 4a: uma tarefa por backend treinado. Entradas idênticas; saída
    de cada um no seu subdiretório <saida>/<backend>/, para que frames de
    backends diferentes nunca se misturem no mesmo GIF.

    :returns: lista de TarefaBackend
    """
    llst_tarefas = []

    for ls_nome_backend in flst_backends:
        lpth_prefixo = pathlib.Path(fargs.saida) / ls_nome_backend / fargs.prefix

        llst_tarefas.append(TarefaBackend(nome_backend=ls_nome_backend,
                                          refletividade_dbz=fnp_pilha_dbz,
                                          radar=ft_radar,
                                          num_passos=CI_NUM_PASSOS_PREVISAO,
                                          passo_minutos=fi_passo_minutos,
                                          prefixo_saida=str(lpth_prefixo)))
    return llst_tarefas

# -------------------------------------------------------------------------------------
def executa_backends_em_paralelo(flst_tarefas: list) -> list:
    """
    passo 4b: um processo por backend (processos, não threads: matplotlib e
    cartopy não são thread-safe, e um processo isola a falha do outro --
    inclusive falta de memória na GPU).

    :param flst_tarefas: lista de TarefaBackend

    :returns: lista de ResultadoBackend, na mesma ordem
    """
    llst_nomes = [lt_tarefa.nome_backend for lt_tarefa in flst_tarefas]
    M_LOG.info(f"[+] executando {len(flst_tarefas)} backend(s) em paralelo: "
               f"{', '.join(llst_nomes)}...")

    with mp.Pool(processes=len(flst_tarefas)) as lpool:
        return lpool.map(executa_backend, flst_tarefas)

# -------------------------------------------------------------------------------------
def registra_resultados(flst_resultados: list) -> list:
    """
    loga o resultado de cada backend.

    :param flst_resultados: lista de ResultadoBackend

    :returns: nomes dos backends que concluíram com sucesso
    """
    llst_backends_ok = []

    for lt_resultado in flst_resultados:
        if lt_resultado.sucesso:
            M_LOG.info(f"[✓] nowcast via {lt_resultado.nome_backend} concluído.")
            llst_backends_ok.append(lt_resultado.nome_backend)
        else:
            M_LOG.error(f"[x] nowcast via {lt_resultado.nome_backend} falhou: "
                        f"{lt_resultado.mensagem_erro}")

    return llst_backends_ok

# -------------------------------------------------------------------------------------
def grava_referencia(fargs: argparse.Namespace, fdct_sequencia: dict,
                     fi_passo_minutos: int, flst_tarefas: list) -> None:
    """
    passo 4 (antes dos backends): grava <saida>/referencia.json, com o que o
    untMakGIF precisa para carimbar o GIF montado: horário de referência
    (última entrada observada, epoch UTC), passo e a ordem dos backends.
    Falha aqui nunca derruba o nowcast (o GIF montado sai sem horário).

    :param fargs: argumentos da linha de comando
    :param fdct_sequencia: a sequência
    :param fi_passo_minutos: passo da previsão
    :param flst_tarefas: lista de TarefaBackend
    """
    llst_backends = [lt_tarefa.nome_backend for lt_tarefa in flst_tarefas]

    ldct_referencia = {"radar": fargs.radar,
                       "alcance": fargs.alcance,
                       "referencia": fdct_sequencia["input_4"],
                       "target": fdct_sequencia["target"],
                       "timestep_min": fi_passo_minutos,
                       "num_passos": CI_NUM_PASSOS_PREVISAO,
                       "backends": llst_backends}

    lpth_json = pathlib.Path(fargs.saida) / CS_ARQUIVO_REFERENCIA

    try:
        lpth_json.parent.mkdir(parents=True, exist_ok=True)
        ov.grava_json(ldct_referencia, lpth_json)

    except OSError as le_erro:
        M_LOG.warning(f"[!] não gravou {lpth_json}: {le_erro}")

# < verificação (--verify) >-------------------------------------------------------------

def _carrega_previsoes_gravadas(fs_saida: str, fs_prefixo: str,
                                flst_backends_ok: list) -> dict:
    """
    lê o <saida>/<backend>/<prefixo>.npy de cada backend que concluiu.

    :returns: nome do backend -> previsão em dBZ (backends sem arquivo, que
              não viram chuva, ficam de fora)
    """
    ldct_previsoes = {}

    for ls_nome_backend in flst_backends_ok:
        lpth_previsao = pathlib.Path(fs_saida) / ls_nome_backend / f"{fs_prefixo}.npy"

        if not lpth_previsao.exists():
            M_LOG.warning(f"[!] verificação: {ls_nome_backend} sem previsão (sem chuva?).")
            continue

        ldct_previsoes[ls_nome_backend] = np.load(lpth_previsao)

    return ldct_previsoes

# -------------------------------------------------------------------------------------
def verifica_contra_target(fargs: argparse.Namespace, fdct_sequencia: dict,
                           fnp_ultima_entrada_dbz: np.ndarray, fi_passo_minutos: int,
                           flst_backends_ok: list) -> bool:
    """
    passo 5 (--verify): compara o passo previsto que cai no horário do target
    com o CAPPI observado do target, para cada backend e para a persistência,
    e grava <saida>/verify.json. Falha aqui nunca derruba o nowcast.

    :param fargs: argumentos da linha de comando
    :param fdct_sequencia: a sequência
    :param fnp_ultima_entrada_dbz: último CAPPI de entrada (persistência)
    :param fi_passo_minutos: passo da previsão
    :param flst_backends_ok: backends que concluíram

    :returns: True se a verificação foi gravada
    """
    ls_preset = fdct_sequencia.get("preset")

    lpth_observado = oc.monta_caminho_cappi(fargs.cappi_dir, fargs.radar, fargs.alcance,
                                            fdct_sequencia["target"], ls_preset)
    if not lpth_observado.exists():
        M_LOG.warning(f"[!] verificação: CAPPI do target ausente ({lpth_observado.name}).")
        return False

    ldct_previsoes = _carrega_previsoes_gravadas(fargs.saida, fargs.prefix, flst_backends_ok)

    li_passo = ov.calcula_passo_do_target(fdct_sequencia["input_4"], fdct_sequencia["target"],
                                          fi_passo_minutos)

    ldct_resultado = ov.verifica_previsoes(ldct_previsoes, fnp_ultima_entrada_dbz,
                                           oc.carrega_cappi(lpth_observado), li_passo)

    # dados da execução, para identificar a linha no histórico (.jsonl)
    ldct_resultado.update({"radar": fargs.radar,
                           "alcance": fargs.alcance,
                           "preset": ls_preset,
                           "quality": fdct_sequencia.get("quality"),
                           "target": fdct_sequencia["target"],
                           "timestep_min": fi_passo_minutos})

    ov.grava_json(ldct_resultado, pathlib.Path(fargs.saida) / "verify.json")

    for ls_linha in ov.formata_resumo(ldct_resultado):
        M_LOG.info(f"[=] {ls_linha}")

    return True

# < main >-------------------------------------------------------------------------------

def main() -> int:
    """
    gera o nowcast de um par radar/alcance (ver o fluxo no topo do módulo).

    :returns: 0 se ao menos um backend concluiu, 1 caso contrário
    """
    largs = arg_parse()

    lconexao = oc.abre_catalogo_somente_leitura(largs.catalog)

    # 1. sequência
    ldct_sequencia = busca_sequencia(lconexao, largs)

    if ldct_sequencia is None:
        return 1

    # 2. CAPPIs de entrada
    llst_caminhos = resolve_caminhos_cappi(lconexao, largs, ldct_sequencia)

    if llst_caminhos is None:
        return 1

    lnp_pilha_dbz = oc.carrega_pilha_cappi(llst_caminhos)

    # 3. radar
    lt_radar = monta_geometria_radar(lconexao, largs.radar, lnp_pilha_dbz.shape)

    if lt_radar is None:
        return 1

    # 4. backends
    if not MLST_BACKENDS_DISPONIVEIS:
        M_LOG.error("[x] nenhum backend de Deep Learning disponível "
                    "(sem torch nesta máquina?).")
        return 1

    llst_backends = backends_treinados(largs.modelos_dir, largs.radar, largs.alcance)

    if not llst_backends:
        M_LOG.error(f"[x] nenhum backend treinado para {largs.radar}_{largs.alcance} em "
                    f"{largs.modelos_dir} (<backend>/atual.pt).")
        return 1

    exporta_ambiente_dos_filhos(largs, len(llst_backends))

    li_passo_minutos = calcula_passo_minutos(oc.lista_timestamps_de_entrada(ldct_sequencia))

    llst_tarefas = monta_tarefas(largs, lnp_pilha_dbz, lt_radar, li_passo_minutos,
                                 llst_backends)
    grava_referencia(largs, ldct_sequencia, li_passo_minutos, llst_tarefas)

    llst_resultados = executa_backends_em_paralelo(llst_tarefas)
    llst_backends_ok = registra_resultados(llst_resultados)

    # 5. verificação
    if largs.verify:
        verifica_contra_target(largs, ldct_sequencia, lnp_pilha_dbz[-1], li_passo_minutos,
                               llst_backends_ok)

    M_LOG.info("[✓] processamento concluído.\n")

    if not llst_backends_ok:
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
