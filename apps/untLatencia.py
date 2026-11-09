# -*- coding: utf-8 -*-
"""
Centro Logístico de Simulação Meteorológica
Copyright (c) 1995-2026, Instituto de Controle do Espaço Aéreo

untLatencia

orçamento de latência da inferência NA MÁQUINA OPERACIONAL: mede quanto cada backend
leva para prever os 6 passos (preve_dbz, grade inteira) em cada largura_base e grade,
nas mesmas condições do untMain (um processo por backend, CPUs / backends threads), e
estima se o ciclo do cron cabe. O tempo não depende dos pesos: as redes são montadas
sem treino, a partir do config.ini -- dá para escolher a largura_base de produção antes
de treinar.

estimativas (CPU-bound, sem medir contenção real):
  - um par: os backends rodam em paralelo -> o mais lento dos backends;
  - par sob o cron: com UNT_MAX_PARES_EM_PARALELO pares ao mesmo tempo, as CPUs são
    divididas -> par x pares_em_paralelo (comparar com UNT_TIMEOUT_SECS);
  - passada: todos os pares -> par x número de pares (comparar com o intervalo do cron).

código de saída: 0 se mediu; 1 se nenhuma combinação pôde ser medida.

$ python untLatencia.py [--backends pytorch,rainnet,smaat] [--larguras 16,32,64]
                        [--grades 500,600] [--pares 4] [--pares-em-paralelo 3]
                        [--timeout-s 300] [--ciclo-s 300] [--repeticoes 3]
                        [--device cpu] [--config config.ini] [--saida latencia.json]

2026.out  mlabru  initial version (Linux/Python)
"""
# < imports >----------------------------------------------------------------------------

# python library
import argparse
import json
import logging
import pathlib
import statistics
import sys
import time
import typing

# numPy
import numpy as np

# pyTorch
import torch

# model
import work.unt_config as uc
import work.unt_device as ue
import work.unt_inferencia as ui
import work.unt_modelos as um

# < constants >--------------------------------------------------------------------------

# entradas e passos do nowcast (os do manifesto padrão)
CI_NUM_ENTRADAS = 4
CI_NUM_PASSOS = 6

# < logging >----------------------------------------------------------------------------

# logger
M_LOG = logging.getLogger(__name__)
M_LOG.setLevel(logging.INFO)

# < argumentos >-------------------------------------------------------------------------

def _lista_inteiros(fs_valor: str) -> list:
    return [int(ls) for ls in fs_valor.split(",") if ls.strip()]


def arg_parse(flst_argv: typing.Optional[list]=None) -> argparse.Namespace:
    """
    lê os argumentos da linha de comando.

    :param flst_argv: argumentos (None = sys.argv)

    :returns: argumentos lidos
    """
    lparser = argparse.ArgumentParser(
        description="Mede a latência da inferência por backend, largura e grade, e "
                    "estima o ciclo do cron")

    lparser.add_argument("--backends", type=str, default=",".join(um.CT_BACKENDS))
    lparser.add_argument("--larguras", type=_lista_inteiros, default=[16, 32, 64],
                         help="largura_base a medir (ex.: 16,32,64)")
    lparser.add_argument("--grades", type=_lista_inteiros, default=[500, 600],
                         help="lados da grade (500 = nwc250v2, 600 = nwc400v2)")
    lparser.add_argument("--pares", type=int, default=1,
                         help="pares radar/alcance operados (estimativa da passada)")
    lparser.add_argument("--pares-em-paralelo", type=int, default=3,
                         help="UNT_MAX_PARES_EM_PARALELO do unt_cfg.sh")
    lparser.add_argument("--timeout-s", type=float, default=300.0,
                         help="UNT_TIMEOUT_SECS (limite de um par)")
    lparser.add_argument("--ciclo-s", type=float, default=300.0,
                         help="intervalo do cron (s)")
    lparser.add_argument("--repeticoes", type=int, default=3)
    lparser.add_argument("--device", type=str, default="cpu")
    lparser.add_argument("--config", type=str, default=None,
                         help="config.ini; padrão ./config.ini se existir")
    lparser.add_argument("--saida", type=str, default=None, help="grava o resultado em JSON")

    return lparser.parse_args(flst_argv)

# < medição >----------------------------------------------------------------------------

def entrada_sintetica(fi_lado: int) -> np.ndarray:
    """
    (4, lado, lado) em dBZ: um núcleo de chuva no meio, sem eco no resto, e os cantos
    fora da cobertura (NaN), como um CAPPI real.
    """
    lnp_y, lnp_x = np.mgrid[:fi_lado, :fi_lado] - fi_lado / 2.0
    lnp_r = np.hypot(lnp_y, lnp_x)

    lnp_campo = np.where(lnp_r < fi_lado / 8.0, 40.0, -32.0).astype(np.float32)
    lnp_campo[lnp_r > fi_lado / 2.0] = np.nan

    return np.repeat(lnp_campo[None], CI_NUM_ENTRADAS, axis=0)


# -------------------------------------------------------------------------------------
def modelo_sem_treino(fconfig: uc.SConfig, fs_backend: str, fi_largura: int,
                      fdevice: torch.device) -> um.SModeloCarregado:
    """
    a rede do backend, pesos aleatórios, em modo eval no device.
    """
    ldct_hparams = um.hiperparametros(fconfig, fs_backend, CI_NUM_ENTRADAS, CI_NUM_PASSOS,
                                      {"largura_base": fi_largura})
    lrede = um.constroi_rede(ldct_hparams).to(fdevice).eval()

    return um.SModeloCarregado(fs_backend, lrede, um.transformacao_padrao(ldct_hparams),
                               ldct_hparams, {})

# -------------------------------------------------------------------------------------
def mede(fmodelo: um.SModeloCarregado, fnp_entrada: np.ndarray, fi_repeticoes: int) -> float:
    """
    mediana (s) de fi_repeticoes previsões de 6 passos, depois de um aquecimento.
    """
    ui.preve_dbz(fmodelo, fnp_entrada, CI_NUM_PASSOS)

    llst_tempos = []
    for _ in range(max(1, fi_repeticoes)):
        lf_inicio = time.perf_counter()
        ui.preve_dbz(fmodelo, fnp_entrada, CI_NUM_PASSOS)
        llst_tempos.append(time.perf_counter() - lf_inicio)

    return statistics.median(llst_tempos)

# -------------------------------------------------------------------------------------
def mede_tudo(fargs: argparse.Namespace, flst_backends: list) -> list:
    """
    uma medição por (backend, largura, grade), com as threads que o untMain daria a
    cada backend.

    :returns: [{"backend", "largura_base", "grade", "parametros", "segundos"}]
    """
    lconfig = uc.carrega(fargs.config)
    ldevice = ue.escolhe_device(fargs.device)
    li_threads = ui.threads_por_backend(len(flst_backends))
    ue.configura_threads_cpu(ldevice, li_threads)

    M_LOG.info(f"[+] device {ue.descreve(ldevice)}, {li_threads} threads por backend "
               f"({len(flst_backends)} backends em paralelo).")

    llst_medidas = []

    for li_largura in fargs.larguras:
        for ls_backend in flst_backends:
            lmodelo = modelo_sem_treino(lconfig, ls_backend, li_largura, ldevice)

            for li_grade in fargs.grades:
                lf_segundos = mede(lmodelo, entrada_sintetica(li_grade), fargs.repeticoes)

                llst_medidas.append({"backend": ls_backend, "largura_base": li_largura,
                                     "grade": li_grade,
                                     "parametros": um.conta_parametros(lmodelo.rede),
                                     "segundos": lf_segundos})
                M_LOG.info(f"[=] {ls_backend:<8} largura {li_largura:>3}  grade {li_grade}  "
                           f"{lf_segundos:7.2f} s")

    return llst_medidas

# < estimativas >------------------------------------------------------------------------

def estima(flst_medidas: list, fargs: argparse.Namespace) -> list:
    """
    por (largura, grade): par (backend mais lento), par sob o cron e passada inteira,
    comparados com o timeout e o ciclo.
    """
    ldct_por_combinacao = {}

    for ldct in flst_medidas:
        lt_chave = (ldct["largura_base"], ldct["grade"])
        ldct_por_combinacao.setdefault(lt_chave, []).append(ldct["segundos"])

    llst_estimativas = []

    for (li_largura, li_grade), llst_segundos in sorted(ldct_por_combinacao.items()):
        lf_par = max(llst_segundos)
        lf_par_no_cron = lf_par * max(1, fargs.pares_em_paralelo)
        lf_passada = lf_par * max(1, fargs.pares)

        llst_estimativas.append({"largura_base": li_largura, "grade": li_grade,
                                 "par_s": lf_par, "par_no_cron_s": lf_par_no_cron,
                                 "passada_s": lf_passada,
                                 "cabe_no_timeout": lf_par_no_cron <= fargs.timeout_s,
                                 "cabe_no_ciclo": lf_passada <= fargs.ciclo_s})

    return llst_estimativas

# -------------------------------------------------------------------------------------
def loga_estimativas(flst_estimativas: list, fargs: argparse.Namespace) -> None:
    """
    tabela das estimativas no log.
    """
    M_LOG.info(f"[=] estimativa ({fargs.pares} pares, {fargs.pares_em_paralelo} em paralelo; "
               f"timeout {fargs.timeout_s:g} s, ciclo {fargs.ciclo_s:g} s):")
    M_LOG.info("[=]   largura  grade     par  par/cron  passada  timeout  ciclo")

    for ldct in flst_estimativas:
        M_LOG.info(f"[=]   {ldct['largura_base']:>7}  {ldct['grade']:>5}  "
                   f"{ldct['par_s']:6.1f}  {ldct['par_no_cron_s']:8.1f}  "
                   f"{ldct['passada_s']:7.1f}  "
                   f"{'ok' if ldct['cabe_no_timeout'] else 'NÃO':>7}  "
                   f"{'ok' if ldct['cabe_no_ciclo'] else 'NÃO':>5}")

# < main >-------------------------------------------------------------------------------

def main(flst_argv: typing.Optional[list]=None) -> int:
    """
    mede, estima e (opcionalmente) grava o JSON.

    :returns: 0 se mediu, 1 caso contrário
    """
    largs = arg_parse(flst_argv)
    llst_backends = [ls.strip() for ls in largs.backends.split(",") if ls.strip()]

    llst_medidas = mede_tudo(largs, llst_backends)

    if not llst_medidas:
        M_LOG.error("[x] nada medido (backends/larguras/grades vazios).")
        return 1

    llst_estimativas = estima(llst_medidas, largs)
    loga_estimativas(llst_estimativas, largs)

    if largs.saida:
        lpth = pathlib.Path(largs.saida)
        lpth.parent.mkdir(parents=True, exist_ok=True)
        lpth.write_text(json.dumps({"medidas": llst_medidas, "estimativas": llst_estimativas,
                                    "parametros": {ls: lv for ls, lv in vars(largs).items()}},
                                   indent=2, ensure_ascii=False))
        M_LOG.info(f"[✓] resultado em {lpth}.")

    return 0

# -------------------------------------------------------------------------------------
# this is the bootstrap process

if "__main__" == __name__:
    # logger
    logging.basicConfig(level=logging.INFO)

    # run
    sys.exit(main())

# < the end >----------------------------------------------------------------------------
