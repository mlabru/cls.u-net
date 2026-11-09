# -*- coding: utf-8 -*-
"""
Centro Logístico de Simulação Meteorológica
Copyright (c) 1995-2026, Instituto de Controle do Espaço Aéreo

untEval

avaliação comparativa dos backends treinados (pytorch, rainnet, smaat) e da
persistência, sobre as janelas de um conjunto do manifesto (padrão: teste), em todos
os passos (+10..+60 min), na grade inteira. Responde às questões de pesquisa do
relatório (seção 6.5): qualidade (CSI, POD, FAR, frequency bias, FSS, MAE) e custo
computacional (parâmetros, tempo de treino, tempo de inferência, memória).

agregação por backend e passo, no conjunto todo e, separadamente, por par
radar/alcance da janela (cada radar tem regime, sazonalidade e calibração próprios, e o
agregado é dominado pelos pares com mais eco):
  - categóricas: acertos/perdas/falsos SOMADOS em todas as janelas, e só então
    POD/FAR/CSI/bias (o CSI médio por janela exagera janelas com pouco eco);
  - FSS: termos somados em todas as janelas, e só então a razão (a média por janela
    vai em "fss_media_janelas", só como referência);
  - incerteza: bootstrap pareado dos blocos de tempo do split (unt_bootstrap,
    [avaliacao]); as somas por bloco vão em "somas_por_bloco" e os intervalos, com a
    diferença de cada backend para a persistência, em "bootstrap";
  - MAE/bias: média ponderada pelo número de pixels com eco.

fluxo de main():
  1. lê o manifesto e escolhe as janelas do conjunto;
  2. carrega o atual.pt de cada backend (os ausentes são pulados com aviso) e os
     checkpoints extras de --checkpoint ROTULO=CAMINHO (ex.: o candidato do untTrain,
     avaliado ao lado do atual para o untPromove);
  3. por janela: carrega entradas + alvos, prevê com cada backend (cronometrado) e
     compara cada passo com o alvo observado (unt_verify.verifica_lead); a
     persistência repete a última entrada;
  4. grava <saida>/comparacao.json (agregados em "resultados", por par em
     "resultados_por_par", desempenho e parâmetros), <saida>/comparacao.csv (uma
     linha por backend x passo) e <saida>/comparacao_por_par.csv (uma linha por
     par x backend x passo), e loga um resumo (e, com mais de um par, um por par).

código de saída: 0 se avaliou ao menos uma janela; 1 caso contrário.

$ python untEval.py --janelas data/janelas/janelas.csv [--split teste]
                    [--cappi-dir ./data/cappi] [--modelos-dir ./data/modelos]
                    [--backends pytorch,rainnet,smaat] [--device auto]
                    [--saida data/avaliacao/<AAAAMMDDhhmmss>] [--max-janelas N]
                    [--checkpoint ROTULO=CAMINHO ...] [--config config.ini]

2026.out  mlabru  bootstrap por bloco (somas_por_bloco, bootstrap); --config
2026.out  mlabru  FSS agregado pelas somas dos termos (unt_avaliacao)
2026.out  mlabru  métricas por par radar/alcance (resultados_por_par, janelas_por_par,
                  comparacao_por_par.csv); "resultados" segue agregado (untPromove)
2026.out  mlabru  acumulador e agregação movidos para work/unt_avaliacao (comuns com a
                  seleção de época do treino)
2026.out  mlabru  --checkpoint ROTULO=CAMINHO (candidato ao lado do atual); arquivo,
                  backend e sha256 de cada checkpoint em "desempenho"
2026.out  mlabru  initial version (Linux/Python)
"""
# < imports >----------------------------------------------------------------------------

# python library
import argparse
import csv
import datetime
import logging
import pathlib
import resource
import sys
import time
import typing

# numPy
import numpy as np

# pyTorch
import torch

# model
import work.unt_avaliacao as ua
import work.unt_bootstrap as ub
import work.unt_catalog as oc
import work.unt_config as uc
import work.unt_dataset as uds
import work.unt_device as ue
import work.unt_inferencia as ui
import work.unt_janelas as uj
import work.unt_modelos as um
import work.unt_verify as ov

# < constants >--------------------------------------------------------------------------

# nome da referência no relatório
CS_PERSISTENCIA = "persistencia"

# arquivos de saída
CS_ARQUIVO_JSON = "comparacao.json"
CS_ARQUIVO_CSV = "comparacao.csv"
CS_ARQUIVO_CSV_POR_PAR = "comparacao_por_par.csv"

# < logging >----------------------------------------------------------------------------

# logger
M_LOG = logging.getLogger(__name__)
M_LOG.setLevel(logging.INFO)

# < argumentos >-------------------------------------------------------------------------

def arg_parse(flst_argv: typing.Optional[list]=None) -> argparse.Namespace:
    """
    lê os argumentos da linha de comando.

    :param flst_argv: argumentos (None = sys.argv)

    :returns: argumentos lidos
    """
    lparser = argparse.ArgumentParser(
        description="Avaliação comparativa dos backends de Deep Learning e da "
                    "persistência sobre as janelas de um conjunto do manifesto")

    lparser.add_argument("--janelas", type=str, required=True,
                         help="manifesto de janelas (.csv do untJanelas)")
    lparser.add_argument("--split", type=str, default=uj.CS_SPLIT_TESTE, choices=uj.CT_SPLITS,
                         help="conjunto avaliado")
    lparser.add_argument("--cappi-dir", type=str, default=oc.diretorio_cappi_padrao(),
                         help="diretório dos CAPPI (.npy)")
    lparser.add_argument("--modelos-dir", type=str, default=um.diretorio_modelos_padrao(),
                         help="raiz dos checkpoints (<dir>/<backend>/atual.pt)")
    lparser.add_argument("--backends", type=str, default=",".join(um.CT_BACKENDS),
                         help="backends avaliados, separados por vírgula")
    lparser.add_argument("--device", type=str, default="auto",
                         help="auto, cpu, cuda, cuda:N ou mps")
    lparser.add_argument("--saida", type=str, default=None,
                         help="diretório de saída (padrão data/avaliacao/<carimbo UTC>)")
    lparser.add_argument("--max-janelas", type=int, default=0,
                         help="avalia no máximo N janelas (0 = todas)")
    lparser.add_argument("--checkpoint", action="append", default=[],
                         metavar="ROTULO=CAMINHO",
                         help="checkpoint extra avaliado sob ROTULO (repetível), ex.: "
                              "rainnet_candidato=data/modelos/rainnet/rainnet_<UTC>.pt")
    lparser.add_argument("--config", type=str, default=None,
                         help="config.ini ([avaliacao] do bootstrap); padrão ./config.ini "
                              "se existir")

    return lparser.parse_args(flst_argv)

# < passos de main() >-------------------------------------------------------------------

def checkpoints_extras(flst_especificacoes: list) -> dict:
    """
    --checkpoint ROTULO=CAMINHO -> {rótulo: caminho}.

    :raises ValueError: especificação sem "=", rótulo vazio/repetido ou que colide com um
                        backend ou com a persistência
    """
    ldct = {}

    for ls_especificacao in flst_especificacoes:
        ls_rotulo, ls_igual, ls_caminho = ls_especificacao.partition("=")
        ls_rotulo = ls_rotulo.strip()

        if not ls_igual or not ls_rotulo or not ls_caminho.strip():
            raise ValueError(f"--checkpoint inválido: {ls_especificacao} (use ROTULO=CAMINHO)")

        if ls_rotulo in ldct or ls_rotulo in um.CT_BACKENDS or CS_PERSISTENCIA == ls_rotulo:
            raise ValueError(f"--checkpoint: rótulo repetido ou reservado: {ls_rotulo}")

        ldct[ls_rotulo] = ls_caminho.strip()

    return ldct

# -------------------------------------------------------------------------------------
def carrega_modelos(fargs: argparse.Namespace) -> dict:
    """
    passo 2: atual.pt de cada backend pedido (ausentes são pulados com aviso) e os
    checkpoints extras (--checkpoint; ausente é erro).

    :returns: rótulo -> SModeloCarregado

    :raises FileNotFoundError: checkpoint extra ausente
    """
    ldevice = ue.escolhe_device(fargs.device)
    ue.configura_threads_cpu(ldevice)

    ldct_modelos = {}

    for ls_backend in [ls.strip() for ls in fargs.backends.split(",") if ls.strip()]:
        lpth_checkpoint = um.caminho_checkpoint_atual(fargs.modelos_dir, ls_backend)

        if not lpth_checkpoint.exists():
            M_LOG.warning(f"[!] {ls_backend}: sem checkpoint ({lpth_checkpoint}); pulado.")
            continue

        ldct_modelos[ls_backend] = _carrega_identificado(lpth_checkpoint, ldevice)

    for ls_rotulo, ls_caminho in checkpoints_extras(fargs.checkpoint).items():
        ldct_modelos[ls_rotulo] = _carrega_identificado(ls_caminho, ldevice)

    return ldct_modelos

# -------------------------------------------------------------------------------------
def _carrega_identificado(fpth_checkpoint, fdevice) -> um.SModeloCarregado:
    """
    carrega e anota o arquivo real e o sha256 (o untPromove confere que promove
    exatamente o que foi avaliado).
    """
    lpth_real = pathlib.Path(fpth_checkpoint).resolve()
    lmodelo = um.carrega_checkpoint(lpth_real, fdevice)

    lmodelo.checkpoint["arquivo"] = str(lpth_real)
    lmodelo.checkpoint["sha256"] = um.sha256_de_arquivo(lpth_real)

    return lmodelo

# -------------------------------------------------------------------------------------
def _sincroniza(fmodelo: um.SModeloCarregado) -> None:
    """
    espera a GPU terminar (cronometragem honesta em cuda).
    """
    if "cuda" == ui.device_do_modelo(fmodelo).type:
        torch.cuda.synchronize()

# -------------------------------------------------------------------------------------
def par_da_janela(fjanela) -> str:
    """
    par radar/alcance da janela, como no escopo dos checkpoints ("STI_250").
    """
    return f"{fjanela.radar}_{fjanela.alcance}"

# -------------------------------------------------------------------------------------
def bloco_da_janela(fjanela, fi_bloco_s: int) -> str:
    """
    bloco de tempo do split (unt_janelas.atribui_split) do último quadro de entrada,
    i.e. da emissão da previsão: a unidade independente do bootstrap.
    """
    return str(fjanela.entradas[-1] // fi_bloco_s)

# -------------------------------------------------------------------------------------
def avalia_janela(fs_cappi_dir: str, fjanela, fdct_modelos: dict, fdct_acumuladores: dict,
                  fdct_tempos: dict, fdct_acumuladores_par: typing.Optional[dict]=None,
                  fdct_acumuladores_bloco: typing.Optional[dict]=None,
                  fi_bloco_s: int=uj.CI_SEGUNDOS_POR_DIA) -> None:
    """
    passo 3: uma janela, todos os backends e a persistência, todos os passos.

    :param fdct_acumuladores: rótulo -> passo -> AcumuladorLead do conjunto todo
    :param fdct_acumuladores_par: par -> rótulo -> passo -> AcumuladorLead (None = não
                                  separa por par)
    :param fdct_acumuladores_bloco: bloco -> rótulo -> passo -> AcumuladorLead (None =
                                    não separa por bloco)
    :param fi_bloco_s: tamanho do bloco (s), o do split do manifesto
    """
    li_num_entradas = len(fjanela.entradas)

    lnp_quadros = uds.carrega_quadros(fs_cappi_dir, fjanela)
    lnp_entradas = lnp_quadros[:li_num_entradas]
    lnp_alvos = lnp_quadros[li_num_entradas:]
    li_num_passos = lnp_alvos.shape[0]

    ldct_previsoes = {CS_PERSISTENCIA: np.repeat(lnp_entradas[-1:], li_num_passos, axis=0)}

    for ls_backend, lmodelo in fdct_modelos.items():
        lf_inicio = time.perf_counter()
        ldct_previsoes[ls_backend] = ui.preve_dbz(lmodelo, lnp_entradas, li_num_passos)
        _sincroniza(lmodelo)
        fdct_tempos.setdefault(ls_backend, []).append(time.perf_counter() - lf_inicio)

    llst_acumuladores = [fdct_acumuladores]

    if fdct_acumuladores_par is not None:
        llst_acumuladores.append(fdct_acumuladores_par.setdefault(par_da_janela(fjanela), {}))

    if fdct_acumuladores_bloco is not None:
        llst_acumuladores.append(fdct_acumuladores_bloco.setdefault(
            bloco_da_janela(fjanela, fi_bloco_s), {}))

    ua.acumula(ldct_previsoes, lnp_alvos, *llst_acumuladores)

# -------------------------------------------------------------------------------------
def desempenho(fdct_modelos: dict, fdct_tempos: dict) -> dict:
    """
    custo computacional de cada backend.
    """
    ldct_desempenho = {}

    for ls_backend, lmodelo in fdct_modelos.items():
        llst_tempos = fdct_tempos.get(ls_backend, [])
        ldct_checkpoint = lmodelo.checkpoint

        ldct_desempenho[ls_backend] = {
            "backend": lmodelo.backend,
            "arquivo": ldct_checkpoint.get("arquivo"),
            "sha256": ldct_checkpoint.get("sha256"),
            "parametros": um.conta_parametros(lmodelo.rede),
            "checkpoint": ldct_checkpoint.get("criado_em"),
            "tempo_treino_s": ldct_checkpoint.get("tempo_treino_s"),
            "device_treino": ldct_checkpoint.get("device"),
            "melhor_epoca": ldct_checkpoint.get("epoca"),
            "inferencia_ms_media": 1000.0 * float(np.mean(llst_tempos)) if llst_tempos else None,
            "device_inferencia": str(ui.device_do_modelo(lmodelo)),
            "largura_base": lmodelo.hparams.get("largura_base"),
            "modo": lmodelo.hparams.get("modo")}

    return ldct_desempenho

# -------------------------------------------------------------------------------------
def memoria_pico() -> dict:
    """
    pico de memória do processo (RSS, MB) e, com cuda, da GPU.
    """
    ldct = {"rss_pico_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0}

    if torch.cuda.is_available():
        ldct["gpu_pico_mb"] = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)

    return ldct

# -------------------------------------------------------------------------------------
def linhas_csv(fdct_resultados: dict, fs_par: typing.Optional[str]=None) -> list:
    """
    uma linha por (backend, passo), métricas principais em colunas.

    :param fs_par: par radar/alcance, gravado na primeira coluna (None = sem a coluna)
    """
    llst_linhas = []

    for ls_nome, ldct_por_passo in fdct_resultados.items():
        for ls_passo, ldct in ldct_por_passo.items():
            ldct_linha = {} if fs_par is None else {"par": fs_par}
            ldct_linha.update({"backend": ls_nome, "passo": int(ls_passo),
                               "janelas": ldct["janelas"],
                               "mae_dbz": ldct["mae_dbz"], "bias_dbz": ldct["bias_dbz"]})

            for ls_limiar, ldct_lim in ldct["limiares"].items():
                for ls_chave in ("pod", "far", "csi", "freq_bias"):
                    ldct_linha[f"{ls_chave}{ls_limiar}"] = ldct_lim[ls_chave]

            for ls_limiar, ldct_escalas in ldct["fss"].items():
                for ls_escala, lf_fss in ldct_escalas.items():
                    ldct_linha[f"fss{ls_limiar}_{ls_escala}km"] = lf_fss

            llst_linhas.append(ldct_linha)

    return llst_linhas

# -------------------------------------------------------------------------------------
def linhas_csv_por_par(fdct_resultados_por_par: dict) -> list:
    """
    uma linha por (par, backend, passo).
    """
    llst_linhas = []

    for ls_par, ldct_resultados in fdct_resultados_por_par.items():
        llst_linhas.extend(linhas_csv(ldct_resultados, ls_par))

    return llst_linhas

# -------------------------------------------------------------------------------------
def grava_csv(flst_linhas: list, fpth_csv: pathlib.Path) -> None:
    """
    grava o CSV com a união das colunas (métricas ausentes ficam vazias).
    """
    llst_colunas = []
    for ldct in flst_linhas:
        for ls_coluna in ldct:
            if ls_coluna not in llst_colunas:
                llst_colunas.append(ls_coluna)

    with fpth_csv.open("w", newline="") as larq:
        lescritor = csv.DictWriter(larq, fieldnames=llst_colunas)
        lescritor.writeheader()
        lescritor.writerows(flst_linhas)

# -------------------------------------------------------------------------------------
def _fmt(fx_valor: typing.Optional[float], fs_formato: str=".2f") -> str:
    return "-" if fx_valor is None else format(fx_valor, fs_formato)

# -------------------------------------------------------------------------------------
def loga_resumo(fdct_resultados: dict) -> None:
    """
    tabela resumida no log: CSI30, FSS30@10 km e MAE por backend e passo.
    """
    M_LOG.info("[=] backend       passo  CSI20  CSI30  CSI36  FSS30@10  MAE(dBZ)")

    for ls_nome, ldct_por_passo in fdct_resultados.items():
        for ls_passo, ldct in ldct_por_passo.items():
            ldct_lim = ldct["limiares"]
            lf_fss = ldct["fss"].get("30", {}).get("10")

            M_LOG.info(f"[=] {ls_nome:<13} +{int(ls_passo) * 10:>3}m  "
                       f"{_fmt(ldct_lim.get('20', {}).get('csi')):>5}  "
                       f"{_fmt(ldct_lim.get('30', {}).get('csi')):>5}  "
                       f"{_fmt(ldct_lim.get('36', {}).get('csi')):>5}  "
                       f"{_fmt(lf_fss):>8}  {_fmt(ldct['mae_dbz'], '.1f'):>8}")

# -------------------------------------------------------------------------------------
def _csi30(fdct_passo: dict) -> typing.Optional[float]:
    return fdct_passo.get("limiares", {}).get("30", {}).get("csi")

# -------------------------------------------------------------------------------------
def _fss30_10(fdct_passo: dict) -> typing.Optional[float]:
    return fdct_passo.get("fss", {}).get("30", {}).get("10")

# -------------------------------------------------------------------------------------
def loga_resumo_por_par(fdct_resultados_por_par: dict, fdct_janelas_por_par: dict) -> None:
    """
    tabela resumida por par no log: CSI30 e FSS30@10 km em +30 e +60 min por backend.
    """
    M_LOG.info("[=] par        backend       janelas  CSI30+30m  CSI30+60m  "
               "FSS30@10+30m  FSS30@10+60m")

    for ls_par, ldct_resultados in fdct_resultados_por_par.items():
        for ls_nome, ldct_por_passo in ldct_resultados.items():
            ldct_p3 = ldct_por_passo.get("3", {})
            ldct_p6 = ldct_por_passo.get("6", {})

            M_LOG.info(f"[=] {ls_par:<10} {ls_nome:<13} {fdct_janelas_por_par[ls_par]:>7}  "
                       f"{_fmt(_csi30(ldct_p3)):>9}  {_fmt(_csi30(ldct_p6)):>9}  "
                       f"{_fmt(_fss30_10(ldct_p3)):>12}  {_fmt(_fss30_10(ldct_p6)):>12}")

# < bootstrap >--------------------------------------------------------------------------

def bootstrap(fdct_somas_por_bloco: dict, fs_config: typing.Optional[str],
              fi_bloco_s: int) -> dict:
    """
    intervalos de confiança por bootstrap dos blocos ([avaliacao] do config.ini).
    """
    lconfig = uc.carrega(fs_config)

    ldct_bootstrap = ub.resume(fdct_somas_por_bloco,
                               lconfig.inteiro("avaliacao", "reamostras"),
                               lconfig.real("avaliacao", "nivel_confianca"),
                               lconfig.inteiro("avaliacao", "semente"),
                               CS_PERSISTENCIA)
    ldct_bootstrap["bloco_s"] = fi_bloco_s

    return ldct_bootstrap

# -------------------------------------------------------------------------------------
def _fmt_ic(fdct: typing.Optional[dict]) -> str:
    """
    "[+0.012, +0.054] p=0.98" de um {"ic", "p_melhor"} (ou "-").
    """
    if not fdct or fdct.get("ic") is None:
        return "-"

    lf_lo, lf_hi = fdct["ic"]

    return f"[{lf_lo:+.3f}, {lf_hi:+.3f}] p={fdct['p_melhor']:.2f}"

# -------------------------------------------------------------------------------------
def loga_bootstrap(fdct_bootstrap: dict, fs_config: typing.Optional[str]) -> None:
    """
    diferença para a persistência (CSI e FSS no limiar/escala do [promocao]) com o
    intervalo do bootstrap, por backend e passo.
    """
    lconfig = uc.carrega(fs_config)
    ls_limiar = uc.chave_numerica(lconfig.texto("promocao", "limiar_dbz"))
    ls_escala = uc.chave_numerica(lconfig.texto("promocao", "escala_fss_km"))
    ls_chave_dif = f"dif_{CS_PERSISTENCIA}"

    M_LOG.info(f"[=] bootstrap: {fdct_bootstrap['blocos']} blocos de "
               f"{fdct_bootstrap.get('bloco_s')} s, {fdct_bootstrap['reamostras']} "
               f"reamostragens, IC {100 * fdct_bootstrap['nivel']:.0f}% da diferença para "
               f"a persistência (p = fração em que o backend é melhor)")

    if fdct_bootstrap["blocos"] < lconfig.inteiro("promocao", "min_blocos"):
        M_LOG.warning(f"[!] só {fdct_bootstrap['blocos']} bloco(s): os intervalos do "
                      f"bootstrap não são confiáveis.")

    M_LOG.info(f"[=] backend            passo  {'CSI' + ls_limiar:<31}"
               f"FSS{ls_limiar}@{ls_escala}km")

    for ls_rotulo, ldct_por_passo in fdct_bootstrap["resultados"].items():
        if CS_PERSISTENCIA == ls_rotulo:
            continue

        for ls_passo, ldct_passo in ldct_por_passo.items():
            ldct_dif = ldct_passo.get(ls_chave_dif, {})
            ldct_csi = ldct_dif.get("csi", {}).get(ls_limiar)
            ldct_fss = ldct_dif.get("fss", {}).get(ls_limiar, {}).get(ls_escala)

            M_LOG.info(f"[=] {ls_rotulo:<18} {ls_passo:>5}  {_fmt_ic(ldct_csi):<31}"
                       f"{_fmt_ic(ldct_fss)}")

# < main >-------------------------------------------------------------------------------

def main(flst_argv: typing.Optional[list]=None) -> int:
    """
    avalia e grava a comparação (ver o fluxo no topo do módulo).

    :returns: 0 se avaliou ao menos uma janela, 1 caso contrário
    """
    largs = arg_parse(flst_argv)

    # 1. janelas
    llst_janelas, ldct_manifesto = uj.le_manifesto(largs.janelas)
    llst_janelas = uj.filtra_split(llst_janelas, largs.split)

    if largs.max_janelas > 0:
        llst_janelas = llst_janelas[:largs.max_janelas]

    if not llst_janelas:
        M_LOG.error(f"[x] nenhuma janela no conjunto '{largs.split}' de {largs.janelas}.")
        return 1

    # 2. modelos
    try:
        ldct_modelos = carrega_modelos(largs)

    except (ValueError, FileNotFoundError) as le_erro:
        M_LOG.error(f"[x] {le_erro}")
        return 1

    if not ldct_modelos:
        M_LOG.warning("[!] nenhum backend treinado: só a persistência será avaliada.")

    # 3. janelas
    ldct_acumuladores = {}
    ldct_acumuladores_par = {}
    ldct_acumuladores_bloco = {}
    ldct_tempos = {}
    li_bloco_s = int(ldct_manifesto.get("bloco_split_s", uj.CI_SEGUNDOS_POR_DIA))

    for li_indice, ljanela in enumerate(llst_janelas, start=1):
        avalia_janela(largs.cappi_dir, ljanela, ldct_modelos, ldct_acumuladores, ldct_tempos,
                      ldct_acumuladores_par, ldct_acumuladores_bloco, li_bloco_s)

        if 0 == li_indice % 50:
            M_LOG.info(f"[+] {li_indice}/{len(llst_janelas)} janelas avaliadas...")

    ldct_resultados = ua.resume(ldct_acumuladores)
    ldct_resultados_por_par = {ls_par: ua.resume(ldct_acumuladores_par[ls_par])
                               for ls_par in sorted(ldct_acumuladores_par)}

    ldct_somas_por_bloco = {ls_bloco: ua.somas(ldct_acumuladores_bloco[ls_bloco])
                            for ls_bloco in sorted(ldct_acumuladores_bloco)}
    ldct_bootstrap = bootstrap(ldct_somas_por_bloco, largs.config, li_bloco_s)

    ldct_janelas_por_par = {}
    for ljanela in llst_janelas:
        ls_par = par_da_janela(ljanela)
        ldct_janelas_por_par[ls_par] = ldct_janelas_por_par.get(ls_par, 0) + 1

    # 4. saída
    ls_carimbo = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d%H%M%S")
    lpth_saida = pathlib.Path(largs.saida or f"./data/avaliacao/{ls_carimbo}")
    lpth_saida.mkdir(parents=True, exist_ok=True)

    ldct_comparacao = {"manifesto": str(largs.janelas),
                       "split": largs.split,
                       "janelas": len(llst_janelas),
                       "janelas_por_par": dict(sorted(ldct_janelas_por_par.items())),
                       "parametros_manifesto": ldct_manifesto,
                       "limiares_dbz": list(ov.CT_LIMIARES_DBZ),
                       "fss_escalas_km": list(ov.CT_ESCALAS_FSS_KM),
                       "desempenho": desempenho(ldct_modelos, ldct_tempos),
                       "memoria": memoria_pico(),
                       "resultados": ldct_resultados,
                       "resultados_por_par": ldct_resultados_por_par,
                       "somas_por_bloco": ldct_somas_por_bloco,
                       "bootstrap": ldct_bootstrap,
                       "criado_em": ls_carimbo}

    ov.grava_json(ldct_comparacao, lpth_saida / CS_ARQUIVO_JSON)
    grava_csv(linhas_csv(ldct_resultados), lpth_saida / CS_ARQUIVO_CSV)
    grava_csv(linhas_csv_por_par(ldct_resultados_por_par), lpth_saida / CS_ARQUIVO_CSV_POR_PAR)

    loga_resumo(ldct_resultados)

    if len(ldct_resultados_por_par) > 1:
        loga_resumo_por_par(ldct_resultados_por_par, ldct_janelas_por_par)

    loga_bootstrap(ldct_bootstrap, largs.config)

    M_LOG.info(f"[✓] {len(llst_janelas)} janelas avaliadas; comparação em {lpth_saida}.")

    return 0

# -------------------------------------------------------------------------------------
# this is the bootstrap process

if "__main__" == __name__:
    # logger
    logging.basicConfig(level=logging.INFO)

    # run
    sys.exit(main())

# < the end >----------------------------------------------------------------------------
