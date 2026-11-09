# -*- coding: utf-8 -*-
"""
Centro Logístico de Simulação Meteorológica
Copyright (c) 1995-2026, Instituto de Controle do Espaço Aéreo

untMonitor

skill dos backends EM OPERAÇÃO: lê os históricos <R>_<A>_verify.jsonl (uma linha por
nowcast verificado, gravados pelo unt_run.sh com UNT_VERIFY=true) e compara cada
backend com a persistência numa janela móvel. Um backend abaixo da persistência
aciona um alarme no log -- sinal para avaliar o rollback (untPromove --reverte).

fluxo de main():
  1. lista os *_verify.jsonl de --entrada (diretório ou arquivo);
  2. por par, as linhas cujo target está na janela ([monitor] janela_h, até --agora);
  3. agrega por backend e persistência (unt_avaliacao.AcumuladorLead: contingências
     somadas antes das razões) e pontua (unt_avaliacao.pontuacao, limiar/escala do
     [promocao]);
  4. alarme: com ao menos [monitor] min_verificacoes verificações com evento observado,
     pontuação do backend < persistência - margem;
  5. grava <saida>/<R>_<A>_monitor.json (para o dashboard) e loga o resumo.

as verificações em operação são de UM passo (o target da sequência, +10 min); o
horizonte inteiro só é avaliado no untEval.

código de saída: 0 sem alarme; 2 com alarme em algum par; 1 sem nenhum histórico;
3 config inválido.

$ python untMonitor.py -i <UNT_OUT> [--saida DIR] [--config config.ini] [--agora EPOCH]
                       [--janela-h H]

2026.out  mlabru  initial version (Linux/Python)
"""
# < imports >----------------------------------------------------------------------------

# python library
import argparse
import json
import logging
import pathlib
import sys
import time
import typing

# model
import work.unt_avaliacao as ua
import work.unt_config as uc

# < constants >--------------------------------------------------------------------------

# sufixos dos arquivos lidos e gravados
CS_SUFIXO_VERIFY = "_verify.jsonl"
CS_SUFIXO_MONITOR = "_monitor.json"

# rótulo da persistência nas linhas do verify (unt_verify.verifica_previsoes)
CS_PERSISTENCIA = "persistencia"

# códigos de saída
CI_SEM_ALARME = 0
CI_SEM_HISTORICO = 1
CI_COM_ALARME = 2
CI_ERRO_CONFIG = 3

# < logging >----------------------------------------------------------------------------

# logger
M_LOG = logging.getLogger(__name__)
M_LOG.setLevel(logging.INFO)

# < tipos >------------------------------------------------------------------------------

class SCriterioMonitor(typing.NamedTuple):
    """
    [monitor] + limiar/escala do [promocao].
    """
    janela_h: float
    min_verificacoes: int
    margem: float
    limiar_dbz: str
    escala_fss_km: str

# < argumentos >-------------------------------------------------------------------------

def arg_parse(flst_argv: typing.Optional[list]=None) -> argparse.Namespace:
    """
    lê os argumentos da linha de comando.

    :param flst_argv: argumentos (None = sys.argv)

    :returns: argumentos lidos
    """
    lparser = argparse.ArgumentParser(
        description="Skill dos backends em operação (verify.jsonl) contra a persistência")

    lparser.add_argument("-i", "--entrada", type=str, required=True,
                         help="diretório com <R>_<A>_verify.jsonl (ou um arquivo)")
    lparser.add_argument("--saida", type=str, default=None,
                         help="onde gravar <R>_<A>_monitor.json (padrão: o da entrada)")
    lparser.add_argument("--config", type=str, default=None,
                         help="config.ini; padrão ./config.ini se existir")
    lparser.add_argument("--agora", type=int, default=None,
                         help="fim da janela (epoch UTC; padrão: agora)")
    lparser.add_argument("--janela-h", type=float, default=None,
                         help="janela móvel em horas (padrão [monitor] janela_h)")

    return lparser.parse_args(flst_argv)

# < leitura >----------------------------------------------------------------------------

def criterio_do_config(fconfig: uc.SConfig, ff_janela_h: typing.Optional[float]) -> SCriterioMonitor:
    """
    [monitor] e [promocao] -> SCriterioMonitor (--janela-h sobrepõe).
    """
    lf_janela_h = ff_janela_h if ff_janela_h is not None else fconfig.real("monitor", "janela_h")

    return SCriterioMonitor(
        janela_h=lf_janela_h,
        min_verificacoes=fconfig.inteiro("monitor", "min_verificacoes"),
        margem=fconfig.real("monitor", "margem"),
        limiar_dbz=uc.chave_numerica(fconfig.texto("promocao", "limiar_dbz")),
        escala_fss_km=uc.chave_numerica(fconfig.texto("promocao", "escala_fss_km")))

# -------------------------------------------------------------------------------------
def lista_historicos(fs_entrada: str) -> list:
    """
    passo 1: os *_verify.jsonl (ou o próprio arquivo).
    """
    lpth = pathlib.Path(fs_entrada)

    if lpth.is_file():
        return [lpth]

    return sorted(lpth.glob(f"*{CS_SUFIXO_VERIFY}"))

# -------------------------------------------------------------------------------------
def le_linhas_na_janela(fpth_historico: pathlib.Path, fi_inicio: int, fi_fim: int) -> list:
    """
    passo 2: linhas válidas cujo target está em [fi_inicio, fi_fim]. Linha corrompida
    (ex.: gravação interrompida) é pulada com aviso.
    """
    llst_linhas = []

    with fpth_historico.open(encoding="utf-8") as larq:
        for li_numero, ls_linha in enumerate(larq, start=1):
            if not ls_linha.strip():
                continue

            try:
                ldct = json.loads(ls_linha)

            except ValueError:
                M_LOG.warning(f"[!] {fpth_historico.name}:{li_numero}: linha inválida; pulada.")
                continue

            li_target = ldct.get("target")

            if isinstance(li_target, int) and fi_inicio <= li_target <= fi_fim:
                llst_linhas.append(ldct)

    return llst_linhas

# < agregação e julgamento >-------------------------------------------------------------

def _tem_evento_observado(fdct_linha: dict, fs_limiar: str) -> bool:
    """
    o observado tem evento no limiar? (acertos + perdas da persistência > 0: os dois
    contam pixels com evento observado, qualquer que seja a previsão)
    """
    ldct = fdct_linha.get(CS_PERSISTENCIA, {}).get("limiares", {}).get(fs_limiar, {})

    return ldct.get("hits", 0) + ldct.get("misses", 0) > 0

# -------------------------------------------------------------------------------------
def agrega(flst_linhas: list, fcriterio: SCriterioMonitor) -> dict:
    """
    passo 3: rótulo (backends e persistência) -> resumo agregado + pontuação.

    :returns: {"verificacoes", "verificacoes_com_evento", "rotulos": {rótulo: {...}}}
    """
    ldct_acumuladores = {}
    li_com_evento = 0

    for ldct_linha in flst_linhas:
        li_com_evento += int(_tem_evento_observado(ldct_linha, fcriterio.limiar_dbz))

        ldct_metricas = dict(ldct_linha.get("backends", {}))
        if CS_PERSISTENCIA in ldct_linha:
            ldct_metricas[CS_PERSISTENCIA] = ldct_linha[CS_PERSISTENCIA]

        for ls_rotulo, ldct in ldct_metricas.items():
            ldct_acumuladores.setdefault(ls_rotulo, ua.AcumuladorLead()).adiciona(ldct)

    ldct_rotulos = {}

    for ls_rotulo, lacumulador in sorted(ldct_acumuladores.items()):
        ldct_resumo = lacumulador.resumo()
        ldct_limiar = ldct_resumo["limiares"].get(fcriterio.limiar_dbz, {})

        ldct_rotulos[ls_rotulo] = {
            "verificacoes": ldct_resumo["janelas"],
            "pontuacao": ua.pontuacao({"1": ldct_resumo}, fcriterio.limiar_dbz,
                                      fcriterio.escala_fss_km),
            "csi": ldct_limiar.get("csi"),
            "pod": ldct_limiar.get("pod"),
            "far": ldct_limiar.get("far"),
            "fss": ldct_resumo["fss"].get(fcriterio.limiar_dbz, {}).get(fcriterio.escala_fss_km),
            "mae_dbz": ldct_resumo["mae_dbz"]}

    return {"verificacoes": len(flst_linhas),
            "verificacoes_com_evento": li_com_evento,
            "rotulos": ldct_rotulos}

# -------------------------------------------------------------------------------------
def alarmes(fdct_agregado: dict, fcriterio: SCriterioMonitor) -> list:
    """
    passo 4: backends abaixo da persistência - margem (só com eventos suficientes).

    :returns: [(backend, pontuação, pontuação da persistência)]
    """
    if fdct_agregado["verificacoes_com_evento"] < fcriterio.min_verificacoes:
        return []

    ldct_rotulos = fdct_agregado["rotulos"]
    lf_persistencia = ldct_rotulos.get(CS_PERSISTENCIA, {}).get("pontuacao")

    if lf_persistencia is None:
        return []

    llst_alarmes = []

    for ls_rotulo, ldct in ldct_rotulos.items():
        if CS_PERSISTENCIA == ls_rotulo or ldct["pontuacao"] is None:
            continue

        if ldct["pontuacao"] < lf_persistencia - fcriterio.margem:
            llst_alarmes.append((ls_rotulo, ldct["pontuacao"], lf_persistencia))

    return llst_alarmes

# < saída >------------------------------------------------------------------------------

def _fmt(fx_valor: typing.Optional[float]) -> str:
    return "-" if fx_valor is None else f"{fx_valor:.3f}"

# -------------------------------------------------------------------------------------
def loga_par(fs_par: str, fdct_estado: dict, fcriterio: SCriterioMonitor) -> None:
    """
    resumo de um par no log.
    """
    M_LOG.info(f"[=] {fs_par}: {fdct_estado['verificacoes']} verificações em "
               f"{fcriterio.janela_h:g} h ({fdct_estado['verificacoes_com_evento']} com "
               f"evento >= {fcriterio.limiar_dbz} dBZ)")

    for ls_rotulo, ldct in fdct_estado["rotulos"].items():
        M_LOG.info(f"[=]   {ls_rotulo:<13} pontuação {_fmt(ldct['pontuacao'])}  "
                   f"CSI {_fmt(ldct['csi'])}  FSS {_fmt(ldct['fss'])}  "
                   f"({ldct['verificacoes']} verif.)")

    for ls_rotulo, lf_backend, lf_persistencia in fdct_estado["alarmes"]:
        M_LOG.warning(f"[!] ALARME {fs_par}: {ls_rotulo} abaixo da persistência nas últimas "
                      f"{fcriterio.janela_h:g} h ({lf_backend:.3f} < {lf_persistencia:.3f}); "
                      f"avalie untPromove -b {ls_rotulo} --reverte.")

    if fdct_estado["verificacoes_com_evento"] < fcriterio.min_verificacoes:
        M_LOG.info(f"[=]   poucas verificações com evento para julgar "
                   f"(< {fcriterio.min_verificacoes}).")

# -------------------------------------------------------------------------------------
def monitora_par(fpth_historico: pathlib.Path, fpth_saida: pathlib.Path, fi_agora: int,
                 fcriterio: SCriterioMonitor) -> dict:
    """
    passos 2-5 para um par.

    :returns: estado gravado no <R>_<A>_monitor.json
    """
    ls_par = fpth_historico.name[:-len(CS_SUFIXO_VERIFY)]
    li_inicio = fi_agora - int(fcriterio.janela_h * 3600)

    ldct_estado = agrega(le_linhas_na_janela(fpth_historico, li_inicio, fi_agora), fcriterio)
    ldct_estado["alarmes"] = alarmes(ldct_estado, fcriterio)

    ldct_estado.update({"par": ls_par, "inicio": li_inicio, "fim": fi_agora,
                        "janela_h": fcriterio.janela_h,
                        "limiar_dbz": fcriterio.limiar_dbz,
                        "escala_fss_km": fcriterio.escala_fss_km,
                        "min_verificacoes": fcriterio.min_verificacoes})

    loga_par(ls_par, ldct_estado, fcriterio)

    fpth_saida.mkdir(parents=True, exist_ok=True)
    lpth_json = fpth_saida / f"{ls_par}{CS_SUFIXO_MONITOR}"
    lpth_temporario = fpth_saida / f".{lpth_json.name}.tmp"
    lpth_temporario.write_text(json.dumps(ldct_estado, indent=2, ensure_ascii=False))
    lpth_temporario.replace(lpth_json)

    return ldct_estado

# < main >-------------------------------------------------------------------------------

def main(flst_argv: typing.Optional[list]=None) -> int:
    """
    monitora todos os pares (ver o fluxo no topo do módulo).

    :returns: CI_SEM_ALARME, CI_COM_ALARME, CI_SEM_HISTORICO ou CI_ERRO_CONFIG
    """
    largs = arg_parse(flst_argv)

    try:
        lcriterio = criterio_do_config(uc.carrega(largs.config), largs.janela_h)

    except (OSError, ValueError) as le_erro:
        M_LOG.error(f"[x] config: {le_erro}")
        return CI_ERRO_CONFIG
    li_agora = largs.agora if largs.agora is not None else int(time.time())

    llst_historicos = lista_historicos(largs.entrada)

    if not llst_historicos:
        M_LOG.error(f"[x] nenhum *{CS_SUFIXO_VERIFY} em {largs.entrada} "
                    f"(UNT_VERIFY=true no unt_run.sh?).")
        return CI_SEM_HISTORICO

    lpth_entrada = pathlib.Path(largs.entrada)
    lpth_saida = pathlib.Path(largs.saida) if largs.saida else (
        lpth_entrada if lpth_entrada.is_dir() else lpth_entrada.parent)

    lb_alarme = False

    for lpth_historico in llst_historicos:
        ldct_estado = monitora_par(lpth_historico, lpth_saida, li_agora, lcriterio)
        lb_alarme = lb_alarme or bool(ldct_estado["alarmes"])

    return CI_COM_ALARME if lb_alarme else CI_SEM_ALARME

# -------------------------------------------------------------------------------------
# this is the bootstrap process

if "__main__" == __name__:
    # logger
    logging.basicConfig(level=logging.INFO)

    # run
    sys.exit(main())

# < the end >----------------------------------------------------------------------------
