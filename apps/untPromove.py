# -*- coding: utf-8 -*-
"""
Centro Logístico de Simulação Meteorológica
Copyright (c) 1995-2026, Instituto de Controle do Espaço Aéreo

untPromove

portão entre o treino (untTrain, que só grava candidatos) e a produção (untMain/cron,
que usa <modelos>/<backend>/atual.pt): promove um candidato a atual.pt quando a
avaliação do untEval o aprova, ou reverte para o anterior (rollback).

fluxo de main():
  1. confere o candidato: arquivo em <modelos>/<backend>/, do backend certo (.json ao
     lado, gravado pelo untTrain);
  2. lê o comparacao.json do untEval e confere que ele avaliou ESTE candidato (sob
     --rotulo, sha256 igual), o atual.pt de agora (se houver) e a persistência, no
     conjunto exigido ([promocao] split);
  3. julga ([promocao]): em cada passo, CSI e FSS (limiar/escala do config) do
     candidato >= os do atual - tolerancia e > os da persistência + margem; o mesmo
     critério em cada par radar/alcance (resultados_por_par, pares com ao menos
     min_janelas_par janelas) gera só ALERTAS: o agregado é dominado pelos pares com
     mais eco e pode esconder a piora de um radar, mas um par sozinho tem poucos
     eventos e é ruidoso demais para bloquear. Com significancia = true, também o
     intervalo do bootstrap por bloco (unt_bootstrap) da diferença candidato -
     persistência precisa ficar acima da margem;
  4. aprovado (ou --forca): anterior.pt <- o atual, atual.pt <- o candidato (trocas
     atômicas de link; o cron pega o novo na próxima passada);
  5. registra a decisão (promove, recusa, reverte) em
     <modelos>/<backend>/promocoes.jsonl (com os alertas por par).

--reverte troca atual.pt e anterior.pt (rollback), também registrado.

código de saída: 0 se promoveu, reverteu ou o candidato já era o atual; 1 se recusou
ou falhou.

$ python untPromove.py -b rainnet --candidato data/modelos/rainnet/rainnet_<UTC>.pt
                       --avaliacao data/avaliacao/<UTC>/comparacao.json
                       [--rotulo rainnet_candidato] [--modelos-dir ./data/modelos]
                       [--config config.ini] [--forca]
$ python untPromove.py -b rainnet --reverte [--modelos-dir ./data/modelos]

2026.out  mlabru  [promocao] significancia: IC do bootstrap por bloco (somas_por_bloco)
2026.out  mlabru  alertas por par radar/alcance (resultados_por_par do untEval)
2026.out  mlabru  initial version (Linux/Python)
"""
# < imports >----------------------------------------------------------------------------

# python library
import argparse
import datetime
import json
import logging
import os
import pathlib
import shutil
import sys
import typing

# model
import work.unt_bootstrap as ub
import work.unt_config as uc
import work.unt_modelos as um

# < constants >--------------------------------------------------------------------------

# registro das decisões, em <modelos>/<backend>/
CS_ARQUIVO_REGISTRO = "promocoes.jsonl"

# rótulo da persistência no comparacao.json (o mesmo de untEval.CS_PERSISTENCIA)
CS_PERSISTENCIA = "persistencia"

# < logging >----------------------------------------------------------------------------

# logger
M_LOG = logging.getLogger(__name__)
M_LOG.setLevel(logging.INFO)

# < tipos >------------------------------------------------------------------------------

class SCriterio(typing.NamedTuple):
    """
    o portão ([promocao] do config.ini).
    """
    passos: tuple
    limiar_dbz: str
    escala_fss_km: str
    tolerancia: float
    margem_persistencia: float
    split: str
    min_janelas_par: int
    # exige o intervalo do bootstrap da diferença para a persistência acima da margem
    significancia: bool = False
    min_blocos: int = 5
    # [avaliacao]: os mesmos do untEval, para os mesmos intervalos
    reamostras: int = 2000
    nivel_confianca: float = 0.90
    semente: int = 42

# -------------------------------------------------------------------------------------
class SJulgamento(typing.NamedTuple):
    """
    resultado do portão.
    """
    aprovado: bool
    motivos: list
    metricas: dict
    # par -> motivos que o recusariam naquele par (só alerta)
    alertas_por_par: typing.Optional[dict] = None

# < argumentos >-------------------------------------------------------------------------

def arg_parse(flst_argv: typing.Optional[list]=None) -> argparse.Namespace:
    """
    lê os argumentos da linha de comando.

    :param flst_argv: argumentos (None = sys.argv)

    :returns: argumentos lidos
    """
    lparser = argparse.ArgumentParser(
        description="Promove um checkpoint candidato a atual.pt (portão do untEval) "
                    "ou reverte para o anterior")

    lparser.add_argument("-b", "--backend", type=str, required=True, choices=um.CT_BACKENDS,
                         help="backend")
    lparser.add_argument("--candidato", type=str, default=None,
                         help="checkpoint candidato (<modelos>/<backend>/<backend>_<UTC>.pt)")
    lparser.add_argument("--avaliacao", type=str, default=None,
                         help="comparacao.json do untEval com o candidato")
    lparser.add_argument("--rotulo", type=str, default=None,
                         help="rótulo do candidato no comparacao.json "
                              "(padrão <backend>_candidato)")
    lparser.add_argument("--modelos-dir", type=str, default=um.diretorio_modelos_padrao(),
                         help="raiz dos checkpoints (<dir>/<backend>/)")
    lparser.add_argument("--config", type=str, default=None,
                         help="config.ini; padrão ./config.ini se existir")
    lparser.add_argument("--forca", action="store_true",
                         help="promove sem avaliação ou contra o portão (registrado)")
    lparser.add_argument("--reverte", action="store_true",
                         help="troca atual.pt e anterior.pt (rollback)")

    return lparser.parse_args(flst_argv)

# < critério e julgamento >--------------------------------------------------------------

def criterio_do_config(fconfig: uc.SConfig) -> SCriterio:
    """
    [promocao] -> SCriterio.
    """
    ltpl_passos = tuple(int(lf) for lf in fconfig.lista_reais("promocao", "passos"))

    return SCriterio(passos=ltpl_passos,
                     limiar_dbz=uc.chave_numerica(fconfig.texto("promocao", "limiar_dbz")),
                     escala_fss_km=uc.chave_numerica(fconfig.texto("promocao", "escala_fss_km")),
                     tolerancia=fconfig.real("promocao", "tolerancia"),
                     margem_persistencia=fconfig.real("promocao", "margem_persistencia"),
                     split=fconfig.texto("promocao", "split"),
                     min_janelas_par=fconfig.inteiro("promocao", "min_janelas_par"),
                     significancia=fconfig.booleano("promocao", "significancia"),
                     min_blocos=fconfig.inteiro("promocao", "min_blocos"),
                     reamostras=fconfig.inteiro("avaliacao", "reamostras"),
                     nivel_confianca=fconfig.real("avaliacao", "nivel_confianca"),
                     semente=fconfig.inteiro("avaliacao", "semente"))

# -------------------------------------------------------------------------------------
def metricas_do_passo(fdct_resultados: dict, fs_rotulo: str, fi_passo: int,
                      fcriterio: SCriterio) -> dict:
    """
    CSI e FSS de um rótulo num passo (None quando indefinidos ou ausentes).
    """
    ldct_passo = fdct_resultados.get(fs_rotulo, {}).get(str(fi_passo), {})

    lf_csi = ldct_passo.get("limiares", {}).get(fcriterio.limiar_dbz, {}).get("csi")
    lf_fss = ldct_passo.get("fss", {}).get(fcriterio.limiar_dbz, {}).get(fcriterio.escala_fss_km)

    return {"csi": lf_csi, "fss": lf_fss}

# -------------------------------------------------------------------------------------
def _compara_passo(fi_passo: int, fdct_candidato: dict, fdct_atual: typing.Optional[dict],
                   fdct_persistencia: dict, fcriterio: SCriterio,
                   fb_ignora_indefinidos: bool=False) -> list:
    """
    motivos de recusa num passo (lista vazia = passa).

    :param fb_ignora_indefinidos: métrica indefinida não é motivo (num par, falta de
                                  eventos não é piora)
    """
    llst_motivos = []

    for ls_metrica in ("csi", "fss"):
        lf_cand = fdct_candidato[ls_metrica]
        lf_pers = fdct_persistencia[ls_metrica]

        if lf_cand is None or lf_pers is None:
            if fb_ignora_indefinidos:
                continue

            ls_quem = "candidato" if lf_cand is None else CS_PERSISTENCIA
            if lf_cand is None and lf_pers is None:
                ls_quem = f"candidato e {CS_PERSISTENCIA}"

            llst_motivos.append(f"passo {fi_passo}: {ls_metrica} indefinido ({ls_quem}; "
                                f"sem eventos no conjunto avaliado?)")
            continue

        if lf_cand <= lf_pers + fcriterio.margem_persistencia:
            llst_motivos.append(f"passo {fi_passo}: {ls_metrica} {lf_cand:.3f} não supera a "
                                f"persistência {lf_pers:.3f} (+{fcriterio.margem_persistencia})")

        lf_atual = None if fdct_atual is None else fdct_atual[ls_metrica]

        if lf_atual is not None and lf_cand < lf_atual - fcriterio.tolerancia:
            llst_motivos.append(f"passo {fi_passo}: {ls_metrica} {lf_cand:.3f} abaixo do "
                                f"atual {lf_atual:.3f} (-{fcriterio.tolerancia})")

    return llst_motivos

# -------------------------------------------------------------------------------------
def julga(fdct_resultados: dict, fs_rotulo: str, fs_rotulo_atual: typing.Optional[str],
          fcriterio: SCriterio, fb_ignora_indefinidos: bool=False) -> SJulgamento:
    """
    passo 3: o portão.

    :param fdct_resultados: "resultados" do comparacao.json (ou os de um par)
    :param fs_rotulo: rótulo do candidato
    :param fs_rotulo_atual: rótulo do atual (None = não há atual.pt)
    :param fcriterio: [promocao]
    :param fb_ignora_indefinidos: ver _compara_passo

    :returns: SJulgamento
    """
    llst_motivos = []
    ldct_metricas = {}

    for li_passo in fcriterio.passos:
        ldct_cand = metricas_do_passo(fdct_resultados, fs_rotulo, li_passo, fcriterio)
        ldct_pers = metricas_do_passo(fdct_resultados, CS_PERSISTENCIA, li_passo, fcriterio)
        ldct_atual = None

        if fs_rotulo_atual is not None:
            ldct_atual = metricas_do_passo(fdct_resultados, fs_rotulo_atual, li_passo, fcriterio)

        ldct_metricas[str(li_passo)] = {"candidato": ldct_cand, "atual": ldct_atual,
                                        CS_PERSISTENCIA: ldct_pers}

        llst_motivos += _compara_passo(li_passo, ldct_cand, ldct_atual, ldct_pers, fcriterio,
                                       fb_ignora_indefinidos)

    return SJulgamento(not llst_motivos, llst_motivos, ldct_metricas)

# -------------------------------------------------------------------------------------
def julga_significancia(fdct_comparacao: dict, fs_rotulo: str,
                        fcriterio: SCriterio) -> tuple:
    """
    passo 3, com [promocao] significancia: em cada passo, o intervalo do bootstrap
    pareado por bloco (somas_por_bloco do untEval) da diferença candidato - persistência,
    em CSI e FSS, precisa ficar todo acima da margem.

    :returns: (motivos de recusa, passo -> {"csi"|"fss": {"ic", "p_melhor"}})
    """
    ldct_somas = fdct_comparacao.get("somas_por_bloco") or {}

    if len(ldct_somas) < fcriterio.min_blocos:
        return [f"significância: {len(ldct_somas)} bloco(s) em somas_por_bloco, o portão "
                f"exige {fcriterio.min_blocos} (untEval antigo ou conjunto pequeno)"], {}

    lnp_pesos = ub.pesos_reamostrados(len(ldct_somas), fcriterio.reamostras, fcriterio.semente)
    ldct_metricas = {"csi": ("csi", fcriterio.limiar_dbz, None),
                     "fss": ("fss", fcriterio.limiar_dbz, fcriterio.escala_fss_km)}

    llst_motivos = []
    ldct_intervalos = {}

    for li_passo in fcriterio.passos:
        for ls_metrica, lt_metrica in ldct_metricas.items():
            ldct_dif = ub.compara(ldct_somas, fs_rotulo, CS_PERSISTENCIA, str(li_passo),
                                  lt_metrica, lnp_pesos, fcriterio.nivel_confianca)
            ldct_intervalos.setdefault(str(li_passo), {})[ls_metrica] = ldct_dif

            if ldct_dif["ic"] is None:
                llst_motivos.append(f"passo {li_passo}: {ls_metrica} sem intervalo "
                                    f"(indefinido no bootstrap)")
                continue

            lf_lo, lf_hi = ldct_dif["ic"]

            if lf_lo <= fcriterio.margem_persistencia:
                llst_motivos.append(
                    f"passo {li_passo}: {ls_metrica}: IC "
                    f"{100 * fcriterio.nivel_confianca:.0f}% da diferença para a "
                    f"persistência [{lf_lo:+.3f}, {lf_hi:+.3f}] não fica acima de "
                    f"{fcriterio.margem_persistencia} (p={ldct_dif['p_melhor']:.2f})")

    return llst_motivos, ldct_intervalos

# -------------------------------------------------------------------------------------
def alertas_por_par(fdct_comparacao: dict, fs_rotulo: str,
                    fs_rotulo_atual: typing.Optional[str], fcriterio: SCriterio) -> dict:
    """
    passo 3, por par: o critério do portão em cada par radar/alcance com ao menos
    min_janelas_par janelas (métrica indefinida não conta). Com um par só, o par é o
    agregado e não há o que alertar; comparacao.json sem resultados_por_par (untEval
    antigo) também não alerta.

    :returns: par -> motivos (só os pares que o critério recusaria)
    """
    ldct_por_par = fdct_comparacao.get("resultados_por_par") or {}
    ldct_janelas = fdct_comparacao.get("janelas_por_par") or {}

    if len(ldct_por_par) < 2:
        return {}

    llst_poucas = sorted(ls_par for ls_par in ldct_por_par
                         if ldct_janelas.get(ls_par, 0) < fcriterio.min_janelas_par)

    if llst_poucas:
        M_LOG.info(f"[=] pares não julgados (< {fcriterio.min_janelas_par} janelas): "
                   f"{','.join(llst_poucas)}.")

    ldct_alertas = {}

    for ls_par, ldct_resultados in sorted(ldct_por_par.items()):
        if ls_par in llst_poucas:
            continue

        ljulgamento = julga(ldct_resultados, fs_rotulo, fs_rotulo_atual, fcriterio,
                            fb_ignora_indefinidos=True)

        if ljulgamento.motivos:
            ldct_alertas[ls_par] = ljulgamento.motivos

    return ldct_alertas

# < conferências >-----------------------------------------------------------------------

def confere_candidato(fs_candidato: str, fs_modelos_dir: str, fs_backend: str) -> tuple:
    """
    passo 1: o candidato existe, mora em <modelos>/<backend>/ (o link é relativo) e é
    do backend.

    :returns: (caminho real, resumo .json)

    :raises ValueError: com o motivo
    """
    lpth_candidato = pathlib.Path(fs_candidato).resolve()
    lpth_dir = (pathlib.Path(fs_modelos_dir) / fs_backend).resolve()

    if not lpth_candidato.is_file():
        raise ValueError(f"candidato ausente: {fs_candidato}")

    if lpth_candidato.parent != lpth_dir:
        raise ValueError(f"o candidato precisa estar em {lpth_dir} (está em "
                         f"{lpth_candidato.parent})")

    ldct_resumo = um.le_resumo_checkpoint(lpth_candidato)

    if ldct_resumo is None:
        raise ValueError(f"{lpth_candidato.name}: sem o .json do untTrain ao lado")

    ls_backend = ldct_resumo.get("backend") or ldct_resumo.get("hparams", {}).get("backend")

    if ls_backend != fs_backend:
        raise ValueError(f"{lpth_candidato.name}: checkpoint do backend {ls_backend}, "
                         f"não {fs_backend}")

    return lpth_candidato, ldct_resumo

# -------------------------------------------------------------------------------------
def confere_avaliacao(fdct_comparacao: dict, fs_rotulo: str, fpth_candidato: pathlib.Path,
                      fpth_atual: typing.Optional[pathlib.Path], fs_backend: str,
                      fcriterio: SCriterio) -> typing.Optional[str]:
    """
    passo 2: a avaliação é deste candidato, do atual.pt de agora e do conjunto exigido.

    :returns: rótulo do atual no comparacao.json (None sem atual.pt)

    :raises ValueError: com o motivo
    """
    if fcriterio.split != fdct_comparacao.get("split"):
        raise ValueError(f"avaliação no conjunto '{fdct_comparacao.get('split')}'; o "
                         f"portão exige '{fcriterio.split}'")

    ldct_desempenho = fdct_comparacao.get("desempenho", {})

    if fs_rotulo not in ldct_desempenho:
        raise ValueError(f"rótulo '{fs_rotulo}' ausente da avaliação (untEval "
                         f"--checkpoint {fs_rotulo}=<candidato>)")

    if ldct_desempenho[fs_rotulo].get("sha256") != um.sha256_de_arquivo(fpth_candidato):
        raise ValueError(f"a avaliação de '{fs_rotulo}' não é de {fpth_candidato.name} "
                         f"(sha256 diferente)")

    if fpth_atual is None:
        return None

    if fs_backend not in ldct_desempenho:
        raise ValueError(f"a avaliação não inclui o atual.pt de {fs_backend}")

    if ldct_desempenho[fs_backend].get("sha256") != um.sha256_de_arquivo(fpth_atual):
        raise ValueError(f"atual.pt de {fs_backend} mudou depois da avaliação; reavalie")

    return fs_backend

# < links >------------------------------------------------------------------------------

def _copia_atomica(fpth_origem: pathlib.Path, fpth_destino: pathlib.Path) -> None:
    """
    cópia via temporário + rename (sistemas de arquivos sem links simbólicos).
    """
    lpth_temporario = fpth_destino.parent / f".{fpth_destino.name}.tmp"
    shutil.copyfile(fpth_origem, lpth_temporario)
    os.replace(lpth_temporario, fpth_destino)

# -------------------------------------------------------------------------------------
def _preserva_como_anterior(fpth_atual: pathlib.Path) -> None:
    """
    anterior.pt passa a apontar para o que atual.pt aponta agora.
    """
    if fpth_atual.is_symlink():
        um.aponta_link(fpth_atual.resolve(), um.CS_CHECKPOINT_ANTERIOR)
        return

    _copia_atomica(fpth_atual, fpth_atual.parent / um.CS_CHECKPOINT_ANTERIOR)

# -------------------------------------------------------------------------------------
def promove(fpth_candidato: pathlib.Path, fs_modelos_dir: str, fs_backend: str) -> None:
    """
    passo 4: anterior.pt <- atual, atual.pt <- candidato.
    """
    lpth_atual = um.caminho_checkpoint_atual(fs_modelos_dir, fs_backend)

    if lpth_atual.exists():
        _preserva_como_anterior(lpth_atual)

    um.aponta_link(fpth_candidato, um.CS_CHECKPOINT_ATUAL)

# -------------------------------------------------------------------------------------
def reverte(fs_modelos_dir: str, fs_backend: str) -> tuple:
    """
    troca atual.pt e anterior.pt.

    :returns: (novo atual, novo anterior), nomes dos arquivos reais

    :raises ValueError: sem atual.pt ou sem anterior.pt
    """
    lpth_atual = um.caminho_checkpoint_atual(fs_modelos_dir, fs_backend)
    lpth_anterior = um.caminho_checkpoint_anterior(fs_modelos_dir, fs_backend)

    if not (lpth_atual.exists() and lpth_anterior.exists()):
        raise ValueError(f"rollback precisa de {lpth_atual.name} e {lpth_anterior.name} em "
                         f"{lpth_atual.parent}")

    if lpth_atual.is_symlink() and lpth_anterior.is_symlink():
        lpth_era_atual = lpth_atual.resolve()
        lpth_era_anterior = lpth_anterior.resolve()

        um.aponta_link(lpth_era_anterior, um.CS_CHECKPOINT_ATUAL)
        um.aponta_link(lpth_era_atual, um.CS_CHECKPOINT_ANTERIOR)

        return lpth_era_anterior.name, lpth_era_atual.name

    # sem links: troca as cópias
    lpth_troca = lpth_atual.parent / ".troca.tmp"
    shutil.copyfile(lpth_atual, lpth_troca)
    _copia_atomica(lpth_anterior, lpth_atual)
    os.replace(lpth_troca, lpth_anterior)

    return lpth_atual.name, lpth_anterior.name

# < registro >---------------------------------------------------------------------------

def registra(fs_modelos_dir: str, fs_backend: str, fdct_registro: dict) -> None:
    """
    passo 5: uma linha em <modelos>/<backend>/promocoes.jsonl.
    """
    lpth_registro = pathlib.Path(fs_modelos_dir) / fs_backend / CS_ARQUIVO_REGISTRO
    lpth_registro.parent.mkdir(parents=True, exist_ok=True)

    ldct = {"quando": datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d%H%M%S"),
            "backend": fs_backend, **fdct_registro}

    with lpth_registro.open("a", encoding="utf-8") as larq:
        larq.write(json.dumps(ldct, ensure_ascii=False) + "\n")

# -------------------------------------------------------------------------------------
def _avisa_escopo_reduzido(fdct_resumo_candidato: dict,
                           fpth_atual: typing.Optional[pathlib.Path]) -> list:
    """
    pares radar/alcance que o atual cobre e o candidato não (deixariam de rodar).
    """
    if fpth_atual is None:
        return []

    ldct_resumo_atual = um.le_resumo_checkpoint(fpth_atual) or {}
    lset_atual = set(ldct_resumo_atual.get("escopo", {}).get("pares", []))
    lset_candidato = set(fdct_resumo_candidato.get("escopo", {}).get("pares", []))

    llst_perdidos = sorted(lset_atual - lset_candidato)

    if llst_perdidos:
        M_LOG.warning(f"[!] o candidato não cobre {','.join(llst_perdidos)}, que o atual "
                      f"cobre: esses pares deixam de rodar este backend.")

    return llst_perdidos

# < main >-------------------------------------------------------------------------------

def _main_reverte(fargs: argparse.Namespace) -> int:
    """
    --reverte.
    """
    try:
        ls_novo_atual, ls_novo_anterior = reverte(fargs.modelos_dir, fargs.backend)

    except ValueError as le_erro:
        M_LOG.error(f"[x] {le_erro}")
        return 1

    registra(fargs.modelos_dir, fargs.backend, {"acao": "reverte", "atual": ls_novo_atual,
                                                "anterior": ls_novo_anterior})
    M_LOG.info(f"[✓] {fargs.backend}: atual.pt -> {ls_novo_atual} (rollback; anterior.pt "
               f"-> {ls_novo_anterior}).")

    return 0

# -------------------------------------------------------------------------------------
def _julga_avaliacao(fargs: argparse.Namespace, fpth_candidato: pathlib.Path,
                     fpth_atual: typing.Optional[pathlib.Path]) -> SJulgamento:
    """
    passos 2 e 3. Erro de conferência vira recusa (com --forca, só aviso).
    """
    if fargs.avaliacao is None:
        return SJulgamento(False, ["sem --avaliacao (comparacao.json do untEval)"], {})

    lcriterio = criterio_do_config(uc.carrega(fargs.config))
    ls_rotulo = fargs.rotulo or f"{fargs.backend}_candidato"

    try:
        ldct_comparacao = json.loads(pathlib.Path(fargs.avaliacao).read_text())
        ls_rotulo_atual = confere_avaliacao(ldct_comparacao, ls_rotulo, fpth_candidato,
                                            fpth_atual, fargs.backend, lcriterio)

    except (OSError, ValueError) as le_erro:
        return SJulgamento(False, [str(le_erro)], {})

    ljulgamento = julga(ldct_comparacao.get("resultados", {}), ls_rotulo, ls_rotulo_atual,
                        lcriterio)

    if lcriterio.significancia:
        llst_motivos, ldct_intervalos = julga_significancia(ldct_comparacao, ls_rotulo,
                                                            lcriterio)
        for ls_passo, ldct in ldct_intervalos.items():
            ljulgamento.metricas.setdefault(ls_passo, {})["dif_persistencia"] = ldct

        ljulgamento = ljulgamento._replace(aprovado=ljulgamento.aprovado and not llst_motivos,
                                           motivos=ljulgamento.motivos + llst_motivos)

    return ljulgamento._replace(alertas_por_par=alertas_por_par(ldct_comparacao, ls_rotulo,
                                                                ls_rotulo_atual, lcriterio))

# -------------------------------------------------------------------------------------
def _avisa_alertas_por_par(fdct_alertas: dict) -> None:
    """
    um aviso por motivo de cada par que o critério recusaria.
    """
    for ls_par, llst_motivos in fdct_alertas.items():
        for ls_motivo in llst_motivos:
            M_LOG.warning(f"[!] {ls_par}: {ls_motivo}")

    if fdct_alertas:
        M_LOG.warning(f"[!] o critério do portão falha em {len(fdct_alertas)} par(es) "
                      f"({','.join(fdct_alertas)}), embora o portão julgue o agregado.")

# -------------------------------------------------------------------------------------
def main(flst_argv: typing.Optional[list]=None) -> int:
    """
    promove, recusa ou reverte (ver o fluxo no topo do módulo).

    :returns: 0 se promoveu/reverteu (ou o candidato já era o atual), 1 caso contrário
    """
    largs = arg_parse(flst_argv)

    if largs.reverte:
        return _main_reverte(largs)

    if largs.candidato is None:
        M_LOG.error("[x] informe --candidato (ou --reverte).")
        return 1

    # 1. candidato
    try:
        lpth_candidato, ldct_resumo = confere_candidato(largs.candidato, largs.modelos_dir,
                                                        largs.backend)
    except ValueError as le_erro:
        M_LOG.error(f"[x] {le_erro}")
        return 1

    lpth_link_atual = um.caminho_checkpoint_atual(largs.modelos_dir, largs.backend)
    lpth_atual = lpth_link_atual.resolve() if lpth_link_atual.exists() else None

    if lpth_atual == lpth_candidato:
        M_LOG.info(f"[=] {lpth_candidato.name} já é o atual.pt de {largs.backend}.")
        return 0

    # 2-3. avaliação e portão
    ljulgamento = _julga_avaliacao(largs, lpth_candidato, lpth_atual)
    ldct_alertas = ljulgamento.alertas_por_par or {}
    _avisa_alertas_por_par(ldct_alertas)

    ldct_registro = {"candidato": lpth_candidato.name,
                     "atual_antes": None if lpth_atual is None else lpth_atual.name,
                     "avaliacao": largs.avaliacao,
                     "forcado": False,
                     "motivos": ljulgamento.motivos,
                     "metricas": ljulgamento.metricas,
                     "alertas_por_par": ldct_alertas,
                     "pares_perdidos": _avisa_escopo_reduzido(ldct_resumo, lpth_atual)}

    if not ljulgamento.aprovado and not largs.forca:
        for ls_motivo in ljulgamento.motivos:
            M_LOG.warning(f"[!] {ls_motivo}")

        registra(largs.modelos_dir, largs.backend, {"acao": "recusa", **ldct_registro})
        M_LOG.error(f"[x] {largs.backend}: {lpth_candidato.name} recusado; atual.pt mantido.")
        return 1

    if not ljulgamento.aprovado:
        ldct_registro["forcado"] = True
        M_LOG.warning(f"[!] --forca: promovendo apesar de {len(ljulgamento.motivos)} "
                      f"motivo(s) de recusa.")

    # 4-5. promoção
    promove(lpth_candidato, largs.modelos_dir, largs.backend)
    registra(largs.modelos_dir, largs.backend, {"acao": "promove", **ldct_registro})

    M_LOG.info(f"[✓] {largs.backend}: atual.pt -> {lpth_candidato.name}"
               f"{'' if lpth_atual is None else f' (anterior.pt -> {lpth_atual.name})'}.")

    return 0

# -------------------------------------------------------------------------------------
# this is the bootstrap process

if "__main__" == __name__:
    # logger
    logging.basicConfig(level=logging.INFO)

    # run
    sys.exit(main())

# < the end >----------------------------------------------------------------------------
