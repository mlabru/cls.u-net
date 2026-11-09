# -*- coding: utf-8 -*-
"""
Centro Logístico de Simulação Meteorológica
Copyright (c) 1995-2026, Instituto de Controle do Espaço Aéreo

unt_modelos

fábrica das três redes e o formato do checkpoint.

backends (o mesmo nome em untMain, untTrain, untEval e nos diretórios de modelos):
  - "pytorch": RedeICEA (unt_rede_icea), previsão direta condicionada ao lead,
               entropia cruzada sobre bins de intensidade;
  - "rainnet": RainNet v1.0 (rainnet/rainnet.py, porte PyTorch), t+1 recursivo;
  - "smaat":   SmaAt-UNet (smaat/, vendorizado), t+1 recursivo ou direto (6 saídas).

checkpoint (torch.save de um dicionário; carrega em CPU com map_location):
  {"versao", "backend", "hparams", "transformacao", "estado", "epoca", "perda_validacao",
   "historico", "tempo_treino_s", "device", "manifesto", "torch", "criado_em",
   "escopo", "manifesto_sha256", "git_commit"}  (os três últimos a partir da versão 2)
gravado em <modelos>/<backend>/<backend>_<AAAAMMDDhhmmss>.pt. O link simbólico
<modelos>/<backend>/atual.pt (o que untMain/untEval usam) só muda na PROMOÇÃO
(untPromove, ou untTrain --promove em dev); anterior.pt guarda o promovido antes dele.

escopo: os pares radar/alcance do manifesto de treino. O untMain só roda um backend num
par coberto pelo escopo do seu atual.pt (lido do .json ao lado, sem carregar os pesos).

2026.out  mlabru  hparams passos_colapso, tentativas_colapso
2026.out  mlabru  hparams media_metrica e, em rainnet/smaat, limites_peso_dbz/peso_dbz
2026.out  mlabru  hparam max_nao_finitos
2026.out  mlabru  hparams min_pixels_evento_val e passos_rollout (só no modo recursivo)
2026.out  mlabru  hparams selecao/val_cada/max_janelas_val/clip_grad, limiar/escala da
                  seleção (do [promocao]), peso_bins = auto
2026.out  mlabru  treino não promove mais sozinho (fb_promove); anterior.pt; escopo e
                  proveniência no checkpoint (versão 2, a 1 ainda é lida)
2026.out  mlabru  initial version (Linux/Python)
"""
# < imports >----------------------------------------------------------------------------

# python library
import datetime
import hashlib
import json
import logging
import os
import pathlib
import shutil
import subprocess
import typing

# pyTorch
import torch
from torch import nn

# model
import work.unt_config as uc
import work.unt_dados as ud
import work.unt_rede_icea as ur

# redes vendorizadas
from rainnet.rainnet import RainNet
from smaat.SmaAt_UNet import SmaAt_UNet

# < constants >--------------------------------------------------------------------------

# backends conhecidos, na ordem do log
CT_BACKENDS = ("pytorch", "rainnet", "smaat")

# versão do formato do checkpoint (gravada) e as que ainda são lidas
CI_VERSAO_CHECKPOINT = 2
CT_VERSOES_LIDAS = (1, 2)

# nome do link para o checkpoint em uso e para o promovido antes dele (rollback)
CS_CHECKPOINT_ATUAL = "atual.pt"
CS_CHECKPOINT_ANTERIOR = "anterior.pt"

# estratégias de previsão
CS_MODO_RECURSIVO = "recursivo"
CS_MODO_DIRETO = "direto"
CS_MODO_LEAD = "lead"

# critério de escolha da melhor época
CS_SELECAO_METRICA = "metrica"
CS_SELECAO_PERDA = "perda"

# peso_bins calculado das frequências do treino
CS_PESO_BINS_AUTO = "auto"

# < logging >----------------------------------------------------------------------------

# logger
M_LOG = logging.getLogger(__name__)
M_LOG.setLevel(logging.INFO)

# < hiperparâmetros >--------------------------------------------------------------------

def _valor_do_backend(fconfig: uc.SConfig, fs_backend: str, fs_opcao: str, ff_tipo):
    """
    opção do backend: a seção do backend se ela tem a opção, senão o [treino].
    """
    ls_secao = fconfig.secao_do_backend(fs_backend, fs_opcao)

    if ff_tipo is bool:
        return fconfig.booleano(ls_secao, fs_opcao)

    if ff_tipo is int:
        return fconfig.inteiro(ls_secao, fs_opcao)

    if ff_tipo is float:
        return fconfig.real(ls_secao, fs_opcao)

    if ff_tipo is list:
        return fconfig.lista_reais(ls_secao, fs_opcao)

    return fconfig.texto(ls_secao, fs_opcao)

# -------------------------------------------------------------------------------------
def hiperparametros(fconfig: uc.SConfig, fs_backend: str, fi_num_entradas: int,
                    fi_num_alvos: int, fdct_sobrescritos: typing.Optional[dict]=None) -> dict:
    """
    hiperparâmetros de treino e de arquitetura de um backend.

    precedência: fdct_sobrescritos (CLI, valores None ignorados) > seção do backend >
    [treino] > defaults de unt_config.

    :param fconfig: configuração
    :param fs_backend: "pytorch", "rainnet" ou "smaat"
    :param fi_num_entradas: quadros de entrada (manifesto)
    :param fi_num_alvos: passos previstos (manifesto)
    :param fdct_sobrescritos: valores da linha de comando

    :returns: dicionário plano (gravado no checkpoint)

    :raises ValueError: backend desconhecido
    """
    if fs_backend not in CT_BACKENDS:
        raise ValueError(f"backend desconhecido: {fs_backend}")

    ldct = {"backend": fs_backend, "num_entradas": fi_num_entradas, "num_alvos": fi_num_alvos}

    for ls_opcao, ltipo in (("device", str), ("epocas", int), ("batch", int), ("lr", float),
                            ("paciencia", int), ("recorte_px", int), ("largura_base", int),
                            ("workers", int), ("semente", int), ("amp", bool),
                            ("augment", bool), ("min_eco_recorte", float),
                            ("tentativas_recorte", int), ("amostras_por_epoca", int),
                            ("selecao", str), ("val_cada", int), ("max_janelas_val", int),
                            ("min_pixels_evento_val", int), ("clip_grad", float),
                            ("max_nao_finitos", float), ("passos_rollout", int),
                            ("media_metrica", int), ("passos_colapso", int),
                            ("tentativas_colapso", int)):
        ldct[ls_opcao] = _valor_do_backend(fconfig, fs_backend, ls_opcao, ltipo)

    # perda de regressão ponderada pela intensidade (o pytorch tem peso_bins)
    if "pytorch" != fs_backend:
        for ls_opcao in ("limites_peso_dbz", "peso_dbz"):
            ldct[ls_opcao] = _valor_do_backend(fconfig, fs_backend, ls_opcao, list)

    # a seleção de época usa o limiar/escala do portão de promoção
    ldct["limiar_selecao"] = uc.chave_numerica(fconfig.texto("promocao", "limiar_dbz"))
    ldct["escala_selecao"] = uc.chave_numerica(fconfig.texto("promocao", "escala_fss_km"))

    if "pytorch" == fs_backend:
        ldct["bins_dbz"] = fconfig.lista_reais("pytorch", "bins_dbz")
        ldct["peso_bins"] = _peso_bins(fconfig)
        ldct["blocos_por_nivel"] = fconfig.inteiro("pytorch", "blocos_por_nivel")
        ldct["modo"] = CS_MODO_LEAD

    elif "smaat" == fs_backend:
        ldct["kernels_per_layer"] = fconfig.inteiro("smaat", "kernels_per_layer")
        ldct["reduction_ratio"] = fconfig.inteiro("smaat", "reduction_ratio")
        ldct["modo"] = fconfig.texto("smaat", "modo")

    else:
        ldct["modo"] = CS_MODO_RECURSIVO

    for ls_chave, lx_valor in (fdct_sobrescritos or {}).items():
        if lx_valor is not None:
            ldct[ls_chave] = lx_valor

    _valida(ldct)

    return ldct

# -------------------------------------------------------------------------------------
def _peso_bins(fconfig: uc.SConfig) -> typing.Union[list, str]:
    """
    [pytorch] peso_bins: lista de pesos, ou "auto" (resolvido pelo untTrain com as
    frequências do conjunto de treino).
    """
    if CS_PESO_BINS_AUTO == fconfig.texto("pytorch", "peso_bins").lower():
        return CS_PESO_BINS_AUTO

    return fconfig.lista_reais("pytorch", "peso_bins")

# -------------------------------------------------------------------------------------
def _valida(fdct_hparams: dict) -> None:
    """
    recusa combinações inválidas antes de construir a rede.

    :raises ValueError: com a opção culpada
    """
    if fdct_hparams["modo"] not in (CS_MODO_RECURSIVO, CS_MODO_DIRETO, CS_MODO_LEAD):
        raise ValueError(f"modo inválido: {fdct_hparams['modo']}")

    if fdct_hparams["selecao"] not in (CS_SELECAO_METRICA, CS_SELECAO_PERDA):
        raise ValueError(f"selecao inválida: {fdct_hparams['selecao']} "
                         f"({CS_SELECAO_METRICA} ou {CS_SELECAO_PERDA})")

    if fdct_hparams["val_cada"] < 1:
        raise ValueError(f"val_cada ({fdct_hparams['val_cada']}) precisa ser >= 1")

    if "pytorch" == fdct_hparams["backend"] and CS_PESO_BINS_AUTO != fdct_hparams["peso_bins"]:
        li_classes = len(fdct_hparams["bins_dbz"]) + 1

        if len(fdct_hparams["peso_bins"]) != li_classes:
            raise ValueError(f"[pytorch] peso_bins precisa de {li_classes} valores "
                             f"(len(bins_dbz) + 1)")

    if fdct_hparams.get("media_metrica", 1) < 1:
        raise ValueError(f"media_metrica ({fdct_hparams['media_metrica']}) precisa ser >= 1")

    if fdct_hparams.get("passos_colapso", 0) < 0:
        raise ValueError(f"passos_colapso ({fdct_hparams['passos_colapso']}) precisa ser >= 0")

    if fdct_hparams.get("tentativas_colapso", 1) < 1:
        raise ValueError(f"tentativas_colapso ({fdct_hparams['tentativas_colapso']}) "
                         f"precisa ser >= 1")

    _valida_peso_dbz(fdct_hparams)

    li_rollout = fdct_hparams.get("passos_rollout", 1)

    if not 1 <= li_rollout <= fdct_hparams["num_alvos"]:
        raise ValueError(f"passos_rollout ({li_rollout}) precisa estar entre 1 e "
                         f"num_alvos ({fdct_hparams['num_alvos']})")

    if li_rollout > 1 and CS_MODO_RECURSIVO != fdct_hparams["modo"]:
        raise ValueError(f"passos_rollout ({li_rollout}) > 1 só vale no modo "
                         f"{CS_MODO_RECURSIVO} (o {fdct_hparams['backend']} está em "
                         f"{fdct_hparams['modo']})")

    li_recorte = fdct_hparams["recorte_px"]

    if li_recorte and li_recorte % ud.CI_MULTIPLO_PADDING:
        raise ValueError(f"recorte_px ({li_recorte}) precisa ser múltiplo de "
                         f"{ud.CI_MULTIPLO_PADDING} (ou 0 = grade inteira)")

# -------------------------------------------------------------------------------------
def _valida_peso_dbz(fdct_hparams: dict) -> None:
    """
    limites_peso_dbz crescentes e um peso positivo por faixa (len(limites) + 1), ou os
    dois vazios.

    :raises ValueError: com a opção culpada
    """
    llst_limites = fdct_hparams.get("limites_peso_dbz") or []
    llst_pesos = fdct_hparams.get("peso_dbz") or []

    if not llst_limites and not llst_pesos:
        return

    if len(llst_pesos) != len(llst_limites) + 1:
        raise ValueError(f"peso_dbz precisa de {len(llst_limites) + 1} valores "
                         f"(len(limites_peso_dbz) + 1)")

    if any(lf_a >= lf_b for lf_a, lf_b in zip(llst_limites, llst_limites[1:])):
        raise ValueError(f"limites_peso_dbz precisa ser crescente: {llst_limites}")

    if any(lf <= 0.0 for lf in llst_pesos):
        raise ValueError(f"peso_dbz precisa ser positivo: {llst_pesos}")

# < construção >-------------------------------------------------------------------------

def transformacao_padrao(fdct_hparams: dict):
    """
    a transformação de dados do backend (unt_dados), a partir dos hiperparâmetros.
    """
    ls_backend = fdct_hparams["backend"]

    if "rainnet" == ls_backend:
        return ud.TransformacaoRainNet()

    if "smaat" == ls_backend:
        return ud.TransformacaoSmaat()

    return ud.TransformacaoPytorch(fdct_hparams["bins_dbz"])

# -------------------------------------------------------------------------------------
def constroi_rede(fdct_hparams: dict) -> nn.Module:
    """
    instancia a rede do backend.

    :param fdct_hparams: saída de hiperparametros() (ou lida do checkpoint)

    :returns: nn.Module (pesos aleatórios)
    """
    ls_backend = fdct_hparams["backend"]
    li_entradas = fdct_hparams["num_entradas"]
    li_largura = fdct_hparams["largura_base"]

    if "rainnet" == ls_backend:
        return RainNet(fi_canais_entrada=li_entradas, fi_largura_base=li_largura)

    if "smaat" == ls_backend:
        li_saidas = fdct_hparams["num_alvos"] if CS_MODO_DIRETO == fdct_hparams["modo"] else 1

        return SmaAt_UNet(n_channels=li_entradas, n_classes=li_saidas,
                          kernels_per_layer=fdct_hparams["kernels_per_layer"],
                          reduction_ratio=fdct_hparams["reduction_ratio"],
                          largura_base=li_largura)

    return ur.RedeICEA(fi_canais_entrada=li_entradas + 1,
                       fi_num_leads=fdct_hparams["num_alvos"],
                       fi_num_classes=len(fdct_hparams["bins_dbz"]) + 1,
                       fi_largura_base=li_largura,
                       fi_blocos_por_nivel=fdct_hparams["blocos_por_nivel"])

# -------------------------------------------------------------------------------------
def conta_parametros(frede: nn.Module) -> int:
    """
    número de parâmetros treináveis.
    """
    return sum(lparam.numel() for lparam in frede.parameters() if lparam.requires_grad)

# < checkpoint >-------------------------------------------------------------------------

def diretorio_modelos_padrao() -> str:
    """
    padrão de --modelos-dir: $UNT_MODELOS_DIR ou ./data/modelos.
    """
    return os.environ.get("UNT_MODELOS_DIR", "./data/modelos")

# -------------------------------------------------------------------------------------
def caminho_checkpoint_atual(fs_modelos_dir: str, fs_backend: str) -> pathlib.Path:
    """
    <modelos>/<backend>/atual.pt
    """
    return pathlib.Path(fs_modelos_dir) / fs_backend / CS_CHECKPOINT_ATUAL

# -------------------------------------------------------------------------------------
def caminho_checkpoint_anterior(fs_modelos_dir: str, fs_backend: str) -> pathlib.Path:
    """
    <modelos>/<backend>/anterior.pt
    """
    return pathlib.Path(fs_modelos_dir) / fs_backend / CS_CHECKPOINT_ANTERIOR

# -------------------------------------------------------------------------------------
def aponta_link(fpth_checkpoint: pathlib.Path, fs_nome_link: str) -> None:
    """
    faz <dir do checkpoint>/<fs_nome_link> apontar para o checkpoint: link simbólico
    relativo; sem suporte a links, uma cópia.
    """
    lpth_link = fpth_checkpoint.parent / fs_nome_link
    lpth_temporario = fpth_checkpoint.parent / f".{fs_nome_link}.tmp"

    if lpth_temporario.is_symlink() or lpth_temporario.exists():
        lpth_temporario.unlink()

    try:
        lpth_temporario.symlink_to(fpth_checkpoint.name)

    except OSError:
        shutil.copyfile(fpth_checkpoint, lpth_temporario)

    # troca atômica: um leitor nunca vê o link ausente
    os.replace(lpth_temporario, lpth_link)

# -------------------------------------------------------------------------------------
def _aponta_atual(fpth_checkpoint: pathlib.Path) -> None:
    """
    atualiza atual.pt.
    """
    aponta_link(fpth_checkpoint, CS_CHECKPOINT_ATUAL)

# < escopo e proveniência >--------------------------------------------------------------

def escopo_das_janelas(flst_janelas: list) -> dict:
    """
    pares radar/alcance cobertos por um manifesto (o escopo de aplicação do modelo).

    :param flst_janelas: unt_janelas.SJanela (todas, de todos os conjuntos)

    :returns: {"pares": ["STI_250", ...]} em ordem
    """
    lset_pares = {f"{ljanela.radar}_{ljanela.alcance}" for ljanela in flst_janelas}

    return {"pares": sorted(lset_pares)}

# -------------------------------------------------------------------------------------
def sha256_de_arquivo(fpth: typing.Union[str, pathlib.Path]) -> typing.Optional[str]:
    """
    sha256 de um arquivo (None se não existe).
    """
    lpth = pathlib.Path(fpth)

    if not lpth.is_file():
        return None

    return hashlib.sha256(lpth.read_bytes()).hexdigest()

# -------------------------------------------------------------------------------------
def commit_git() -> typing.Optional[str]:
    """
    commit do código que treinou (None fora de um repositório git).
    """
    lpth_aqui = pathlib.Path(__file__).resolve().parent

    try:
        lresultado = subprocess.run(["git", "-C", str(lpth_aqui), "rev-parse", "HEAD"],
                                    capture_output=True, text=True, timeout=10, check=False)

    except (OSError, subprocess.SubprocessError):
        return None

    if 0 != lresultado.returncode:
        return None

    return lresultado.stdout.strip() or None

# -------------------------------------------------------------------------------------
def le_resumo_checkpoint(fpth_checkpoint: typing.Union[str, pathlib.Path]) -> typing.Optional[dict]:
    """
    o .json gravado ao lado do checkpoint (untTrain), sem carregar os pesos. Aceita o
    link (atual.pt/anterior.pt): segue até o arquivo real.

    :returns: dicionário, ou None se não há .json legível
    """
    lpth_json = pathlib.Path(fpth_checkpoint).resolve().with_suffix(".json")

    try:
        return json.loads(lpth_json.read_text())

    except (OSError, ValueError):
        return None

# -------------------------------------------------------------------------------------
def cobre_par(fdct_resumo: typing.Optional[dict], fs_radar: str, fi_alcance: int) -> typing.Optional[bool]:
    """
    o checkpoint se aplica ao par radar/alcance?

    :param fdct_resumo: le_resumo_checkpoint()

    :returns: True/False; None se o escopo é desconhecido (checkpoint da versão 1 ou
              sem .json)
    """
    if not fdct_resumo or "escopo" not in fdct_resumo:
        return None

    return f"{fs_radar}_{fi_alcance}" in fdct_resumo["escopo"].get("pares", [])

# -------------------------------------------------------------------------------------
def _nome_livre(fpth_dir: pathlib.Path, fs_base: str) -> pathlib.Path:
    """
    <base>.pt, ou <base>_N.pt se já existe (dois treinos no mesmo segundo não se
    sobrescrevem).
    """
    lpth = fpth_dir / f"{fs_base}.pt"
    li_sufixo = 0

    while lpth.exists():
        li_sufixo += 1
        lpth = fpth_dir / f"{fs_base}_{li_sufixo}.pt"

    return lpth

# -------------------------------------------------------------------------------------
def salva_checkpoint(fs_modelos_dir: str, fdct_hparams: dict, frede: nn.Module, ftransformacao,
                     fdct_extras: dict, fb_promove: bool=False) -> pathlib.Path:
    """
    grava o checkpoint (candidato). Só com fb_promove aponta atual.pt para ele -- na
    rotina, quem promove é o untPromove, depois do untEval.

    :param fs_modelos_dir: raiz dos modelos
    :param fdct_hparams: hiperparâmetros
    :param frede: rede treinada (os pesos vão para CPU)
    :param ftransformacao: transformação de dados (como_dict())
    :param fdct_extras: epoca, perda_validacao, historico, tempo_treino_s, device,
                        manifesto, escopo, manifesto_sha256, git_commit
    :param fb_promove: aponta atual.pt para o checkpoint (dev/testes)

    :returns: caminho do checkpoint gravado
    """
    ls_backend = fdct_hparams["backend"]
    ls_carimbo = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d%H%M%S")

    lpth_dir = pathlib.Path(fs_modelos_dir) / ls_backend
    lpth_dir.mkdir(parents=True, exist_ok=True)

    lpth_checkpoint = _nome_livre(lpth_dir, f"{ls_backend}_{ls_carimbo}")

    ldct_estado = {lchave: ltensor.detach().cpu() for lchave, ltensor in frede.state_dict().items()}

    ldct_checkpoint = {"versao": CI_VERSAO_CHECKPOINT,
                       "backend": ls_backend,
                       "hparams": fdct_hparams,
                       "transformacao": ftransformacao.como_dict(),
                       "estado": ldct_estado,
                       "torch": torch.__version__,
                       "criado_em": ls_carimbo}
    ldct_checkpoint.update(fdct_extras)

    # grava num temporário e renomeia: atual.pt nunca aponta para um arquivo pela metade
    lpth_temporario = lpth_dir / f".{lpth_checkpoint.name}.tmp"
    torch.save(ldct_checkpoint, lpth_temporario)
    os.replace(lpth_temporario, lpth_checkpoint)

    if fb_promove:
        _aponta_atual(lpth_checkpoint)

    return lpth_checkpoint

# -------------------------------------------------------------------------------------
class SModeloCarregado(typing.NamedTuple):
    """
    um checkpoint pronto para inferência.
    """
    backend: str
    rede: nn.Module
    transformacao: typing.Any
    hparams: dict
    checkpoint: dict

# -------------------------------------------------------------------------------------
def carrega_checkpoint(fpth_checkpoint: typing.Union[str, pathlib.Path],
                       fdevice: typing.Union[str, torch.device]="cpu") -> SModeloCarregado:
    """
    lê um checkpoint e reconstrói rede (em modo eval) e transformação.

    funciona sem GPU: os pesos são lidos para CPU e só então movidos para fdevice.

    :param fpth_checkpoint: caminho do .pt (ou atual.pt)
    :param fdevice: device de inferência

    :returns: SModeloCarregado

    :raises FileNotFoundError: checkpoint ausente
    :raises ValueError: versão de formato desconhecida
    """
    lpth = pathlib.Path(fpth_checkpoint)

    if not lpth.exists():
        raise FileNotFoundError(f"checkpoint ausente: {lpth}")

    # weights_only=False: o checkpoint é nosso e guarda dicionários/listas, não só tensores
    ldct_checkpoint = torch.load(lpth, map_location="cpu", weights_only=False)

    if ldct_checkpoint.get("versao") not in CT_VERSOES_LIDAS:
        raise ValueError(f"{lpth.name}: versão de checkpoint desconhecida "
                         f"({ldct_checkpoint.get('versao')})")

    ldct_hparams = ldct_checkpoint["hparams"]

    lrede = constroi_rede(ldct_hparams)
    lrede.load_state_dict(ldct_checkpoint["estado"])
    lrede.to(fdevice)
    lrede.eval()

    ltransformacao = ud.transformacao_de_dict(ldct_checkpoint["transformacao"])

    return SModeloCarregado(ldct_checkpoint["backend"], lrede, ltransformacao, ldct_hparams,
                            ldct_checkpoint)

# < the end >----------------------------------------------------------------------------
