# -*- coding: utf-8 -*-
"""
Centro Logístico de Simulação Meteorológica
Copyright (c) 1995-2026, Instituto de Controle do Espaço Aéreo

unt_treino

loop de treino genérico (PyTorch puro, sem Lightning) e as perdas de cada backend:

  - rainnet: log-cosh (artigo), só nos pixels cobertos;
  - smaat:   MSE (artigo), só nos pixels cobertos;
    nos dois, peso opcional por faixa de intensidade do alvo (limites_peso_dbz,
    peso_dbz; PesoIntensidade). Sem peso, a regressão em log(R) puxa os núcleos fortes
    para a mediana e o rainnet atenua o eco a cada passo;
  - pytorch: entropia cruzada ponderada por classe (bins de intensidade), pixels
             inválidos ignorados (classe -1).

o loop: Adam, ReduceLROnPlateau (pela perda de validação), corte da norma do gradiente
(clip_grad), semente fixa, AMP só em cuda (bfloat16 se a GPU suporta, senão float16).

lotes com perda não finita são descartados, mas uma época em que eles passam de
max_nao_finitos (fração, no treino ou na validação) interrompe o treino com
TreinoDivergiu: descartar em silêncio esconde a divergência (a perda média dos lotes
restantes até "melhora"). A retomada da última época boa fica gravada.

seleção da melhor época e early stopping (hparam selecao):
  - "metrica": a cada val_cada épocas, previsão COMPLETA (todos os passos, grade
    inteira, rollout) em até max_janelas_val janelas de validação, pontuada por CSI e
    FSS no limiar/escala do portão de promoção (unt_avaliacao.pontuacao) -- o mesmo
    critério do untEval/untPromove. Com media_metrica = N > 1, melhora e early
    stopping seguem a média das últimas N métricas (a época escolhida é a última da
    melhor média). Sem nenhuma métrica definida (validação sem eventos), vale a perda,
    com aviso;
  - "perda": a perda de validação (a de treino, com aviso, sem janelas de validação).

orçamento comum: todo backend faz o mesmo número de itens (passos) por época --
amostras_por_epoca janelas, ou todas. No pytorch (um item por janela x lead) sorteia-se
esse número de itens entre todos os (janela, lead).

retomada: com fpth_retomada, o estado completo (rede, otimizador, escalonadores,
melhores épocas, histórico, sementes) é gravado a cada época; fb_retoma continua dali.
Os recortes aleatórios do dataset não retomam a mesma sequência (o contador é por
processo).

2026.out  mlabru  rede colapsada (saída constante) no início do treino -> TreinoColapsou
2026.out  mlabru  perda de regressão ponderada pela intensidade (PesoIntensidade);
                  seleção pela média móvel da métrica (media_metrica)
2026.out  mlabru  AMP em bfloat16 quando a GPU suporta (ue.dtype_amp; GradScaler só com
                  float16); treino interrompido (TreinoDivergiu) quando a fração de
                  lotes não finitos de uma época passa de max_nao_finitos
2026.out  mlabru  realimenta do rainnet limitado à faixa física (divergia no rollout)
2026.out  mlabru  passos_rollout (perda_rollout); lotes de perda não finita descartados
                  e registrados; métrica de validação só em janelas com evento
                  (min_pixels_evento_val)
2026.out  mlabru  seleção pela métrica do rollout (selecao = metrica), retomada por
                  época, mesmo número de itens por época em todos os backends,
                  clip_grad, passos no histórico
2026.out  mlabru  initial version (Linux/Python)
"""
# < imports >----------------------------------------------------------------------------

# python library
import copy
import functools
import logging
import math
import os
import pathlib
import random
import time
import typing

# numPy
import numpy as np

# pyTorch
import torch
from torch import nn
from torch.utils.data import DataLoader, RandomSampler

# model
import work.unt_avaliacao as ua
import work.unt_dados as ud
import work.unt_device as ue
import work.unt_modelos as um

# < constants >--------------------------------------------------------------------------

# log(2), da forma estável do log-cosh
CF_LOG_2 = math.log(2.0)

# ReduceLROnPlateau: fator de redução da taxa de aprendizado
CF_FATOR_REDUCAO_LR = 0.5

# versão do arquivo de retomada
CI_VERSAO_RETOMADA = 1

# hiperparâmetros que precisam ser iguais para retomar um treino (os demais -- épocas,
# paciência, workers, device -- podem mudar na retomada)
CT_HPARAMS_DA_RETOMADA = ("backend", "modo", "num_entradas", "num_alvos", "largura_base",
                          "recorte_px", "batch", "bins_dbz", "peso_bins", "blocos_por_nivel",
                          "kernels_per_layer", "reduction_ratio", "selecao",
                          "media_metrica", "limites_peso_dbz", "peso_dbz")

# < logging >----------------------------------------------------------------------------

# logger
M_LOG = logging.getLogger(__name__)
M_LOG.setLevel(logging.INFO)

# < tipos >------------------------------------------------------------------------------

class TreinoDivergiu(RuntimeError):
    """
    fração de lotes com perda não finita numa época acima de max_nao_finitos.
    """

class TreinoColapsou(RuntimeError):
    """
    saída da rede constante no início do treino (ReLUs mortas): não aprende mais.
    """

class SResultadoTreino(typing.NamedTuple):
    """
    o que o treino devolve para untTrain gravar o checkpoint.
    """
    rede: nn.Module
    epoca: int
    perda_validacao: float
    metrica_validacao: typing.Optional[float]
    criterio: str
    historico: list
    tempo_treino_s: float
    passos: int

# desvio-padrão da saída abaixo do qual a rede é dada como colapsada (saudável: ~0,5 no
# rainnet em log(R + 0.01); colapsada: exatamente 0)
CF_DESVIO_COLAPSO = 1e-3

# < perdas >-----------------------------------------------------------------------------

class PesoIntensidade:
    """
    peso por pixel conforme a faixa de intensidade do ALVO: limites_peso_dbz levados ao
    espaço do alvo pela transformação do backend (rainnet: log(R + 0.01); smaat: R/100),
    um peso_dbz por faixa (abaixo do 1º limite, entre limites, a partir do último).
    """
    def __init__(self, flst_limites_dbz: list, flst_pesos: list, ftransformacao):
        lnp_limites = ftransformacao.alvo(np.asarray(flst_limites_dbz, dtype=np.float32))

        self.limites = torch.as_tensor(np.asarray(lnp_limites, dtype=np.float32))
        self.pesos = torch.as_tensor(flst_pesos, dtype=torch.float32)

    def __call__(self, falvo: torch.Tensor) -> torch.Tensor:
        # o alvo de um passo do rollout é uma fatia (não contígua) de y
        lidx = torch.bucketize(falvo.float().contiguous(), self.limites.to(falvo.device),
                               right=True)

        return self.pesos.to(falvo.device)[lidx]

# -------------------------------------------------------------------------------------
def peso_intensidade(fdct_hparams: dict, ftransformacao) -> typing.Optional[PesoIntensidade]:
    """
    o PesoIntensidade dos hiperparâmetros, ou None (sem peso: perda dos artigos).
    """
    if not fdct_hparams.get("peso_dbz"):
        return None

    return PesoIntensidade(fdct_hparams["limites_peso_dbz"], fdct_hparams["peso_dbz"],
                           ftransformacao)

# -------------------------------------------------------------------------------------
def _media_mascarada(fperda_por_pixel: torch.Tensor, fmascara: torch.Tensor,
                     fpesos: typing.Optional[torch.Tensor]=None) -> torch.Tensor:
    """
    média só nos pixels válidos (0 se nenhum é válido, sem NaN); com fpesos, média
    ponderada (normalizada pela soma dos pesos válidos).
    """
    lmascara = fmascara if fpesos is None else fmascara * fpesos
    lsoma_mascara = lmascara.sum()

    return (fperda_por_pixel * lmascara).sum() / lsoma_mascara.clamp_min(1.0)

# -------------------------------------------------------------------------------------
def perda_log_cosh(fprevisto: torch.Tensor, falvo: torch.Tensor, fmascara: torch.Tensor,
                   fpeso: typing.Optional[PesoIntensidade]=None) -> torch.Tensor:
    """
    log(cosh(e)) estável: e + softplus(-2e) - log 2 (o cosh estoura em float16/32).
    """
    lerro = fprevisto - falvo
    lperda = lerro + nn.functional.softplus(-2.0 * lerro) - CF_LOG_2

    return _media_mascarada(lperda, fmascara, None if fpeso is None else fpeso(falvo))

# -------------------------------------------------------------------------------------
def perda_mse(fprevisto: torch.Tensor, falvo: torch.Tensor, fmascara: torch.Tensor,
              fpeso: typing.Optional[PesoIntensidade]=None) -> torch.Tensor:
    """
    erro quadrático médio só nos pixels válidos.
    """
    return _media_mascarada((fprevisto - falvo) ** 2, fmascara,
                            None if fpeso is None else fpeso(falvo))

# -------------------------------------------------------------------------------------
class PerdaEntropiaBins(nn.Module):
    """
    entropia cruzada ponderada por classe; pixels com classe -1 (fora da cobertura)
    não entram.
    """
    def __init__(self, flst_pesos: list):
        super().__init__()

        self.register_buffer("pesos", torch.tensor(flst_pesos, dtype=torch.float32))

    def forward(self, flogits: torch.Tensor, fclasses: torch.Tensor) -> torch.Tensor:
        # pixels todos inválidos: perda zero (sem NaN)
        if not (fclasses != ud.CI_CLASSE_IGNORADA).any():
            return flogits.sum() * 0.0

        return nn.functional.cross_entropy(flogits.float(), fclasses, weight=self.pesos,
                                           ignore_index=ud.CI_CLASSE_IGNORADA)

# -------------------------------------------------------------------------------------
def calcula_perda(frede: nn.Module, fdct_lote: dict, fdct_hparams: dict,
                  fperda_bins: typing.Optional[PerdaEntropiaBins],
                  fpeso: typing.Optional[PesoIntensidade]=None) -> torch.Tensor:
    """
    passa um lote pela rede e calcula a perda do backend.

    :param frede: a rede
    :param fdct_lote: lote do DataLoader (já no device)
    :param fdct_hparams: hiperparâmetros (backend, modo)
    :param fperda_bins: perda do backend pytorch (None nos outros)
    :param fpeso: peso por intensidade da perda de regressão (None = sem peso)

    :returns: perda escalar
    """
    ls_backend = fdct_hparams["backend"]

    if "pytorch" == ls_backend:
        llogits = frede(fdct_lote["x"], fdct_lote["lead"])
        return fperda_bins(llogits, fdct_lote["y"])

    lperda_regressao = functools.partial(
        perda_log_cosh if "rainnet" == ls_backend else perda_mse, fpeso=fpeso)

    if um.CS_MODO_RECURSIVO != fdct_hparams["modo"]:
        return lperda_regressao(frede(fdct_lote["x"]), fdct_lote["y"], fdct_lote["m"])

    return perda_rollout(frede, fdct_lote, ls_backend, lperda_regressao)

# -------------------------------------------------------------------------------------
def realimenta(fprevisto: torch.Tensor, fs_backend: str) -> torch.Tensor:
    """
    saída da rede no formato da entrada, como unt_dados.Transformacao*.realimenta na
    inferência: rainnet já é log(R + 0.01), limitado à faixa física (senão diverge no
    rollout); smaat sem chuva negativa.
    """
    if "smaat" == fs_backend:
        return fprevisto.clamp_min(0.0)

    lf_min, lf_max = ud.faixa_realimentacao_rainnet()

    return fprevisto.clamp(lf_min, lf_max)

# -------------------------------------------------------------------------------------
def perda_rollout(frede: nn.Module, fdct_lote: dict, fs_backend: str,
                  fperda_regressao) -> torch.Tensor:
    """
    modo recursivo: t+1 realimentado nos K = y.shape[1] passos do lote (passos_rollout),
    como na previsão; perda média dos K passos. Com K = 1 é o treino t+1 dos artigos.
    """
    lx = fdct_lote["x"]
    li_passos = fdct_lote["y"].shape[1]
    llst_perdas = []

    for li_passo in range(li_passos):
        lprevisto = frede(lx)
        llst_perdas.append(fperda_regressao(lprevisto,
                                            fdct_lote["y"][:, li_passo:li_passo + 1],
                                            fdct_lote["m"][:, li_passo:li_passo + 1]))

        if li_passo + 1 < li_passos:
            lx = torch.cat([lx[:, 1:], realimenta(lprevisto, fs_backend).to(lx.dtype)], dim=1)

    return torch.stack(llst_perdas).mean()

# < carregadores e uma época >-----------------------------------------------------------

def fixa_semente(fi_semente: int) -> None:
    """
    semente de python, numpy e torch (reprodutibilidade).
    """
    random.seed(fi_semente)
    np.random.seed(fi_semente)
    torch.manual_seed(fi_semente)

# -------------------------------------------------------------------------------------
def itens_por_epoca(fdataset, fdct_hparams: dict) -> int:
    """
    itens de treino por época: amostras_por_epoca, ou o número de janelas -- o mesmo em
    todos os backends (no modo lead o dataset tem num_alvos itens por janela).
    """
    li_amostras = fdct_hparams["amostras_por_epoca"]

    if li_amostras > 0:
        return li_amostras

    return len(fdataset.janelas)

# -------------------------------------------------------------------------------------
def monta_carregador(fdataset, fdct_hparams: dict, fdevice: torch.device,
                     fb_treino: bool) -> DataLoader:
    """
    DataLoader do conjunto: treino com itens_por_epoca() sorteados por época,
    validação inteira e na ordem.
    """
    lsampler = None

    if fb_treino and len(fdataset) > 0:
        li_itens = itens_por_epoca(fdataset, fdct_hparams)
        lb_reposicao = li_itens > len(fdataset)
        lsampler = RandomSampler(fdataset, replacement=lb_reposicao, num_samples=li_itens)

    li_workers = fdct_hparams["workers"]

    return DataLoader(fdataset, batch_size=fdct_hparams["batch"], shuffle=False,
                      sampler=lsampler, num_workers=li_workers,
                      pin_memory=ue.usa_pin_memory(fdevice),
                      persistent_workers=li_workers > 0, drop_last=False)

# -------------------------------------------------------------------------------------
def _para_device(fdct_lote: dict, fdevice: torch.device) -> dict:
    """
    move os tensores do lote para o device.
    """
    lb_assincrono = "cuda" == fdevice.type

    return {lchave: ltensor.to(fdevice, non_blocking=lb_assincrono)
            for lchave, ltensor in fdct_lote.items()}

# -------------------------------------------------------------------------------------
def _passo_otimizador(frede: nn.Module, flperda: torch.Tensor, fotimizador,
                      fescalonador_amp, ff_clip_grad: float) -> None:
    """
    backward, corte da norma do gradiente (na escala real, depois do unscale_ do AMP)
    e passo do otimizador.
    """
    fotimizador.zero_grad(set_to_none=True)
    fescalonador_amp.scale(flperda).backward()

    if ff_clip_grad > 0.0:
        fescalonador_amp.unscale_(fotimizador)
        nn.utils.clip_grad_norm_(frede.parameters(), ff_clip_grad)

    fescalonador_amp.step(fotimizador)
    fescalonador_amp.update()

# -------------------------------------------------------------------------------------
def confere_colapso(frede: nn.Module, fdct_lote: dict, fi_passos: int,
                    fdtype_amp: typing.Optional[torch.dtype], fdevice: torch.device) -> None:
    """
    rainnet/smaat: saída constante no lote = rede colapsada. No rainnet, os 2 canais ReLU
    antes da saída morrem nos primeiros passos com algumas inicializações e a rede fica
    presa numa constante (a perda só desce pelo viés da última camada).

    :raises TreinoColapsou: desvio-padrão da saída < CF_DESVIO_COLAPSO
    """
    with torch.no_grad(), torch.autocast(device_type=fdevice.type,
                                         dtype=fdtype_amp or torch.float16,
                                         enabled=fdtype_amp is not None):
        lf_desvio = float(frede(fdct_lote["x"]).float().std())

    if lf_desvio < CF_DESVIO_COLAPSO:
        raise TreinoColapsou(f"saída constante depois de {fi_passos} passos (desvio "
                             f"{lf_desvio:.1e}): rede colapsada")

# -------------------------------------------------------------------------------------
def executa_epoca(frede: nn.Module, fcarregador: DataLoader, fdct_hparams: dict,
                  fdevice: torch.device, fperda_bins, fotimizador=None,
                  fescalonador_amp=None,
                  fdtype_amp: typing.Optional[torch.dtype]=None,
                  fpeso: typing.Optional[PesoIntensidade]=None,
                  fi_passo_colapso: int=0) -> tuple:
    """
    uma passada pelo carregador. Com fotimizador, treina; sem ele, só avalia.
    fdtype_amp: tipo da precisão mista (None = float32); fpeso: peso por intensidade da
    perda de regressão; fi_passo_colapso > 0: confere_colapso depois desse passo.

    um lote com perda não finita (NaN/inf: estouro em float16 ou dado inválido) é
    contado e descartado -- sem passo do otimizador e fora da média, que um só NaN
    tornaria NaN para a época inteira.

    :returns: (perda média dos lotes finitos -- NaN se não houve nenhum --, lotes
              finitos, lotes descartados)
    """
    lb_treino = fotimizador is not None
    frede.train(lb_treino)

    lf_soma = 0.0
    li_lotes = 0
    li_nao_finitos = 0

    with torch.set_grad_enabled(lb_treino):
        for ldct_lote in fcarregador:
            ldct_lote = _para_device(ldct_lote, fdevice)

            with torch.autocast(device_type=fdevice.type, dtype=fdtype_amp or torch.float16,
                                enabled=fdtype_amp is not None):
                lperda = calcula_perda(frede, ldct_lote, fdct_hparams, fperda_bins, fpeso)

            if not torch.isfinite(lperda.detach()):
                li_nao_finitos += 1
                continue

            if lb_treino:
                _passo_otimizador(frede, lperda, fotimizador, fescalonador_amp,
                                  fdct_hparams["clip_grad"])

            lf_soma += float(lperda.detach())
            li_lotes += 1

            if lb_treino and li_lotes == fi_passo_colapso:
                confere_colapso(frede, ldct_lote, li_lotes, fdtype_amp, fdevice)

    if 0 == li_lotes:
        return float("nan"), 0, li_nao_finitos

    return lf_soma / li_lotes, li_lotes, li_nao_finitos

# -------------------------------------------------------------------------------------
def janelas_da_metrica(fds_validacao, fdct_hparams: dict) -> list:
    """
    as janelas de validação da métrica (até max_janelas_val, igualmente espaçadas), só
    as com evento se min_pixels_evento_val > 0. Escolhidas uma vez e guardadas no
    dataset: a mesma amostra em todas as épocas.
    """
    if getattr(fds_validacao, "janelas_metrica", None) is not None:
        return fds_validacao.janelas_metrica

    li_maximo = fdct_hparams["max_janelas_val"]
    li_min_pixels = fdct_hparams.get("min_pixels_evento_val", 0)
    llst_janelas = fds_validacao.janelas

    if li_min_pixels > 0:
        # sorteia de um conjunto 4x maior, igualmente espaçado, para não ler tudo
        llst_candidatas = ua.amostra_janelas(llst_janelas, 4 * li_maximo if li_maximo else 0)
        llst_janelas = ua.janelas_com_evento(llst_candidatas, fds_validacao.cappi_dir,
                                             float(fdct_hparams["limiar_selecao"]),
                                             li_min_pixels, fdct_hparams["num_entradas"])
        M_LOG.info(f"[+] métrica de validação: {len(llst_janelas)} de "
                   f"{len(llst_candidatas)} janelas candidatas com evento.")

    fds_validacao.janelas_metrica = ua.amostra_janelas(llst_janelas, li_maximo)

    return fds_validacao.janelas_metrica

# -------------------------------------------------------------------------------------
def avalia_metrica(frede: nn.Module, fds_validacao, fdct_hparams: dict) -> typing.Optional[float]:
    """
    pontuação (unt_avaliacao.pontuacao) da previsão completa da rede nas janelas de
    validação (até max_janelas_val).

    :returns: pontuação (maior = melhor), ou None sem eventos
    """
    llst_janelas = janelas_da_metrica(fds_validacao, fdct_hparams)
    lmodelo = um.SModeloCarregado(fdct_hparams["backend"], frede, fds_validacao.transformacao,
                                  fdct_hparams, {})

    frede.eval()
    ldct_por_passo = ua.avalia_modelo(lmodelo, llst_janelas, fds_validacao.cappi_dir)

    return ua.pontuacao(ldct_por_passo, fdct_hparams["limiar_selecao"],
                        fdct_hparams["escala_selecao"])

# < estado do loop >---------------------------------------------------------------------

class EstadoLoop:
    """
    o que muda de época em época (e o que a retomada grava): histórico, contadores e as
    duas melhores épocas -- pela perda e pela métrica.
    """
    def __init__(self, frede: nn.Module):
        self.epoca = 0
        self.passos = 0
        self.tempo_s = 0.0
        self.sem_melhora = 0
        self.encerrado = False
        self.historico = []

        self.melhor_perda = float("inf")
        self.epoca_melhor_perda = 0
        self.estado_melhor_perda = copy.deepcopy(frede.state_dict())

        self.melhor_metrica = None
        self.epoca_melhor_metrica = 0
        self.estado_melhor_metrica = None

    # ---------------------------------------------------------------------------------
    def registra(self, frede: nn.Module, fs_selecao: str, ff_perda: float,
                 ff_metrica: typing.Optional[float]) -> bool:
        """
        atualiza as melhores épocas com a época corrente (self.epoca).

        :returns: True se o critério ativo melhorou (zera a paciência)
        """
        lb_melhorou_perda = math.isfinite(ff_perda) and ff_perda < self.melhor_perda

        if lb_melhorou_perda:
            self.melhor_perda = ff_perda
            self.epoca_melhor_perda = self.epoca
            self.estado_melhor_perda = copy.deepcopy(frede.state_dict())

        lb_melhorou_metrica = ff_metrica is not None and (
            self.melhor_metrica is None or ff_metrica > self.melhor_metrica)

        if lb_melhorou_metrica:
            self.melhor_metrica = ff_metrica
            self.epoca_melhor_metrica = self.epoca
            self.estado_melhor_metrica = copy.deepcopy(frede.state_dict())

        # pela métrica assim que alguma foi definida; antes disso, pela perda
        if um.CS_SELECAO_METRICA == fs_selecao and self.melhor_metrica is not None:
            return lb_melhorou_metrica

        return lb_melhorou_perda

    # ---------------------------------------------------------------------------------
    def escolhida(self, fs_selecao: str) -> tuple:
        """
        a melhor época pelo critério (métrica, se alguma foi definida).

        :returns: (época, estado da rede, critério usado)
        """
        if um.CS_SELECAO_METRICA == fs_selecao and self.estado_melhor_metrica is not None:
            return self.epoca_melhor_metrica, self.estado_melhor_metrica, um.CS_SELECAO_METRICA

        if um.CS_SELECAO_METRICA == fs_selecao:
            M_LOG.warning("[!] nenhuma época com métrica definida (validação sem eventos?): "
                          "melhor época escolhida pela perda.")

        return self.epoca_melhor_perda, self.estado_melhor_perda, um.CS_SELECAO_PERDA

    # ---------------------------------------------------------------------------------
    def como_dict(self) -> dict:
        return dict(vars(self))

    # ---------------------------------------------------------------------------------
    def de_dict(self, fdct: dict) -> None:
        for ls_chave, lx_valor in fdct.items():
            setattr(self, ls_chave, lx_valor)

# -------------------------------------------------------------------------------------
def metrica_media(flst_historico: list, ff_metrica: typing.Optional[float],
                  fi_janela: int) -> typing.Optional[float]:
    """
    média da métrica da época com as fi_janela - 1 métricas definidas anteriores do
    histórico (menos, no início do treino). None se a época não tem métrica.
    """
    if ff_metrica is None:
        return None

    llst = [ld["metrica_validacao"] for ld in flst_historico
            if ld.get("metrica_validacao") is not None]
    llst = llst[max(0, len(llst) - (fi_janela - 1)):] if fi_janela > 1 else []
    llst.append(ff_metrica)

    return sum(llst) / len(llst)

# < retomada >---------------------------------------------------------------------------

def identidade_retomada(fdct_hparams: dict, fs_manifesto_sha256: typing.Optional[str]) -> dict:
    """
    o que precisa ser igual para continuar um treino: hiperparâmetros de arquitetura e
    de dados, e o manifesto.
    """
    ldct = {ls_chave: fdct_hparams.get(ls_chave) for ls_chave in CT_HPARAMS_DA_RETOMADA}
    ldct["manifesto_sha256"] = fs_manifesto_sha256

    return ldct

# -------------------------------------------------------------------------------------
def _estados_aleatorios() -> dict:
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch": torch.get_rng_state()}

# -------------------------------------------------------------------------------------
def _restaura_aleatorios(fdct: dict) -> None:
    random.setstate(fdct["python"])
    np.random.set_state(fdct["numpy"])
    torch.set_rng_state(fdct["torch"])

# -------------------------------------------------------------------------------------
def grava_retomada(fpth_retomada: pathlib.Path, frede: nn.Module, fdct_otimizacao: dict,
                   festado: EstadoLoop, fdct_identidade: dict) -> None:
    """
    estado completo do treino ao fim de uma época (temporário + rename).

    :param fdct_otimizacao: {"otimizador", "escalonador_lr", "escalonador_amp"}
    """
    ldct = {"versao": CI_VERSAO_RETOMADA,
            "identidade": fdct_identidade,
            "estado": frede.state_dict(),
            "otimizacao": {ls: lobj.state_dict() for ls, lobj in fdct_otimizacao.items()},
            "loop": festado.como_dict(),
            "aleatorios": _estados_aleatorios()}

    fpth_retomada.parent.mkdir(parents=True, exist_ok=True)
    lpth_temporario = fpth_retomada.parent / f".{fpth_retomada.name}.tmp"
    torch.save(ldct, lpth_temporario)
    os.replace(lpth_temporario, fpth_retomada)

# -------------------------------------------------------------------------------------
def carrega_retomada(fpth_retomada: pathlib.Path, frede: nn.Module, fdct_otimizacao: dict,
                     festado: EstadoLoop, fdct_identidade: dict) -> None:
    """
    restaura o treino gravado por grava_retomada (rede já no device).

    :raises ValueError: versão desconhecida ou identidade diferente (outro manifesto,
                        outra arquitetura...)
    """
    ldct = torch.load(fpth_retomada, map_location="cpu", weights_only=False)

    if CI_VERSAO_RETOMADA != ldct.get("versao"):
        raise ValueError(f"{fpth_retomada.name}: versão de retomada desconhecida "
                         f"({ldct.get('versao')})")

    llst_diferentes = sorted(ls for ls in fdct_identidade
                             if ldct["identidade"].get(ls) != fdct_identidade[ls])
    if llst_diferentes:
        raise ValueError(f"{fpth_retomada.name}: não dá para retomar, mudou "
                         f"{', '.join(llst_diferentes)}")

    frede.load_state_dict(ldct["estado"])

    for ls_nome, lobj in fdct_otimizacao.items():
        lobj.load_state_dict(ldct["otimizacao"][ls_nome])

    festado.de_dict(ldct["loop"])
    _restaura_aleatorios(ldct["aleatorios"])

    M_LOG.info(f"[+] retomando de {fpth_retomada.name}: época {festado.epoca} concluída, "
               f"{festado.passos} passos.")

# < loop >-------------------------------------------------------------------------------

def _uma_epoca(frede: nn.Module, fdct_carregadores: dict, fds_validacao, fdct_hparams: dict,
               fdevice: torch.device, fdct_perdas: dict, fdct_otimizacao: dict,
               fdtype_amp: typing.Optional[torch.dtype], festado: EstadoLoop) -> dict:
    """
    treino + validação (perda e, quando é a vez, métrica) de uma época.
    fdct_perdas: {"bins": PerdaEntropiaBins ou None, "peso": PesoIntensidade ou None}.

    :returns: registro da época (histórico)

    :raises TreinoDivergiu: lotes não finitos acima de max_nao_finitos
    :raises TreinoColapsou: saída constante no início da 1ª época (rainnet/smaat)
    """
    lf_inicio = time.perf_counter()

    # o colapso acontece nos primeiros passos; o pytorch (logits por bin) não é verificado
    li_passo_colapso = 0
    if 1 == festado.epoca and "pytorch" != fdct_hparams["backend"]:
        li_passo_colapso = fdct_hparams.get("passos_colapso", 0)

    lf_perda_treino, li_lotes, li_descartados = executa_epoca(
        frede, fdct_carregadores["treino"], fdct_hparams, fdevice, fdct_perdas["bins"],
        fdct_otimizacao["otimizador"], fdct_otimizacao["escalonador_amp"], fdtype_amp,
        fdct_perdas["peso"], li_passo_colapso)
    festado.passos += li_lotes
    li_descartados_val = 0

    _confere_nao_finitos(festado.epoca, "treino", li_lotes, li_descartados,
                         fdct_hparams["max_nao_finitos"])

    lb_tem_validacao = len(fds_validacao) > 0
    lf_perda_val = lf_perda_treino
    lf_metrica = None

    if lb_tem_validacao:
        lf_perda_val, li_lotes_val, li_descartados_val = executa_epoca(
            frede, fdct_carregadores["validacao"], fdct_hparams, fdevice,
            fdct_perdas["bins"], fdtype_amp=fdtype_amp, fpeso=fdct_perdas["peso"])
        _confere_nao_finitos(festado.epoca, "validação", li_lotes_val, li_descartados_val,
                             fdct_hparams["max_nao_finitos"])

        lb_vez_da_metrica = 0 == festado.epoca % fdct_hparams["val_cada"]
        if um.CS_SELECAO_METRICA == fdct_hparams["selecao"] and lb_vez_da_metrica:
            lf_metrica = avalia_metrica(frede, fds_validacao, fdct_hparams)

    if li_descartados or li_descartados_val:
        M_LOG.warning(f"[!] época {festado.epoca}: perda não finita em {li_descartados} "
                      f"lote(s) de treino e {li_descartados_val} de validação "
                      f"(descartados).")

    if math.isfinite(lf_perda_val):
        fdct_otimizacao["escalonador_lr"].step(lf_perda_val)

    return {"epoca": festado.epoca,
            "perda_treino": lf_perda_treino,
            "perda_validacao": lf_perda_val if lb_tem_validacao else None,
            "metrica_validacao": lf_metrica,
            "lr": fdct_otimizacao["otimizador"].param_groups[0]["lr"],
            "passos": festado.passos,
            "lotes_nao_finitos": {"treino": li_descartados, "validacao": li_descartados_val},
            "duracao_s": time.perf_counter() - lf_inicio}

# -------------------------------------------------------------------------------------
def _confere_nao_finitos(fi_epoca: int, fs_conjunto: str, fi_finitos: int,
                         fi_descartados: int, ff_max_fracao: float) -> None:
    """
    interrompe o treino se a fração de lotes não finitos passou de ff_max_fracao.

    :raises TreinoDivergiu: fração acima do limite
    """
    li_total = fi_finitos + fi_descartados
    if 0 == li_total or fi_descartados / li_total <= ff_max_fracao:
        return

    raise TreinoDivergiu(f"época {fi_epoca}: perda não finita em {fi_descartados} de "
                         f"{li_total} lotes de {fs_conjunto} ({fi_descartados / li_total:.1%}"
                         f", limite max_nao_finitos = {ff_max_fracao:.1%}); treino "
                         f"interrompido -- a retomada da época {fi_epoca - 1} foi mantida.")

# -------------------------------------------------------------------------------------
def _monta_otimizacao(frede: nn.Module, fdct_hparams: dict,
                      fdtype_amp: typing.Optional[torch.dtype],
                      fdevice: torch.device) -> dict:
    """
    otimizador e escalonadores (o que a retomada grava junto com a rede). O GradScaler
    só é ativo com float16 (o bfloat16 tem a faixa do float32).
    """
    lotimizador = torch.optim.Adam(frede.parameters(), lr=fdct_hparams["lr"])

    return {"otimizador": lotimizador,
            "escalonador_lr": torch.optim.lr_scheduler.ReduceLROnPlateau(
                lotimizador, factor=CF_FATOR_REDUCAO_LR,
                patience=max(1, fdct_hparams["paciencia"] // 2)),
            "escalonador_amp": torch.amp.GradScaler(fdevice.type,
                                                    enabled=torch.float16 == fdtype_amp)}

# -------------------------------------------------------------------------------------
def _resultado(frede: nn.Module, festado: EstadoLoop, fs_selecao: str) -> SResultadoTreino:
    """
    carrega na rede os pesos da época escolhida e monta o resultado.
    """
    li_epoca, ldct_estado, ls_criterio = festado.escolhida(fs_selecao)
    frede.load_state_dict(ldct_estado)

    ldct_registro = next((ld for ld in festado.historico if ld["epoca"] == li_epoca), {})
    lf_perda = ldct_registro.get("perda_validacao")

    if lf_perda is None:
        lf_perda = ldct_registro.get("perda_treino", float("nan"))

    return SResultadoTreino(frede, li_epoca, lf_perda, ldct_registro.get("metrica_validacao"),
                            ls_criterio, festado.historico, festado.tempo_s, festado.passos)

# -------------------------------------------------------------------------------------
def treina(frede: nn.Module, fds_treino, fds_validacao, fdct_hparams: dict,
           fdevice: torch.device,
           ff_ao_fim_da_epoca: typing.Optional[typing.Callable]=None,
           fpth_retomada: typing.Optional[pathlib.Path]=None, fb_retoma: bool=False,
           fdct_identidade: typing.Optional[dict]=None) -> SResultadoTreino:
    """
    treina até epocas, ou até paciencia épocas sem melhora no critério de seleção.

    :param frede: rede (pesos iniciais)
    :param fds_treino: JanelasDataset de treino
    :param fds_validacao: JanelasDataset de validação (pode ser vazio)
    :param fdct_hparams: hiperparâmetros
    :param fdevice: device
    :param ff_ao_fim_da_epoca: chamada com o registro de cada época (log/progresso)
    :param fpth_retomada: arquivo de retomada, gravado a cada época (None = não grava)
    :param fb_retoma: continua do fpth_retomada, se ele existe
    :param fdct_identidade: identidade_retomada() (conferida na retomada)

    :returns: SResultadoTreino com a rede nos pesos da época ESCOLHIDA

    :raises ValueError: conjunto de treino vazio; retomada incompatível
    :raises TreinoDivergiu: lotes não finitos acima de max_nao_finitos numa época
    :raises TreinoColapsou: saída constante no início da 1ª época (rainnet/smaat)
    """
    if 0 == len(fds_treino):
        raise ValueError("conjunto de treino vazio (veja o split do manifesto)")

    fixa_semente(fdct_hparams["semente"])

    ldtype_amp = None
    if ue.usa_amp(fdevice, fdct_hparams["amp"]):
        ldtype_amp = ue.dtype_amp(fdevice)
        M_LOG.info(f"[+] precisão mista em {str(ldtype_amp).removeprefix('torch.')}.")

    frede.to(fdevice)

    ldct_perdas = {"bins": None,
                   "peso": peso_intensidade(fdct_hparams, fds_treino.transformacao)}
    if "pytorch" == fdct_hparams["backend"]:
        ldct_perdas["bins"] = PerdaEntropiaBins(fdct_hparams["peso_bins"]).to(fdevice)

    ldct_carregadores = {"treino": monta_carregador(fds_treino, fdct_hparams, fdevice, True),
                         "validacao": monta_carregador(fds_validacao, fdct_hparams, fdevice,
                                                       False)}

    if 0 == len(fds_validacao):
        M_LOG.warning("[!] sem janelas de validação: a melhor época será escolhida pela "
                      "perda de TREINO.")

    ldct_otimizacao = _monta_otimizacao(frede, fdct_hparams, ldtype_amp, fdevice)
    lestado = EstadoLoop(frede)
    ldct_identidade = fdct_identidade or identidade_retomada(fdct_hparams, None)

    if fb_retoma and fpth_retomada is not None and fpth_retomada.exists():
        carrega_retomada(fpth_retomada, frede, ldct_otimizacao, lestado, ldct_identidade)

    ls_selecao = fdct_hparams["selecao"]

    while not lestado.encerrado and lestado.epoca < fdct_hparams["epocas"]:
        lestado.epoca += 1

        ldct_registro = _uma_epoca(frede, ldct_carregadores, fds_validacao, fdct_hparams,
                                   fdevice, ldct_perdas, ldct_otimizacao, ldtype_amp, lestado)
        ldct_registro["metrica_media"] = metrica_media(
            lestado.historico, ldct_registro["metrica_validacao"],
            fdct_hparams.get("media_metrica", 1))
        lestado.historico.append(ldct_registro)
        lestado.tempo_s += ldct_registro["duracao_s"]

        if ff_ao_fim_da_epoca is not None:
            ff_ao_fim_da_epoca(ldct_registro)

        lf_perda = ldct_registro["perda_validacao"]
        if lf_perda is None:
            lf_perda = ldct_registro["perda_treino"]

        if lestado.registra(frede, ls_selecao, lf_perda, ldct_registro["metrica_media"]):
            lestado.sem_melhora = 0
        else:
            lestado.sem_melhora += 1

        if lestado.sem_melhora >= fdct_hparams["paciencia"]:
            M_LOG.info(f"[+] early stopping na época {lestado.epoca}.")
            lestado.encerrado = True

        if fpth_retomada is not None:
            grava_retomada(fpth_retomada, frede, ldct_otimizacao, lestado, ldct_identidade)

    return _resultado(frede, lestado, ls_selecao)

# < the end >----------------------------------------------------------------------------
