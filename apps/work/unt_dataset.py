# -*- coding: utf-8 -*-
"""
Centro Logístico de Simulação Meteorológica
Copyright (c) 1995-2026, Instituto de Controle do Espaço Aéreo

unt_dataset

torch Dataset sobre um manifesto de janelas (unt_janelas). Cada item carrega os .npy da
janela (entradas + alvos, NaN fora da cobertura), recorta, aumenta (opcional) e aplica a
transformação do backend (unt_dados). O que vai para a rede depende do modo:

  - recursivo (rainnet, smaat): x = 4 entradas, y = os passos_rollout primeiros alvos
                                (1 = só t+1), m = máscaras;
  - direto (smaat opcional):    x = 4 entradas, y = os 6 alvos, m = máscaras;
  - lead (pytorch):             um item por (janela, lead): x = 4 entradas + máscara,
                                lead = índice do passo, y = classes do alvo daquele passo
                                (pixels inválidos = classe ignorada).

recorte (recorte_px > 0): posição aleatória (treino) ou central (validação); no treino,
até tentativas_recorte sorteios procurando um recorte com ao menos min_eco_recorte de
pixels com eco (>= 10 dBZ) na última entrada -- sem isso, quase todo recorte seria céu
limpo. recorte_px = 0 usa a grade inteira com padding até múltiplo de 32.

aumento (treino, augment=true): espelhamentos e rotações de 90°, iguais em x, y e m.

calcula_pesos_bins(): peso_bins = auto do backend pytorch -- sqrt(f_max / f_classe) nos
pixels válidos dos alvos de treino (cada quadro lido uma vez), com teto.

2026.out  mlabru  recursivo com passos_rollout alvos
2026.out  mlabru  calcula_pesos_bins (peso_bins = auto)
2026.out  mlabru  initial version (Linux/Python)
"""
# < imports >----------------------------------------------------------------------------

# python library
import logging
import typing

# numPy
import numpy as np

# pyTorch
import torch
from torch.utils.data import Dataset

# model
import work.unt_catalog as oc
import work.unt_dados as ud
import work.unt_modelos as um

# < constants >--------------------------------------------------------------------------

# a partir deste valor (dBZ) um pixel conta como eco na escolha do recorte
CF_DBZ_ECO_RECORTE = 10.0

# teto do peso de uma classe em peso_bins = auto (classe rara ou ausente no treino)
CF_PESO_BIN_MAXIMO = 50.0

# < logging >----------------------------------------------------------------------------

# logger
M_LOG = logging.getLogger(__name__)
M_LOG.setLevel(logging.INFO)

# < funções auxiliares >-----------------------------------------------------------------

def carrega_quadros(fs_cappi_dir: str, fjanela) -> np.ndarray:
    """
    carrega entradas e alvos de uma janela, na ordem cronológica.

    :param fs_cappi_dir: raiz dos CAPPIs
    :param fjanela: unt_janelas.SJanela

    :returns: (entradas + alvos, H, W) em dBZ, NaN fora da cobertura
    """
    llst_caminhos = []

    for li_timestamp in fjanela.entradas + fjanela.alvos:
        llst_caminhos.append(oc.monta_caminho_cappi(fs_cappi_dir, fjanela.radar,
                                                    fjanela.alcance, li_timestamp,
                                                    fjanela.preset))

    return oc.carrega_pilha_cappi(llst_caminhos)

# -------------------------------------------------------------------------------------
def _fracao_eco(fnp_dbz: np.ndarray) -> float:
    """
    fração de pixels com eco (>= CF_DBZ_ECO_RECORTE) num campo 2D.
    """
    with np.errstate(invalid="ignore"):
        return float(np.mean(fnp_dbz >= CF_DBZ_ECO_RECORTE))

# -------------------------------------------------------------------------------------
def escolhe_recorte(fnp_quadros: np.ndarray, fi_lado: int, fi_num_entradas: int,
                    frng: typing.Optional[np.random.Generator], ff_min_eco: float,
                    fi_tentativas: int) -> tuple:
    """
    canto superior esquerdo do recorte.

    :param fnp_quadros: (quadros, H, W) em dBZ
    :param fi_lado: lado do recorte (px)
    :param fi_num_entradas: quantos quadros são entradas (o eco é medido na última)
    :param frng: gerador aleatório (None = recorte central, determinístico)
    :param ff_min_eco: fração mínima de eco aceita
    :param fi_tentativas: sorteios antes de aceitar qualquer posição

    :returns: (linha, coluna)
    """
    li_altura, li_largura = fnp_quadros.shape[-2:]
    li_max_linha = li_altura - fi_lado
    li_max_coluna = li_largura - fi_lado

    if frng is None:
        return li_max_linha // 2, li_max_coluna // 2

    lnp_ultima = fnp_quadros[fi_num_entradas - 1]
    lt_posicao = (0, 0)

    for _ in range(max(1, fi_tentativas)):
        lt_posicao = (int(frng.integers(0, li_max_linha + 1)),
                      int(frng.integers(0, li_max_coluna + 1)))

        li_lin, li_col = lt_posicao
        lnp_trecho = lnp_ultima[li_lin:li_lin + fi_lado, li_col:li_col + fi_lado]

        if _fracao_eco(lnp_trecho) >= ff_min_eco:
            break

    return lt_posicao

# -------------------------------------------------------------------------------------
def aumenta(flst_arrays: list, frng: np.random.Generator) -> list:
    """
    o mesmo espelhamento/rotação de 90° em todos os arrays (eixos -2, -1).
    """
    li_rotacoes = int(frng.integers(0, 4))
    lb_espelha = bool(frng.integers(0, 2))

    llst_saida = []

    for lnp in flst_arrays:
        lnp = np.rot90(lnp, k=li_rotacoes, axes=(-2, -1))

        if lb_espelha:
            lnp = np.flip(lnp, axis=-1)

        llst_saida.append(np.ascontiguousarray(lnp))

    return llst_saida

# -------------------------------------------------------------------------------------
def calcula_pesos_bins(flst_janelas: list, fs_cappi_dir: str, flst_limites_dbz: list) -> list:
    """
    pesos da entropia cruzada a partir da frequência de cada classe (bin de
    intensidade) nos pixels válidos dos alvos: sqrt(f_max / f_classe), a classe mais
    frequente com peso 1, teto CF_PESO_BIN_MAXIMO (também para classe ausente).

    :param flst_janelas: janelas de TREINO
    :param fs_cappi_dir: raiz dos CAPPIs
    :param flst_limites_dbz: bins_dbz

    :returns: len(flst_limites_dbz) + 1 pesos
    """
    li_classes = len(flst_limites_dbz) + 1
    lnp_contagem = np.zeros(li_classes, dtype=np.int64)

    # alvos compartilhados entre janelas vizinhas: cada quadro uma vez
    lset_quadros = {(ljanela.radar, ljanela.alcance, ljanela.preset, li_timestamp)
                    for ljanela in flst_janelas for li_timestamp in ljanela.alvos}

    for ls_radar, li_alcance, ls_preset, li_timestamp in sorted(lset_quadros):
        lnp_dbz = oc.carrega_cappi(oc.monta_caminho_cappi(fs_cappi_dir, ls_radar, li_alcance,
                                                          li_timestamp, ls_preset))
        lnp_validos = lnp_dbz[np.isfinite(lnp_dbz)]
        lnp_contagem += np.bincount(np.digitize(lnp_validos, flst_limites_dbz),
                                    minlength=li_classes)

    if 0 == lnp_contagem.sum():
        raise ValueError("peso_bins = auto: nenhum pixel válido nos alvos de treino")

    lnp_frequencia = lnp_contagem / lnp_contagem.sum()

    with np.errstate(divide="ignore"):
        lnp_pesos = np.sqrt(lnp_frequencia.max() / lnp_frequencia)

    lnp_pesos = np.minimum(lnp_pesos, CF_PESO_BIN_MAXIMO)

    M_LOG.info(f"[+] peso_bins = auto ({len(lset_quadros)} quadros): "
               f"{', '.join(f'{lf:.2f}' for lf in lnp_pesos)}")

    return [round(float(lf), 4) for lf in lnp_pesos]

# < dataset >----------------------------------------------------------------------------

class JanelasDataset(Dataset):
    """
    janelas de um conjunto (treino/validação/teste) para um backend.
    """
    def __init__(self, flst_janelas: list, fs_cappi_dir: str, fdct_hparams: dict,
                 ftransformacao, fb_treino: bool):
        """
        :param flst_janelas: SJanela de um único conjunto
        :param fs_cappi_dir: raiz dos CAPPIs
        :param fdct_hparams: hiperparâmetros (unt_modelos.hiperparametros)
        :param ftransformacao: transformação do backend (unt_dados)
        :param fb_treino: True = recorte aleatório e aumento; False = determinístico
        """
        self.janelas = list(flst_janelas)
        self.cappi_dir = fs_cappi_dir
        self.hparams = fdct_hparams
        self.transformacao = ftransformacao
        self.treino = fb_treino
        self.modo = fdct_hparams["modo"]
        self.num_entradas = fdct_hparams["num_entradas"]
        self.num_alvos = fdct_hparams["num_alvos"]
        self.semente = fdct_hparams.get("semente", 0)
        self._contador = 0

    def __len__(self) -> int:
        if um.CS_MODO_LEAD == self.modo:
            return len(self.janelas) * self.num_alvos

        return len(self.janelas)

    def _gerador(self, fi_indice: int) -> typing.Optional[np.random.Generator]:
        """
        treino: gerador novo a cada chamada (recortes diferentes por época, mas a
        sequência é reprodutível pela semente); validação: None (determinístico).
        """
        if not self.treino:
            return None

        self._contador += 1

        return np.random.default_rng([self.semente, fi_indice, self._contador])

    def _recorta(self, fnp_quadros: np.ndarray, frng) -> np.ndarray:
        """
        recorte (recorte_px > 0) ou padding da grade inteira até múltiplo de 32.
        """
        li_lado = self.hparams["recorte_px"]

        if not li_lado:
            return ud.aplica_padding(fnp_quadros, ud.calcula_padding(fnp_quadros.shape))

        li_altura, li_largura = fnp_quadros.shape[-2:]

        # grade menor que o recorte (ex.: testes): usa a grade com padding
        if li_lado > min(li_altura, li_largura):
            return ud.aplica_padding(fnp_quadros, ud.calcula_padding(fnp_quadros.shape))

        li_lin, li_col = escolhe_recorte(fnp_quadros, li_lado, self.num_entradas, frng,
                                         self.hparams["min_eco_recorte"],
                                         self.hparams["tentativas_recorte"])

        return fnp_quadros[:, li_lin:li_lin + li_lado, li_col:li_col + li_lado]

    def __getitem__(self, fi_indice: int) -> dict:
        if um.CS_MODO_LEAD == self.modo:
            li_janela, li_lead = divmod(fi_indice, self.num_alvos)
        else:
            li_janela, li_lead = fi_indice, 0

        lrng = self._gerador(fi_indice)

        lnp_quadros = carrega_quadros(self.cappi_dir, self.janelas[li_janela])
        lnp_quadros = self._recorta(lnp_quadros, lrng)

        if self.treino and self.hparams["augment"] and lrng is not None:
            lnp_quadros = aumenta([lnp_quadros], lrng)[0]

        lnp_entradas = lnp_quadros[:self.num_entradas]
        lnp_alvos = lnp_quadros[self.num_entradas:]

        ldct_item = {"x": torch.from_numpy(self.transformacao.entrada(lnp_entradas))}

        if um.CS_MODO_LEAD == self.modo:
            ldct_item["lead"] = torch.tensor(li_lead, dtype=torch.int64)
            ldct_item["y"] = torch.from_numpy(self.transformacao.alvo(lnp_alvos[li_lead]))
            return ldct_item

        if um.CS_MODO_DIRETO == self.modo:
            lnp_alvo = lnp_alvos
        else:
            lnp_alvo = lnp_alvos[:self.hparams.get("passos_rollout", 1)]

        ldct_item["y"] = torch.from_numpy(self.transformacao.alvo(lnp_alvo))
        ldct_item["m"] = torch.from_numpy(ud.mascara_valida(lnp_alvo).astype(np.float32))

        return ldct_item

# < the end >----------------------------------------------------------------------------
