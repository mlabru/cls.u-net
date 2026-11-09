# -*- coding: utf-8 -*-
"""
Centro Logístico de Simulação Meteorológica
Copyright (c) 1995-2026, Instituto de Controle do Espaço Aéreo

unt_dados

conversões entre os CAPPIs (dBZ) e o que cada rede recebe/devolve. Cada backend mantém
o pré-processamento do seu artigo; a comparação entre eles é sempre feita em dBZ, na
saída (unt_verify).

  - rainnet: dBZ -> mm/h (Z-R 200/1.6) -> log(R + 0.01)   (Ayzel et al., 2020)
  - smaat:   dBZ -> mm/h -> R / escala_mm_h               (min-max do artigo; a escala
                                                           fica gravada no checkpoint)
  - pytorch: dBZ em [0, 70] -> [0, 1], mais um canal com a máscara de cobertura; o alvo
             é a classe (bin de intensidade) de cada pixel, para a entropia cruzada.

máscara comum: NaN (o -9999 "fora da cobertura", já convertido por
unt_catalog.carrega_cappi) é pixel INVÁLIDO -- vira 0 na entrada da rede e fica fora da
perda e das métricas. O -32 ("coberto, sem eco") é chuva zero.

também aqui: padding refletido até múltiplo de 32 (500 -> 512, 600 -> 608) e o recorte
de volta, iguais nos três backends.

2026.out  mlabru  rainnet realimenta limitado à faixa física (faixa_realimentacao_rainnet)
2026.out  mlabru  initial version (Linux/Python)
"""
# < imports >----------------------------------------------------------------------------

# python library
import typing

# numPy
import numpy as np

# backends (Z-R)
import work.unt_backend_comum as obc

# < constants >--------------------------------------------------------------------------

# piso da conversão mm/h -> dBZ (R = 0 vira "sem eco", não -inf)
CF_DBZ_SEM_ECO = -32.0

# faixa de dBZ da entrada do backend pytorch
CF_DBZ_MIN_ESCALA = 0.0
CF_DBZ_MAX_ESCALA = 70.0

# deslocamento do log do rainnet (artigo: log(x + 0.01))
CF_RAINNET_EPSILON = 0.01

# escala padrão do smaat (mm/h); o artigo usa o máximo do conjunto de treino
CF_SMAAT_ESCALA_MM_H = 100.0

# classe ignorada na entropia cruzada (pixel inválido)
CI_CLASSE_IGNORADA = -1

# múltiplo do padding (5 níveis de pooling no pior caso)
CI_MULTIPLO_PADDING = 32

# < máscara e Z-R >----------------------------------------------------------------------

def mascara_valida(fnp_dbz: np.ndarray) -> np.ndarray:
    """
    pixels com dado (não-NaN).

    :param fnp_dbz: campo(s) em dBZ, NaN fora da cobertura

    :returns: array booleano de mesmo shape
    """
    return np.isfinite(fnp_dbz)

# -------------------------------------------------------------------------------------
def dbz_para_mm_h(fnp_dbz: np.ndarray) -> np.ndarray:
    """
    dBZ -> mm/h (Z-R de unt_backend_comum); inválido vira 0.
    """
    lnp_dbz = np.nan_to_num(fnp_dbz, nan=CF_DBZ_SEM_ECO, neginf=CF_DBZ_SEM_ECO)

    return obc.converte_dbz_para_mm_h(lnp_dbz).astype(np.float32)

# -------------------------------------------------------------------------------------
def mm_h_para_dbz(fnp_mm_h: np.ndarray) -> np.ndarray:
    """
    mm/h -> dBZ com piso em CF_DBZ_SEM_ECO (R <= 0 vira "sem eco", nunca -inf).
    """
    lnp_mm_h = np.maximum(np.nan_to_num(fnp_mm_h, nan=0.0), 0.0)
    lnp_dbz = obc.converte_mm_h_para_dbz(lnp_mm_h)

    return np.maximum(np.nan_to_num(lnp_dbz, neginf=CF_DBZ_SEM_ECO), CF_DBZ_SEM_ECO).astype(np.float32)

# -------------------------------------------------------------------------------------
def faixa_realimentacao_rainnet(ff_epsilon: float=CF_RAINNET_EPSILON) -> tuple:
    """
    faixa física de log(R + epsilon): de R = 0 até R de CF_DBZ_MAX_ESCALA. A saída do
    rainnet não é limitada; sem este corte, a rede ainda não treinada realimentada no
    rollout cresce ~17x por passo e estoura o float16 no 3º (2026-10-22: -204, -3594,
    NaN).

    :returns: (mínimo, máximo) no espaço log
    """
    lf_mm_h_max = float(obc.converte_dbz_para_mm_h(np.float64(CF_DBZ_MAX_ESCALA)))

    return float(np.log(ff_epsilon)), float(np.log(lf_mm_h_max + ff_epsilon))

# < transformações por backend >---------------------------------------------------------

class TransformacaoRainNet:
    """
    log(R + 0.01) na entrada e no alvo (regressão).
    """
    nome = "rainnet"

    def __init__(self, ff_epsilon: float=CF_RAINNET_EPSILON):
        self.epsilon = ff_epsilon

    def entrada(self, fnp_dbz: np.ndarray) -> np.ndarray:
        """
        :param fnp_dbz: (quadros, H, W) em dBZ

        :returns: (quadros, H, W) float32
        """
        return np.log(dbz_para_mm_h(fnp_dbz) + self.epsilon).astype(np.float32)

    def alvo(self, fnp_dbz: np.ndarray) -> np.ndarray:
        return self.entrada(fnp_dbz)

    def saida_para_dbz(self, fnp_saida: np.ndarray) -> np.ndarray:
        """
        inversa: exp(x) - 0.01, negativo vira 0, e então dBZ.
        """
        lnp_mm_h = np.exp(fnp_saida.astype(np.float64)) - self.epsilon

        return mm_h_para_dbz(np.where(lnp_mm_h > 0.0, lnp_mm_h, 0.0))

    def realimenta(self, fnp_saida: np.ndarray) -> np.ndarray:
        """
        saída da rede no formato da entrada (previsão recursiva): já é log(R + 0.01),
        limitada à faixa física (como unt_treino.realimenta).
        """
        lf_min, lf_max = faixa_realimentacao_rainnet(self.epsilon)

        return np.clip(fnp_saida, lf_min, lf_max).astype(np.float32)

    def como_dict(self) -> dict:
        return {"nome": self.nome, "epsilon": self.epsilon}

# -------------------------------------------------------------------------------------
class TransformacaoSmaat:
    """
    R / escala_mm_h na entrada e no alvo (regressão, perda MSE).
    """
    nome = "smaat"

    def __init__(self, ff_escala_mm_h: float=CF_SMAAT_ESCALA_MM_H):
        self.escala_mm_h = ff_escala_mm_h

    def entrada(self, fnp_dbz: np.ndarray) -> np.ndarray:
        return (dbz_para_mm_h(fnp_dbz) / self.escala_mm_h).astype(np.float32)

    def alvo(self, fnp_dbz: np.ndarray) -> np.ndarray:
        return self.entrada(fnp_dbz)

    def saida_para_dbz(self, fnp_saida: np.ndarray) -> np.ndarray:
        return mm_h_para_dbz(fnp_saida.astype(np.float64) * self.escala_mm_h)

    def realimenta(self, fnp_saida: np.ndarray) -> np.ndarray:
        """
        saída no formato da entrada (previsão recursiva), sem chuva negativa.
        """
        return np.maximum(fnp_saida, 0.0).astype(np.float32)

    def como_dict(self) -> dict:
        return {"nome": self.nome, "escala_mm_h": self.escala_mm_h}

# -------------------------------------------------------------------------------------
class TransformacaoPytorch:
    """
    entrada em dBZ escalado + máscara; alvo em classes (bins de intensidade).

    limites (dBZ) [10, 20, ..., 55] definem len(limites) + 1 classes: classe 0 = abaixo
    de 10 dBZ (inclui o "sem eco"), última = acima do último limite.
    """
    nome = "pytorch"

    def __init__(self, flst_limites_dbz: list, fs_saida: str="mediana"):
        if fs_saida not in ("mediana", "esperanca"):
            raise ValueError(f"saída inválida: {fs_saida}")

        self.limites_dbz = [float(lf) for lf in flst_limites_dbz]
        self.saida = fs_saida
        self.centros_dbz = self._centros()

    def _centros(self) -> np.ndarray:
        """
        valor (dBZ) que representa cada classe: o ponto médio entre limites; nas pontas,
        meio intervalo além do limite (classe 0 = 5 dBZ com o limite 10).
        """
        llst_limites = self.limites_dbz
        llst_centros = [llst_limites[0] / 2.0]

        for li in range(len(llst_limites) - 1):
            llst_centros.append((llst_limites[li] + llst_limites[li + 1]) / 2.0)

        llst_centros.append(llst_limites[-1] + 5.0)

        return np.asarray(llst_centros, dtype=np.float32)

    @property
    def num_classes(self) -> int:
        return len(self.limites_dbz) + 1

    def entrada(self, fnp_dbz: np.ndarray) -> np.ndarray:
        """
        :param fnp_dbz: (quadros, H, W) em dBZ

        :returns: (quadros + 1, H, W): dBZ escalado em [0, 1] e, por último, a máscara
                  de cobertura do quadro mais recente
        """
        lnp_valido = mascara_valida(fnp_dbz[-1:]).astype(np.float32)
        lnp_dbz = np.nan_to_num(fnp_dbz, nan=CF_DBZ_MIN_ESCALA)
        lnp_escalado = np.clip((lnp_dbz - CF_DBZ_MIN_ESCALA) /
                               (CF_DBZ_MAX_ESCALA - CF_DBZ_MIN_ESCALA), 0.0, 1.0)

        return np.concatenate([lnp_escalado, lnp_valido]).astype(np.float32)

    def alvo(self, fnp_dbz: np.ndarray) -> np.ndarray:
        """
        :param fnp_dbz: (H, W) em dBZ

        :returns: (H, W) int64 com a classe de cada pixel (CI_CLASSE_IGNORADA fora da
                  cobertura)
        """
        lnp_classes = np.digitize(np.nan_to_num(fnp_dbz, nan=CF_DBZ_SEM_ECO),
                                  self.limites_dbz).astype(np.int64)
        lnp_classes[~mascara_valida(fnp_dbz)] = CI_CLASSE_IGNORADA

        return lnp_classes

    def probabilidades_para_dbz(self, fnp_probabilidades: np.ndarray) -> np.ndarray:
        """
        campo determinístico (dBZ) a partir das probabilidades por classe.

        mediana: centro da classe onde a probabilidade acumulada passa de 0.5 (não borra
        os núcleos); esperança: média dos centros ponderada pelas probabilidades.

        :param fnp_probabilidades: (classes, H, W)

        :returns: (H, W) em dBZ; a classe 0 vira o "sem eco" (-32)
        """
        if "esperanca" == self.saida:
            lnp_dbz = np.tensordot(self.centros_dbz, fnp_probabilidades, axes=(0, 0))
            return lnp_dbz.astype(np.float32)

        lnp_acumulada = np.cumsum(fnp_probabilidades, axis=0)
        lnp_classe = np.argmax(lnp_acumulada >= 0.5, axis=0)

        lnp_dbz = self.centros_dbz[lnp_classe]
        lnp_dbz[0 == lnp_classe] = CF_DBZ_SEM_ECO

        return lnp_dbz.astype(np.float32)

    def como_dict(self) -> dict:
        return {"nome": self.nome, "limites_dbz": self.limites_dbz, "saida": self.saida}

# -------------------------------------------------------------------------------------
def transformacao_de_dict(fdct: dict):
    """
    reconstrói a transformação gravada no checkpoint (como_dict()).

    :raises ValueError: nome desconhecido
    """
    ls_nome = fdct["nome"]

    if "rainnet" == ls_nome:
        return TransformacaoRainNet(fdct.get("epsilon", CF_RAINNET_EPSILON))

    if "smaat" == ls_nome:
        return TransformacaoSmaat(fdct.get("escala_mm_h", CF_SMAAT_ESCALA_MM_H))

    if "pytorch" == ls_nome:
        return TransformacaoPytorch(fdct["limites_dbz"], fdct.get("saida", "mediana"))

    raise ValueError(f"transformação desconhecida: {ls_nome}")

# < padding e recorte >------------------------------------------------------------------

def calcula_padding(ft_shape: tuple, fi_multiplo: int=CI_MULTIPLO_PADDING) -> tuple:
    """
    padding (cima, baixo, esquerda, direita) até o próximo múltiplo, dividido entre os
    dois lados.

    :param ft_shape: (..., H, W)

    :returns: (cima, baixo, esquerda, direita)
    """
    li_altura, li_largura = ft_shape[-2:]

    li_falta_h = (-li_altura) % fi_multiplo
    li_falta_w = (-li_largura) % fi_multiplo

    return (li_falta_h // 2, li_falta_h - li_falta_h // 2,
            li_falta_w // 2, li_falta_w - li_falta_w // 2)

# -------------------------------------------------------------------------------------
def aplica_padding(fnp_campo: np.ndarray, ft_padding: tuple) -> np.ndarray:
    """
    padding refletido nas duas últimas dimensões (como o np.pad "reflect" do RainNet).
    """
    li_cima, li_baixo, li_esq, li_dir = ft_padding
    llst_larguras = [(0, 0)] * (fnp_campo.ndim - 2) + [(li_cima, li_baixo), (li_esq, li_dir)]

    return np.pad(fnp_campo, llst_larguras, mode="reflect")

# -------------------------------------------------------------------------------------
def remove_padding(fnp_campo: np.ndarray, ft_padding: tuple) -> np.ndarray:
    """
    recorta de volta para a grade original.
    """
    li_cima, li_baixo, li_esq, li_dir = ft_padding
    li_altura, li_largura = fnp_campo.shape[-2:]

    return fnp_campo[..., li_cima:li_altura - li_baixo, li_esq:li_largura - li_dir]

# < the end >----------------------------------------------------------------------------
