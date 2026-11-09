# -*- coding: utf-8 -*-
"""
Centro Logístico de Simulação Meteorológica
Copyright (c) 1995-2026, Instituto de Controle do Espaço Aéreo

unt_backend_comum

peças compartilhadas pelos três backends de nowcast por Deep Learning
(unt_pytorch, unt_rainnet, unt_smaat). Tudo o que deve ser IGUAL nos três
mora aqui, para que as saídas sejam diretamente comparáveis:

  - a relação Z-R (dBZ <-> mm/h);
  - a área do mapa (caixa geográfica em graus) e os metadados da grade;
  - as validações da entrada (campo todo inválido, ausência de chuva);
  - o desenho e a gravação dos frames PNG (mesmo layout de mapa).

Cada backend só implementa o que o torna diferente: o pré-processamento
do seu artigo, a rede e a estratégia de previsão (recursiva ou direta).

contrato de um backend (ver untMain.MDCT_MODULOS_BACKEND): o módulo expõe

    gera_nowcast(fnp_refletividade_dbz, ft_radar, fi_num_passos,
                 fi_passo_minutos, fs_prefixo_saida) -> ResultadoNowcast

grava <fs_prefixo_saida>_NN.png (um por passo) e devolve a previsão em
dBZ e em mm/h -- ou None nas duas quando não há chuva para prever.

2026.out  mlabru  portado de 25.oflow (ofl_backend_comum); textos dos
                  backends de optical flow trocados pelos de Deep Learning.

2026.out  mlabru  extraído dos três backends (código antes triplicado)
"""
# < imports >----------------------------------------------------------------------------

# python library
import logging
import typing

# numPy
import numpy as np

# matplotlib
import matplotlib.pyplot as plt

# cartopy
import cartopy.crs as ccrs

# mapa (feições recortadas uma vez por extensão)
import work.unt_mapa as om

# < constants >--------------------------------------------------------------------------

# relação Z-R de Marshall-Palmer: Z = CF_ZR_A * R ^ CF_ZR_B
# (Z em mm^6/m^3, R em mm/h). Os três backends usam os mesmos valores.
CF_ZR_A = 200.0
CF_ZR_B = 1.6

# quilômetros por grau de latitude (aproximação esférica). Um grau de
# longitude vale isso vezes cos(latitude).
CF_KM_POR_GRAU = 111.32

# faixa de refletividade exibida nos mapas: abaixo do mínimo o pixel fica
# transparente (ruído/chuva insignificante), acima do máximo satura a cor
CF_DBZ_MIN_EXIBIDO = 10.0
CF_DBZ_MAX_EXIBIDO = 65.0

# paleta de cores dos mapas de refletividade
CS_PALETA_CORES = "turbo"

# tamanho da figura (polegadas) e resolução: 10x8 pol a 100 dpi = 1000x800
# px, já acima dos 800x600 do GIF (util/utl_gifs.GIF_SIZE). Mais dpi só
# gastaria tempo gerando pixels que o GIF descarta.
CT_TAMANHO_FIGURA_POL = (10, 8)
CI_PLOT_DPI = 100

# < logging >----------------------------------------------------------------------------

# logger
M_LOG = logging.getLogger(__name__)
M_LOG.setLevel(logging.INFO)

# < tipos >------------------------------------------------------------------------------

class GeometriaRadar(typing.NamedTuple):
    """
    posição do radar e tamanho do mapa a desenhar em volta dele.

    raio_mapa_km é o meio-lado da GRADE do CAPPI (ex.: 300 km no nwc400v2),
    não o alcance nominal do radar -- é ele que define a área do mapa.
    """
    latitude: float
    longitude: float
    altitude_m: float
    raio_mapa_km: float


class CaixaGeografica(typing.NamedTuple):
    """
    limites do mapa em graus (retângulo lat/lon em volta do radar).
    """
    lng_min: float
    lng_max: float
    lat_min: float
    lat_max: float

    def como_extensao(self) -> list:
        """
        limites no formato [x1, x2, y1, y2] esperado pelo matplotlib/Cartopy.
        """
        return [self.lng_min, self.lng_max, self.lat_min, self.lat_max]


class ResultadoNowcast(typing.NamedTuple):
    """
    o que todo backend devolve.

    previsao_dbz / previsao_mm_h: arrays (num_passos, linhas, colunas), ou
    None quando não havia chuva para prever.
    metadados: descrição da grade (ver monta_metadados), sempre presente.
    """
    previsao_dbz: typing.Optional[np.ndarray]
    previsao_mm_h: typing.Optional[np.ndarray]
    metadados: dict

# < conversão Z-R >----------------------------------------------------------------------

def converte_dbz_para_mm_h(fnp_dbz: np.ndarray) -> np.ndarray:
    """
    converte refletividade (dBZ) em taxa de chuva (mm/h) pela relação Z-R.

    passo a passo: Z = 10 ^ (dBZ / 10)  e  R = (Z / a) ^ (1 / b)

    :param fnp_dbz: refletividade em dBZ (qualquer shape)

    :returns: taxa de chuva em mm/h (mesmo shape). NaN continua NaN.
    """
    lnp_z_linear = np.power(10.0, fnp_dbz / 10.0)

    return np.power(lnp_z_linear / CF_ZR_A, 1.0 / CF_ZR_B)

# -------------------------------------------------------------------------------------
def converte_mm_h_para_dbz(fnp_mm_h: np.ndarray) -> np.ndarray:
    """
    converte taxa de chuva (mm/h) de volta em refletividade (dBZ).

    passo a passo: Z = a * R ^ b  e  dBZ = 10 * log10(Z)

    :param fnp_mm_h: taxa de chuva em mm/h (qualquer shape)

    :returns: refletividade em dBZ (mesmo shape). R = 0 vira -inf, o que é
              esperado ("sem chuva"); por isso o aviso de log(0) é silenciado.
    """
    lnp_z_linear = CF_ZR_A * np.power(fnp_mm_h, CF_ZR_B)

    with np.errstate(divide="ignore"):
        return 10.0 * np.log10(lnp_z_linear)

# < preparação e validação da entrada >--------------------------------------------------

def converte_frame_unico_em_sequencia(fnp_refletividade_dbz: np.ndarray,
                                      fi_num_copias: int) -> np.ndarray:
    """
    garante que a entrada seja uma SEQUÊNCIA de frames (3D).

    as redes recebem uma pilha de frames consecutivos, então um único frame
    (2D) não basta: ele é repetido fi_num_copias vezes. Como os frames são
    iguais, a rede não vê movimento algum.

    :param fnp_refletividade_dbz: um frame (linhas, colunas) ou uma sequência
                                  (frames, linhas, colunas), em dBZ
    :param fi_num_copias: quantas cópias montar quando vier um frame só

    :returns: a sequência 3D (a própria entrada, se já era 3D)
    """
    # já é uma sequência: nada a fazer
    if 3 == fnp_refletividade_dbz.ndim:
        return fnp_refletividade_dbz

    M_LOG.warning("[!] Apenas um frame fornecido. "
                  "A rede não verá movimento (frames repetidos).")

    return np.stack([fnp_refletividade_dbz] * fi_num_copias)

# -------------------------------------------------------------------------------------
def valida_refletividade(fnp_refletividade_dbz: np.ndarray) -> None:
    """
    recusa uma entrada sem nenhum valor utilizável.

    :param fnp_refletividade_dbz: sequência de refletividade em dBZ

    :raises ValueError: se todos os valores forem NaN/infinitos
    """
    M_LOG.info(f"[+] Reflectivity shape: {fnp_refletividade_dbz.shape}")

    if not np.isfinite(fnp_refletividade_dbz).any():
        raise ValueError("Todos os valores do campo de refletividade são não finitos.")

# -------------------------------------------------------------------------------------
def substitui_invalidos_por_zero(fnp_chuva_mm_h: np.ndarray) -> np.ndarray:
    """
    troca NaN e infinitos por 0 mm/h (fora da cobertura = "sem chuva").

    as redes não aceitam NaN.

    :param fnp_chuva_mm_h: taxa de chuva em mm/h

    :returns: cópia sem NaN/infinitos
    """
    return np.nan_to_num(fnp_chuva_mm_h, nan=0.0, posinf=0.0, neginf=0.0)

# -------------------------------------------------------------------------------------
def existe_chuva(fnp_chuva_mm_h: np.ndarray) -> bool:
    """
    diz se há alguma chuva no campo -- sem chuva não há o que prever.

    :param fnp_chuva_mm_h: taxa de chuva em mm/h, já sem NaN
                           (ver substitui_invalidos_por_zero)

    :returns: True se algum pixel tem chuva (> 0 mm/h)
    """
    lf_chuva_maxima = float(fnp_chuva_mm_h.max())

    M_LOG.info(f"[✓] Conversão dBZ → mm/h concluída. Máx R: {lf_chuva_maxima:.2f} mm/h")

    if lf_chuva_maxima <= 0.0:
        M_LOG.warning("Nenhum valor válido de chuva detectado. Nowcast ignorado.")
        return False

    return True

# < geografia e metadados >--------------------------------------------------------------

def calcula_caixa_geografica(ft_radar: GeometriaRadar) -> CaixaGeografica:
    """
    calcula o retângulo lat/lon (graus) do mapa em volta do radar.

    converte o raio do mapa (km) em graus: em latitude é sempre
    raio / 111.32; em longitude o grau "encolhe" com o cosseno da latitude.

    :param ft_radar: posição do radar e raio do mapa

    :returns: limites do mapa em graus
    """
    lf_delta_lat_graus = ft_radar.raio_mapa_km / CF_KM_POR_GRAU
    lf_delta_lng_graus = ft_radar.raio_mapa_km / (CF_KM_POR_GRAU *
                                                  np.cos(np.radians(ft_radar.latitude)))

    return CaixaGeografica(lng_min=ft_radar.longitude - lf_delta_lng_graus,
                           lng_max=ft_radar.longitude + lf_delta_lng_graus,
                           lat_min=ft_radar.latitude - lf_delta_lat_graus,
                           lat_max=ft_radar.latitude + lf_delta_lat_graus)

# -------------------------------------------------------------------------------------
def monta_metadados(ft_radar: GeometriaRadar, fnp_refletividade_dbz: np.ndarray) -> dict:
    """
    descreve a grade no formato de metadados do pySteps (o pySteps exige
    este dicionário; os outros backends o devolvem para manter o contrato).

    :param ft_radar: posição do radar e raio do mapa
    :param fnp_refletividade_dbz: sequência de entrada (só o shape é usado)

    :returns: dicionário com unidade, relação Z-R e posição/tamanho da grade
    """
    li_num_linhas, li_num_colunas = fnp_refletividade_dbz.shape[-2:]

    lt_caixa = calcula_caixa_geografica(ft_radar)

    return {
        # unidade dos dados
        "transform": "dB",
        "unit": "dBZ",

        # valor "sem chuva" e relação Z-R
        "zerovalue": np.nan,
        "zr_a": CF_ZR_A,
        "zr_b": CF_ZR_B,

        # limites da grade em graus
        "x1": lt_caixa.lng_min,
        "x2": lt_caixa.lng_max,
        "y1": lt_caixa.lat_min,
        "y2": lt_caixa.lat_max,

        # tamanho de cada pixel em graus
        "xpixelsize": (lt_caixa.lng_max - lt_caixa.lng_min) / li_num_colunas,
        "ypixelsize": (lt_caixa.lat_max - lt_caixa.lat_min) / li_num_linhas,

        # dimensões da grade
        "xsize": li_num_colunas,
        "ysize": li_num_linhas,

        # a linha 0 da matriz é o SUL (origem embaixo)
        "xorigin": "lower",
        "yorigin": "lower",

        "projection": "latlon",
    }

# < desenho dos frames >-----------------------------------------------------------------

def desenha_campo_com_imshow(fax, fnp_campo_exibido: np.ndarray, fdct_metadados: dict,
                             ft_caixa: CaixaGeografica, fs_titulo: str):
    """
    desenha a refletividade no mapa com o imshow do matplotlib (usado pelos
    três backends).

    :param fax: eixo Cartopy (GeoAxes) já criado
    :param fnp_campo_exibido: campo em dBZ, com NaN onde não deve ser pintado
    :param fdct_metadados: metadados da grade (não usados aqui; existem para
                           que todas as funções de desenho tenham a mesma
                           assinatura)
    :param ft_caixa: limites do mapa em graus
    :param fs_titulo: título do frame

    :returns: o eixo onde o campo foi desenhado
    """
    limagem = fax.imshow(fnp_campo_exibido, origin="lower",
                         extent=ft_caixa.como_extensao(),
                         transform=ccrs.PlateCarree(), cmap=CS_PALETA_CORES,
                         vmin=CF_DBZ_MIN_EXIBIDO, vmax=CF_DBZ_MAX_EXIBIDO)

    plt.colorbar(limagem, ax=fax, label="dBZ", shrink=0.8)
    fax.set_title(fs_titulo)

    return fax

# -------------------------------------------------------------------------------------
def esconde_valores_baixos(fnp_campo_dbz: np.ndarray) -> np.ndarray:
    """
    devolve uma cópia do campo com NaN abaixo de CF_DBZ_MIN_EXIBIDO, para que
    esses pixels fiquem transparentes no mapa (só o eco relevante aparece).

    :param fnp_campo_dbz: campo em dBZ (2D)

    :returns: cópia pronta para desenhar
    """
    lnp_campo_exibido = np.copy(fnp_campo_dbz)
    lnp_campo_exibido[lnp_campo_exibido < CF_DBZ_MIN_EXIBIDO] = np.nan

    return lnp_campo_exibido

# -------------------------------------------------------------------------------------
def salva_um_frame(fnp_campo_dbz: np.ndarray, ft_radar: GeometriaRadar,
                   fdct_metadados: dict, fs_titulo: str, fs_arquivo_png: str,
                   ff_desenha_campo: typing.Callable) -> None:
    """
    monta e grava um frame PNG: mapa + refletividade + feições geográficas
    + grade lat/lon + marcador do radar.

    :param fnp_campo_dbz: campo previsto em dBZ (2D)
    :param ft_radar: posição do radar e raio do mapa
    :param fdct_metadados: metadados da grade (ver monta_metadados)
    :param fs_titulo: título do frame
    :param fs_arquivo_png: caminho do PNG a gravar
    :param ff_desenha_campo: função que pinta o campo no eixo (mesma
                             assinatura de desenha_campo_com_imshow)
    """
    lt_caixa = calcula_caixa_geografica(ft_radar)

    # figura e eixo de mapa (projeção lat/lon simples)
    lfigura = plt.figure(figsize=CT_TAMANHO_FIGURA_POL)
    leixo = plt.axes(projection=ccrs.PlateCarree())
    leixo.set_extent(lt_caixa.como_extensao(), crs=ccrs.PlateCarree())

    # campo de refletividade (cada backend pode pintar do seu jeito)
    leixo = ff_desenha_campo(leixo, esconde_valores_baixos(fnp_campo_dbz),
                             fdct_metadados, lt_caixa, fs_titulo)

    # fronteiras, costa e estados
    om.desenha_feicoes_geograficas(leixo, tuple(lt_caixa))

    # grade de latitude/longitude
    leixo.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False)

    # cruz vermelha na posição do radar
    leixo.plot(ft_radar.longitude, ft_radar.latitude, "r+", markersize=10, label="Radar")

    # grava sem bordas brancas e libera a memória da figura
    plt.savefig(fs_arquivo_png, dpi=CI_PLOT_DPI, bbox_inches="tight")
    plt.close(lfigura)

# -------------------------------------------------------------------------------------
def salva_frames_da_previsao(fnp_previsao_dbz: np.ndarray, ft_radar: GeometriaRadar,
                             fdct_metadados: dict, fs_nome_backend: str,
                             fi_passo_minutos: int, fs_prefixo_saida: str,
                             ff_desenha_campo: typing.Callable=desenha_campo_com_imshow) -> None:
    """
    grava um PNG por passo previsto: <fs_prefixo_saida>_01.png, _02.png, ...

    :param fnp_previsao_dbz: previsão em dBZ (num_passos, linhas, colunas)
    :param ft_radar: posição do radar e raio do mapa
    :param fdct_metadados: metadados da grade (ver monta_metadados)
    :param fs_nome_backend: nome mostrado no título (ex.: "rainnet")
    :param fi_passo_minutos: minutos entre passos (para o título "+NN min")
    :param fs_prefixo_saida: caminho + prefixo dos arquivos PNG
    :param ff_desenha_campo: função que pinta o campo (padrão: imshow)
    """
    li_num_passos = fnp_previsao_dbz.shape[0]

    for li_passo in range(1, li_num_passos + 1):
        ls_titulo = (f"Nowcast +{li_passo * fi_passo_minutos} min "
                     f"(Raio {ft_radar.raio_mapa_km:g}km) [{fs_nome_backend}]")

        ls_arquivo_png = f"{fs_prefixo_saida}_{li_passo:02d}.png"

        # o passo N está na posição N-1 do array
        salva_um_frame(fnp_previsao_dbz[li_passo - 1], ft_radar, fdct_metadados,
                       ls_titulo, ls_arquivo_png, ff_desenha_campo)

    M_LOG.info(f"[✓] Imagens salvas com prefixo '{fs_prefixo_saida}_*.png'.")

# < the end >----------------------------------------------------------------------------
