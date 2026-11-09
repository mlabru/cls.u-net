# -*- coding: utf-8 -*-
"""
Centro Logístico de Simulação Meteorológica
Copyright (c) 1995-2026, Instituto de Controle do Espaço Aéreo

unt_mapa

feições geográficas (fronteiras, costa, estados) desenhadas nos mapas dos
três backends.

por que existe: o cf.STATES/BORDERS/COASTLINE do Cartopy relê e recorta o
shapefile a cada desenho. Com 6 frames por backend isso era mais da metade
do tempo de cada execução (só ler o shapefile de estados 10m custa ~6 s).
Como a área de cada radar é fixa, as geometrias são recortadas UMA vez por
área e guardadas em dois caches:

  1. em memória (functools.lru_cache), dentro do processo;
  2. em disco (arquivo .pkl por área), entre execuções. Diretório:
     $UNT_MAPA_CACHE (o unt_cfg.sh aponta para <UNT_PATH>/data/cache/mapa);
     sem a variável, ~/.cache/unt_mapa.

falha no cache em disco nunca impede o mapa: as geometrias são recalculadas.

2026.out  mlabru  portado de 25.oflow (ofl_mapa), sem mudança de lógica.

2026.out  mlabru  leitura/gravação do cache em disco em funções próprias;
                  nomes descritivos.

2026.out  mlabru  initial version (Linux/Python)
"""
# < imports >----------------------------------------------------------------------------

# python library
import functools
import hashlib
import logging
import os
import pathlib
import pickle
import typing

# cartopy
import cartopy.crs as ccrs
import cartopy.feature as cf

# < constants >--------------------------------------------------------------------------

# (feição do Cartopy, estilo do desenho)
MT_FEICOES = ((cf.BORDERS, {"linestyle": "--"}),
              (cf.COASTLINE, {"linewidth": 1.0}),
              (cf.STATES.with_scale("10m"), {"linewidth": 0.5, "edgecolor": "gray"}))

# versão do conteúdo do cache em disco -- INCREMENTAR se MT_FEICOES mudar,
# para que caches antigos não sejam reaproveitados
CI_VERSAO_CACHE = 1

# < logging >----------------------------------------------------------------------------

# logger
M_LOG = logging.getLogger(__name__)
M_LOG.setLevel(logging.INFO)

# -------------------------------------------------------------------------------------
def caminho_do_cache(ft_extensao: tuple) -> pathlib.Path:
    """
    caminho do arquivo de cache em disco de uma área.

    o nome do arquivo é um hash da área (arredondada a 1e-4 grau) e da
    versão do cache: áreas diferentes -> arquivos diferentes.

    :param ft_extensao: (lng_min, lng_max, lat_min, lat_max) em graus

    :returns: caminho do .pkl (pode não existir)
    """
    lpth_diretorio_padrao = pathlib.Path.home() / ".cache" / "unt_mapa"
    lpth_diretorio = pathlib.Path(os.environ.get("UNT_MAPA_CACHE", lpth_diretorio_padrao))

    llst_coordenadas = []

    for lf_coordenada in ft_extensao:
        llst_coordenadas.append(f"{lf_coordenada:.4f}")

    ls_chave = ",".join(llst_coordenadas) + f"|v{CI_VERSAO_CACHE}"
    ls_hash = hashlib.sha1(ls_chave.encode()).hexdigest()[:16]

    return lpth_diretorio / f"{ls_hash}.pkl"

# -------------------------------------------------------------------------------------
def _le_cache_em_disco(fpth_cache: pathlib.Path) -> typing.Optional[tuple]:
    """
    lê as geometrias do cache em disco.

    :param fpth_cache: arquivo do cache

    :returns: as geometrias, ou None se o arquivo não existe ou está ilegível
    """
    try:
        return pickle.loads(fpth_cache.read_bytes())

    # ausente, corrompido, de outra versão do Python, ...: recalcula
    except Exception:
        return None

# -------------------------------------------------------------------------------------
def _grava_cache_em_disco(fpth_cache: pathlib.Path, ft_geometrias: tuple) -> None:
    """
    grava as geometrias no cache em disco, de forma atômica: escreve num
    arquivo temporário e depois renomeia. Os processos dos backends rodam em
    paralelo e podem gravar o mesmo arquivo ao mesmo tempo; a renomeação
    garante que ninguém leia um arquivo pela metade.

    falha (ex.: sem permissão) só gera um aviso.

    :param fpth_cache: arquivo do cache
    :param ft_geometrias: geometrias a gravar
    """
    try:
        fpth_cache.parent.mkdir(parents=True, exist_ok=True)

        lpth_temporario = fpth_cache.with_suffix(f".{os.getpid()}.tmp")
        lpth_temporario.write_bytes(pickle.dumps(ft_geometrias))
        lpth_temporario.replace(fpth_cache)

    except OSError as le_erro:
        M_LOG.warning(f"[!] cache do mapa não gravado ({fpth_cache}): {le_erro}")

# -------------------------------------------------------------------------------------
def _recorta_feicoes(ft_extensao: tuple) -> tuple:
    """
    recorta, de cada feição de MT_FEICOES, as geometrias que cruzam a área
    (a parte lenta: lê os shapefiles).

    :param ft_extensao: (lng_min, lng_max, lat_min, lat_max) em graus

    :returns: uma tupla de geometrias por feição, na ordem de MT_FEICOES
    """
    llst_geometrias_por_feicao = []

    for lfeicao_cartopy, _ in MT_FEICOES:
        lt_geometrias = tuple(lfeicao_cartopy.intersecting_geometries(list(ft_extensao)))
        llst_geometrias_por_feicao.append(lt_geometrias)

    return tuple(llst_geometrias_por_feicao)

# -------------------------------------------------------------------------------------
@functools.lru_cache(maxsize=8)
def geometrias_da_area(ft_extensao: tuple) -> tuple:
    """
    geometrias de cada feição recortadas para a área -- do cache em memória
    (lru_cache), do cache em disco ou, em último caso, recortadas agora (e
    então gravadas no disco).

    :param ft_extensao: (lng_min, lng_max, lat_min, lat_max) em graus; precisa
                        ser tupla (o lru_cache exige argumentos imutáveis)

    :returns: uma tupla de geometrias por feição, na ordem de MT_FEICOES
    """
    lpth_cache = caminho_do_cache(ft_extensao)

    lt_geometrias = _le_cache_em_disco(lpth_cache)

    if lt_geometrias is not None:
        return lt_geometrias

    lt_geometrias = _recorta_feicoes(ft_extensao)
    _grava_cache_em_disco(lpth_cache, lt_geometrias)

    return lt_geometrias

# -------------------------------------------------------------------------------------
def desenha_feicoes_geograficas(fax, ft_extensao: tuple) -> None:
    """
    desenha fronteiras, costa e estados no eixo do mapa.

    :param fax: eixo Cartopy (projeção PlateCarree)
    :param ft_extensao: (lng_min, lng_max, lat_min, lat_max) em graus
    """
    lt_geometrias_por_feicao = geometrias_da_area(tuple(ft_extensao))

    for li_indice, (_, ldct_estilo) in enumerate(MT_FEICOES):
        lfeicao_recortada = cf.ShapelyFeature(lt_geometrias_por_feicao[li_indice],
                                              ccrs.PlateCarree(), facecolor="none",
                                              **ldct_estilo)
        fax.add_feature(lfeicao_recortada)

# < the end >----------------------------------------------------------------------------
