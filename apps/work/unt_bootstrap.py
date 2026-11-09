# -*- coding: utf-8 -*-
"""
Centro Logístico de Simulação Meteorológica
Copyright (c) 1995-2026, Instituto de Controle do Espaço Aéreo

unt_bootstrap

incerteza amostral das métricas do untEval por bootstrap de BLOCOS de tempo (o bloco do
split do manifesto, padrão um dia UTC). As janelas deslizam de 10 em 10 min e são
fortemente correlacionadas: a unidade independente é o bloco, não a janela. Reamostrar
janelas daria intervalos estreitos demais.

entrada: somas por bloco (unt_avaliacao.AcumuladorLead.somas), gravadas pelo untEval em
"somas_por_bloco" do comparacao.json:

    {bloco: {rótulo: {passo: {"janelas", "limiares": {lim: {hits, misses,
                                                           false_alarms}},
                              "fss_somas": {lim: {escala: {erro, referencia}}}}}}}

cada reamostragem sorteia blocos com reposição (os MESMOS para todos os rótulos:
bootstrap pareado, a diferença candidato - persistência herda a correlação entre eles),
soma as contingências e os termos do FSS dos blocos sorteados e só então toma as razões,
como na agregação do conjunto todo.

métricas: ("csi", limiar, None) ou ("fss", limiar, escala).

2026.out  mlabru  initial version (Linux/Python)
"""
# < imports >----------------------------------------------------------------------------

# python library
import logging
import typing

# numPy
import numpy as np

# < constants >--------------------------------------------------------------------------

# rótulo da referência (o mesmo de untEval.CS_PERSISTENCIA)
CS_PERSISTENCIA = "persistencia"

# < logging >----------------------------------------------------------------------------

# logger
M_LOG = logging.getLogger(__name__)
M_LOG.setLevel(logging.INFO)

# < reamostragem >-----------------------------------------------------------------------

def pesos_reamostrados(fi_blocos: int, fi_reamostras: int, fi_semente: int) -> np.ndarray:
    """
    quantas vezes cada bloco foi sorteado em cada reamostragem (com reposição).

    :returns: matriz (reamostras, blocos) de contagens
    """
    lrng = np.random.default_rng(fi_semente)
    lnp_sorteios = lrng.integers(0, fi_blocos, size=(fi_reamostras, fi_blocos))

    lnp_pesos = np.zeros((fi_reamostras, fi_blocos))
    np.add.at(lnp_pesos, (np.arange(fi_reamostras)[:, None], lnp_sorteios), 1.0)

    return lnp_pesos

# -------------------------------------------------------------------------------------
def _termos(fdct_passo: typing.Optional[dict], ft_metrica: tuple) -> tuple:
    """
    (numerador, denominador) de uma métrica num bloco; (0, 0) se ausente.
    """
    ls_tipo, ls_limiar, ls_escala = ft_metrica

    if fdct_passo is None:
        return 0.0, 0.0

    if "csi" == ls_tipo:
        ldct = fdct_passo.get("limiares", {}).get(ls_limiar)
        if ldct is None:
            return 0.0, 0.0
        return (float(ldct["hits"]),
                float(ldct["hits"] + ldct["misses"] + ldct["false_alarms"]))

    ldct = fdct_passo.get("fss_somas", {}).get(ls_limiar, {}).get(ls_escala)
    if ldct is None:
        return 0.0, 0.0

    return float(ldct["erro"]), float(ldct["referencia"])

# -------------------------------------------------------------------------------------
def distribuicao(fdct_somas_por_bloco: dict, fs_rotulo: str, fs_passo: str,
                 ft_metrica: tuple, fnp_pesos: np.ndarray) -> np.ndarray:
    """
    valor da métrica em cada reamostragem (NaN onde ela é indefinida).

    :param fdct_somas_por_bloco: ver o topo do módulo
    :param fs_rotulo: rótulo (backend, candidato ou persistencia)
    :param fs_passo: passo (texto, "1".."6")
    :param ft_metrica: ("csi", limiar, None) ou ("fss", limiar, escala)
    :param fnp_pesos: pesos_reamostrados(), colunas na ordem de sorted(blocos)

    :returns: vetor (reamostras,)
    """
    llst_termos = [_termos(fdct_somas_por_bloco[ls_bloco].get(fs_rotulo, {}).get(fs_passo),
                           ft_metrica)
                   for ls_bloco in sorted(fdct_somas_por_bloco)]

    lnp_termos = np.array(llst_termos, dtype=np.float64).reshape(-1, 2)
    lnp_numerador = fnp_pesos @ lnp_termos[:, 0]
    lnp_denominador = fnp_pesos @ lnp_termos[:, 1]

    with np.errstate(divide="ignore", invalid="ignore"):
        lnp_razao = np.where(lnp_denominador > 0.0, lnp_numerador / lnp_denominador, np.nan)

    return lnp_razao if "csi" == ft_metrica[0] else 1.0 - lnp_razao

# -------------------------------------------------------------------------------------
def intervalo(fnp_valores: np.ndarray, ff_nivel: float) -> typing.Optional[list]:
    """
    intervalo percentil bilateral de nível ff_nivel (ex.: 0.90 -> P5..P95); None se a
    métrica é indefinida em todas as reamostragens.
    """
    lnp_definidos = fnp_valores[np.isfinite(fnp_valores)]

    if 0 == lnp_definidos.size:
        return None

    lf_cauda = 100.0 * (1.0 - ff_nivel) / 2.0

    return [float(np.percentile(lnp_definidos, lf_cauda)),
            float(np.percentile(lnp_definidos, 100.0 - lf_cauda))]

# -------------------------------------------------------------------------------------
def compara(fdct_somas_por_bloco: dict, fs_rotulo: str, fs_referencia: str, fs_passo: str,
            ft_metrica: tuple, fnp_pesos: np.ndarray, ff_nivel: float) -> dict:
    """
    diferença pareada rótulo - referência numa métrica.

    :returns: {"ic": [lo, hi] ou None, "p_melhor": fração das reamostragens em que o
              rótulo supera a referência (None se indefinida)}
    """
    lnp_diferenca = (distribuicao(fdct_somas_por_bloco, fs_rotulo, fs_passo, ft_metrica,
                                  fnp_pesos) -
                     distribuicao(fdct_somas_por_bloco, fs_referencia, fs_passo, ft_metrica,
                                  fnp_pesos))

    lnp_definidas = lnp_diferenca[np.isfinite(lnp_diferenca)]
    lf_p_melhor = float(np.mean(lnp_definidas > 0.0)) if lnp_definidas.size else None

    return {"ic": intervalo(lnp_diferenca, ff_nivel), "p_melhor": lf_p_melhor}

# < resumo para o comparacao.json >------------------------------------------------------

def metricas_presentes(fdct_somas_por_bloco: dict) -> list:
    """
    todas as métricas (csi por limiar, fss por limiar e escala) que aparecem nas somas.
    """
    lset_metricas = set()

    for ldct_rotulos in fdct_somas_por_bloco.values():
        for ldct_passos in ldct_rotulos.values():
            for ldct_passo in ldct_passos.values():
                for ls_limiar in ldct_passo.get("limiares", {}):
                    lset_metricas.add(("csi", ls_limiar, None))
                for ls_limiar, ldct_escalas in ldct_passo.get("fss_somas", {}).items():
                    for ls_escala in ldct_escalas:
                        lset_metricas.add(("fss", ls_limiar, ls_escala))

    return sorted(lset_metricas, key=lambda lt: (lt[0], float(lt[1]), float(lt[2] or 0)))

# -------------------------------------------------------------------------------------
def _grava_em(fdct_destino: dict, ft_metrica: tuple, fx_valor) -> None:
    """
    fdct_destino["csi"][limiar] ou fdct_destino["fss"][limiar][escala] = fx_valor.
    """
    ls_tipo, ls_limiar, ls_escala = ft_metrica
    ldct_tipo = fdct_destino.setdefault(ls_tipo, {})

    if ls_escala is None:
        ldct_tipo[ls_limiar] = fx_valor
    else:
        ldct_tipo.setdefault(ls_limiar, {})[ls_escala] = fx_valor

# -------------------------------------------------------------------------------------
def resume(fdct_somas_por_bloco: dict, fi_reamostras: int, ff_nivel: float,
           fi_semente: int, fs_referencia: str=CS_PERSISTENCIA) -> dict:
    """
    intervalos de cada rótulo, passo e métrica, e da diferença para a referência.

    :returns: {"blocos", "reamostras", "nivel", "semente", "resultados": {rótulo: {passo:
              {"ic": {"csi": {lim: [lo, hi]}, "fss": {lim: {esc: [lo, hi]}}},
               "dif_<referencia>": {"csi": {lim: {"ic", "p_melhor"}}, "fss": ...}}}}}
    """
    li_blocos = len(fdct_somas_por_bloco)
    ldct_resumo = {"blocos": li_blocos, "reamostras": fi_reamostras, "nivel": ff_nivel,
                   "semente": fi_semente, "resultados": {}}

    if 0 == li_blocos:
        return ldct_resumo

    lnp_pesos = pesos_reamostrados(li_blocos, fi_reamostras, fi_semente)
    llst_metricas = metricas_presentes(fdct_somas_por_bloco)

    lset_rotulos = set()
    lset_passos = set()
    for ldct_rotulos in fdct_somas_por_bloco.values():
        lset_rotulos.update(ldct_rotulos)
        for ldct_passos in ldct_rotulos.values():
            lset_passos.update(ldct_passos)

    for ls_rotulo in sorted(lset_rotulos):
        for ls_passo in sorted(lset_passos, key=int):
            ldct_passo = {"ic": {}}

            for lt_metrica in llst_metricas:
                lnp_valores = distribuicao(fdct_somas_por_bloco, ls_rotulo, ls_passo,
                                           lt_metrica, lnp_pesos)
                _grava_em(ldct_passo["ic"], lt_metrica, intervalo(lnp_valores, ff_nivel))

                if ls_rotulo != fs_referencia and fs_referencia in lset_rotulos:
                    _grava_em(ldct_passo.setdefault(f"dif_{fs_referencia}", {}), lt_metrica,
                              compara(fdct_somas_por_bloco, ls_rotulo, fs_referencia,
                                      ls_passo, lt_metrica, lnp_pesos, ff_nivel))

            ldct_resumo["resultados"].setdefault(ls_rotulo, {})[ls_passo] = ldct_passo

    return ldct_resumo

# < the end >----------------------------------------------------------------------------
