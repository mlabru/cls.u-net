# -*- coding: utf-8 -*-
"""
Centro Logístico de Simulação Meteorológica
Copyright (c) 1995-2026, Instituto de Controle do Espaço Aéreo

unt_verify

verificação objetiva do nowcast contra o volume observado. O "target" de
cada sequência do 27.nowcast é sempre um volume observado (nunca
interpolado), então o passo previsto que cai no horário do target pode ser
comparado diretamente com ele.

métricas (só nos pixels cobertos pelo radar no target, i.e. não-NaN):
  - categóricas, por limiar (dBZ): POD, FAR, CSI (+ acertos, perdas e
    falsos alarmes). Um pixel é "evento" se tem dBZ >= limiar.
      POD = acertos / (acertos + perdas)              (quanto do observado foi previsto)
      FAR = falsos / (acertos + falsos)               (quanto do previsto era falso)
      CSI = acertos / (acertos + perdas + falsos)     (nota geral, 1 = perfeito)
  - contínuas, nos pixels com eco (obs ou previsão >= CF_ECHO_DBZ): MAE e
    bias em dBZ, com os dois campos limitados por baixo a CF_ECHO_DBZ. Sem
    esse limite, um eco perdido custaria >= 42 dBZ (contra o "sem eco" de
    -32) e o MAE mediria acerto de área, que já é o papel do CSI.

  - FSS (Fractions Skill Score, Roberts & Lean 2008) por limiar e escala
    (janela quadrada de N km), calculado sobre os campos 2D com a cobertura
    como máscara: a fração de cada janela considera só os pixels cobertos.
  - frequency bias por limiar: (acertos + falsos) / (acertos + perdas).

a persistência (último input repetido) é avaliada com as mesmas métricas,
como referência: um nowcast que não supera a persistência não agrega nada.

ATENÇÃO: as chaves do JSON gravado ("lead", "backends", "persistencia",
"limiares", "csi", ...) são lidas por cls.test/tst_report.sh.
Não renomear. As chaves "fss", "fss_somas" e "freq_bias" são acréscimos do 25.u-net.

2026.out  mlabru  calcula_somas_fss(): termos do FSS somados nos pixels ("fss_somas"),
                  para o FSS agregado de várias janelas (somas antes da razão).

2026.out  mlabru  portado de 25.oflow (ofl_verify); + FSS (calcula_fss),
                  frequency bias e verifica_lead() para a avaliação
                  multi-lead do untEval.

2026.out  mlabru  nomes descritivos; calcula_metricas() dividida em
                  categóricas e contínuas; formata_resumo() sem compreensões
                  aninhadas.

2026.out  mlabru  initial version (Linux/Python)
"""
# < imports >----------------------------------------------------------------------------

# python library
import json
import logging
import pathlib
import typing

# numPy
import numpy as np

# sciPy (média móvel do FSS)
from scipy import ndimage

# catálogo (valor "sem eco" dos CAPPIs)
import work.unt_catalog as oc

# < constants >--------------------------------------------------------------------------

# limiares (dBZ) das métricas categóricas: 20 (chuva fraca), 30 (moderada),
# 36 (convecção)
CT_LIMIARES_DBZ = (20.0, 30.0, 36.0)

# limiares (dBZ) e escalas (km = pixels, grade de 1 km) do FSS
CT_LIMIARES_FSS_DBZ = (30.0, 36.0)
CT_ESCALAS_FSS_KM = (1, 5, 10, 20)

# a partir deste valor (dBZ) um pixel conta como "eco" para MAE/bias
CF_ECHO_DBZ = 10.0

# piso (dBZ) aplicado antes das métricas: -inf (R = 0 na conversão
# mm/h -> dBZ), NaN e valores abaixo do "sem eco" viram este valor
CF_PISO_DBZ = oc.CF_DBZ_SEM_ECO

# < logging >----------------------------------------------------------------------------

# logger
M_LOG = logging.getLogger(__name__)
M_LOG.setLevel(logging.INFO)

# -------------------------------------------------------------------------------------
def calcula_passo_do_target(fi_timestamp_ultima_entrada: int, fi_timestamp_target: int,
                            fi_passo_minutos: int) -> int:
    """
    descobre qual passo da previsão (1, 2, ...) cai no horário do target.

    nas sequências do 27.nowcast o target vem um intervalo depois da última
    entrada, então o resultado normalmente é 1.

    :param fi_timestamp_ultima_entrada: epoch UTC do último CAPPI de entrada
    :param fi_timestamp_target: epoch UTC do target observado
    :param fi_passo_minutos: minutos entre passos da previsão

    :returns: número do passo (no mínimo 1)
    """
    lf_segundos_ate_target = fi_timestamp_target - fi_timestamp_ultima_entrada
    lf_passos_ate_target = lf_segundos_ate_target / (fi_passo_minutos * 60.0)

    li_passo = int(round(lf_passos_ate_target))

    return max(1, li_passo)

# -------------------------------------------------------------------------------------
def _razao_ou_none(ff_numerador: float, ff_denominador: float) -> typing.Optional[float]:
    """
    divisão que devolve None quando o denominador é zero (métrica
    indefinida, ex.: POD sem nenhum pixel observado acima do limiar).
    """
    if 0 == ff_denominador:
        return None

    return float(ff_numerador) / float(ff_denominador)

# -------------------------------------------------------------------------------------
def _aplica_piso(fnp_campo_dbz: np.ndarray) -> np.ndarray:
    """
    troca NaN/-inf por CF_PISO_DBZ e limita tudo por baixo a CF_PISO_DBZ.
    """
    lnp_sem_invalidos = np.nan_to_num(fnp_campo_dbz, nan=CF_PISO_DBZ, neginf=CF_PISO_DBZ)

    return np.maximum(lnp_sem_invalidos, CF_PISO_DBZ)

# -------------------------------------------------------------------------------------
def _maximo_ou_none(fnp_valores: np.ndarray) -> typing.Optional[float]:
    """
    maior valor do array, ou None se ele estiver vazio.
    """
    if 0 == fnp_valores.size:
        return None

    return float(fnp_valores.max())

# -------------------------------------------------------------------------------------
def calcula_metricas_categoricas(fnp_previsto_dbz: np.ndarray, fnp_observado_dbz: np.ndarray,
                                 ff_limiar_dbz: float) -> dict:
    """
    tabela de contingência e POD/FAR/CSI para um limiar.

    :param fnp_previsto_dbz: pixels previstos (1D, só os cobertos, com piso)
    :param fnp_observado_dbz: pixels observados (1D, mesmos pixels)
    :param ff_limiar_dbz: limiar do evento

    :returns: {"hits", "misses", "false_alarms", "pod", "far", "csi"}
    """
    lnp_evento_observado = fnp_observado_dbz >= ff_limiar_dbz
    lnp_evento_previsto = fnp_previsto_dbz >= ff_limiar_dbz

    li_acertos = int(np.sum(lnp_evento_observado & lnp_evento_previsto))
    li_perdas = int(np.sum(lnp_evento_observado & ~lnp_evento_previsto))
    li_falsos_alarmes = int(np.sum(~lnp_evento_observado & lnp_evento_previsto))

    return {
        "hits": li_acertos,
        "misses": li_perdas,
        "false_alarms": li_falsos_alarmes,
        "pod": _razao_ou_none(li_acertos, li_acertos + li_perdas),
        "far": _razao_ou_none(li_falsos_alarmes, li_acertos + li_falsos_alarmes),
        "csi": _razao_ou_none(li_acertos, li_acertos + li_perdas + li_falsos_alarmes),
        "freq_bias": _razao_ou_none(li_acertos + li_falsos_alarmes, li_acertos + li_perdas),
    }

# -------------------------------------------------------------------------------------
def calcula_metricas_continuas(fnp_previsto_dbz: np.ndarray,
                               fnp_observado_dbz: np.ndarray) -> dict:
    """
    MAE e bias (dBZ) só nos pixels com eco em algum dos dois campos -- senão
    o céu limpo (a maioria dos pixels) dilui tudo para ~0.

    abaixo do limiar de eco o valor conta como o próprio limiar, para que um
    eco perdido custe no máximo (eco - CF_ECHO_DBZ).

    :param fnp_previsto_dbz: pixels previstos (1D, só os cobertos, com piso)
    :param fnp_observado_dbz: pixels observados (1D, mesmos pixels)

    :returns: {"pixels_eco", "mae_dbz", "bias_dbz"} (None se não há eco)
    """
    lnp_tem_eco = (fnp_observado_dbz >= CF_ECHO_DBZ) | (fnp_previsto_dbz >= CF_ECHO_DBZ)
    li_pixels_eco = int(lnp_tem_eco.sum())

    if 0 == li_pixels_eco:
        return {"pixels_eco": 0, "mae_dbz": None, "bias_dbz": None}

    lnp_previsto_limitado = np.maximum(fnp_previsto_dbz[lnp_tem_eco], CF_ECHO_DBZ)
    lnp_observado_limitado = np.maximum(fnp_observado_dbz[lnp_tem_eco], CF_ECHO_DBZ)

    lnp_erro_dbz = lnp_previsto_limitado - lnp_observado_limitado

    return {
        "pixels_eco": li_pixels_eco,
        "mae_dbz": float(np.mean(np.abs(lnp_erro_dbz))),
        "bias_dbz": float(np.mean(lnp_erro_dbz)),
    }

# -------------------------------------------------------------------------------------
def _fracoes_na_janela(fnp_evento: np.ndarray, fnp_coberto: np.ndarray,
                       fi_escala_px: int) -> np.ndarray:
    """
    fração de pixels-evento em cada janela fi_escala_px x fi_escala_px, contando só os
    pixels cobertos (média móvel do evento / média móvel da cobertura).

    :param fnp_evento: campo binário 2D (float, 0 fora da cobertura)
    :param fnp_coberto: máscara 2D da cobertura (float 0/1)
    :param fi_escala_px: lado da janela em pixels

    :returns: campo 2D de frações (0 onde a janela não tem pixel coberto)
    """
    lnp_soma_evento = ndimage.uniform_filter(fnp_evento, size=fi_escala_px, mode="constant")
    lnp_soma_coberto = ndimage.uniform_filter(fnp_coberto, size=fi_escala_px, mode="constant")

    lnp_fracoes = np.zeros_like(fnp_evento)
    lnp_tem_cobertura = lnp_soma_coberto > 1e-9
    lnp_fracoes[lnp_tem_cobertura] = (lnp_soma_evento[lnp_tem_cobertura] /
                                      lnp_soma_coberto[lnp_tem_cobertura])

    return lnp_fracoes

# -------------------------------------------------------------------------------------
def calcula_somas_fss(fnp_previsto_dbz: np.ndarray, fnp_observado_dbz: np.ndarray,
                      ff_limiar_dbz: float, fi_escala_px: int) -> dict:
    """
    termos do FSS de um limiar em uma escala, SOMADOS sobre os pixels cobertos no
    observado (não-NaN):

        erro       = soma((Pp - Po)²)
        referencia = soma(Pp²) + soma(Po²)

    somados em várias janelas, dão o FSS agregado (Roberts & Lean 2008; pysteps):
    1 - soma(erro) / soma(referencia).

    :param fnp_previsto_dbz: campo previsto 2D (dBZ)
    :param fnp_observado_dbz: campo observado 2D (dBZ, NaN fora da cobertura)
    :param ff_limiar_dbz: limiar do evento
    :param fi_escala_px: lado da janela em pixels

    :returns: {"erro", "referencia"} (referencia 0 se nenhum dos dois campos tem evento)
    """
    lnp_coberto = np.isfinite(fnp_observado_dbz)
    lnp_coberto_float = lnp_coberto.astype(np.float64)

    lnp_previsto = _aplica_piso(fnp_previsto_dbz)
    lnp_observado = _aplica_piso(fnp_observado_dbz)

    lnp_evento_previsto = ((lnp_previsto >= ff_limiar_dbz) & lnp_coberto).astype(np.float64)
    lnp_evento_observado = ((lnp_observado >= ff_limiar_dbz) & lnp_coberto).astype(np.float64)

    lnp_fracao_prev = _fracoes_na_janela(lnp_evento_previsto, lnp_coberto_float, fi_escala_px)
    lnp_fracao_obs = _fracoes_na_janela(lnp_evento_observado, lnp_coberto_float, fi_escala_px)

    lnp_fracao_prev = lnp_fracao_prev[lnp_coberto]
    lnp_fracao_obs = lnp_fracao_obs[lnp_coberto]

    return {"erro": float(np.sum((lnp_fracao_prev - lnp_fracao_obs) ** 2)),
            "referencia": float(np.sum(lnp_fracao_prev ** 2) + np.sum(lnp_fracao_obs ** 2))}

# -------------------------------------------------------------------------------------
def fss_das_somas(ff_erro: float, ff_referencia: float) -> typing.Optional[float]:
    """
    FSS = 1 - erro / referencia, ou None se referencia <= 0 (sem evento: indefinido).
    """
    if ff_referencia <= 0.0:
        return None

    return 1.0 - ff_erro / ff_referencia

# -------------------------------------------------------------------------------------
def calcula_fss(fnp_previsto_dbz: np.ndarray, fnp_observado_dbz: np.ndarray,
                ff_limiar_dbz: float, fi_escala_px: int) -> typing.Optional[float]:
    """
    Fractions Skill Score de um limiar em uma escala.

        FSS = 1 - MSE(Pp, Po) / (média(Pp²) + média(Po²))

    sobre os pixels cobertos no observado (não-NaN). 1 = perfeito, 0 = sem habilidade.

    :param fnp_previsto_dbz: campo previsto 2D (dBZ)
    :param fnp_observado_dbz: campo observado 2D (dBZ, NaN fora da cobertura)
    :param ff_limiar_dbz: limiar do evento
    :param fi_escala_px: lado da janela em pixels

    :returns: FSS, ou None se nenhum dos dois campos tem evento (indefinido)
    """
    ldct_somas = calcula_somas_fss(fnp_previsto_dbz, fnp_observado_dbz, ff_limiar_dbz,
                                   fi_escala_px)

    return fss_das_somas(ldct_somas["erro"], ldct_somas["referencia"])

# -------------------------------------------------------------------------------------
def calcula_somas_fss_todas(fnp_previsto_dbz: np.ndarray, fnp_observado_dbz: np.ndarray,
                            ft_limiares_dbz: tuple=CT_LIMIARES_FSS_DBZ,
                            ft_escalas_km: tuple=CT_ESCALAS_FSS_KM) -> dict:
    """
    termos do FSS (calcula_somas_fss) de cada limiar em cada escala.

    :returns: {"30": {"1": {"erro", "referencia"}, "5": {...}, ...}, "36": {...}}
    """
    ldct_somas = {}

    for lf_limiar_dbz in ft_limiares_dbz:
        ldct_por_escala = {}

        for li_escala_km in ft_escalas_km:
            ldct_por_escala[str(li_escala_km)] = calcula_somas_fss(
                fnp_previsto_dbz, fnp_observado_dbz, lf_limiar_dbz, li_escala_km)

        ldct_somas[f"{lf_limiar_dbz:g}"] = ldct_por_escala

    return ldct_somas

# -------------------------------------------------------------------------------------
def fss_das_somas_todas(fdct_somas: dict) -> dict:
    """
    FSS de cada limiar em cada escala a partir dos termos (calcula_somas_fss_todas).

    :returns: {"30": {"1": fss, "5": fss, ...}, "36": {...}}
    """
    ldct_fss = {}

    for ls_limiar, ldct_escalas in fdct_somas.items():
        ldct_fss[ls_limiar] = {ls_escala: fss_das_somas(ldct["erro"], ldct["referencia"])
                               for ls_escala, ldct in ldct_escalas.items()}

    return ldct_fss

# -------------------------------------------------------------------------------------
def calcula_fss_todos(fnp_previsto_dbz: np.ndarray, fnp_observado_dbz: np.ndarray,
                      ft_limiares_dbz: tuple=CT_LIMIARES_FSS_DBZ,
                      ft_escalas_km: tuple=CT_ESCALAS_FSS_KM) -> dict:
    """
    FSS de cada limiar em cada escala.

    :returns: {"30": {"1": fss, "5": fss, ...}, "36": {...}}
    """
    return fss_das_somas_todas(calcula_somas_fss_todas(fnp_previsto_dbz, fnp_observado_dbz,
                                                       ft_limiares_dbz, ft_escalas_km))

# -------------------------------------------------------------------------------------
def calcula_metricas(fnp_previsto_dbz: np.ndarray, fnp_observado_dbz: np.ndarray,
                     ft_limiares_dbz: tuple=CT_LIMIARES_DBZ) -> dict:
    """
    compara um campo previsto com o observado (ambos em dBZ, 2D, mesma grade).

    pixels NaN no observado (fora da cobertura) são ignorados.

    :param fnp_previsto_dbz: campo previsto
    :param fnp_observado_dbz: campo observado (target)
    :param ft_limiares_dbz: limiares das métricas categóricas

    :returns: {"pixels", "max_obs", "max_prev", "limiares": {"20": {...}, ...},
               "pixels_eco", "mae_dbz", "bias_dbz", "fss": {...}, "fss_somas": {...}}
    """
    # só os pixels cobertos pelo radar no observado (vetor 1D)
    lnp_pixel_coberto = np.isfinite(fnp_observado_dbz)

    lnp_previsto = _aplica_piso(fnp_previsto_dbz[lnp_pixel_coberto])
    lnp_observado = np.maximum(fnp_observado_dbz[lnp_pixel_coberto], CF_PISO_DBZ)

    ldct_metricas = {
        "pixels": int(lnp_pixel_coberto.sum()),
        "max_obs": _maximo_ou_none(lnp_observado),
        "max_prev": _maximo_ou_none(lnp_previsto),
        "limiares": {},
    }

    for lf_limiar_dbz in ft_limiares_dbz:
        # chave sem casas decimais desnecessárias: 20.0 -> "20"
        ls_chave_limiar = f"{lf_limiar_dbz:g}"

        ldct_metricas["limiares"][ls_chave_limiar] = calcula_metricas_categoricas(
            lnp_previsto, lnp_observado, lf_limiar_dbz)

    ldct_metricas.update(calcula_metricas_continuas(lnp_previsto, lnp_observado))

    # FSS nos campos 2D (a vizinhança precisa da grade); os termos somados vão junto,
    # para a agregação de várias janelas (unt_avaliacao.AcumuladorLead)
    ldct_metricas["fss_somas"] = calcula_somas_fss_todas(fnp_previsto_dbz, fnp_observado_dbz)
    ldct_metricas["fss"] = fss_das_somas_todas(ldct_metricas["fss_somas"])

    return ldct_metricas

# -------------------------------------------------------------------------------------
def verifica_previsoes(fdct_previsoes: dict, fnp_ultima_entrada_dbz: np.ndarray,
                       fnp_observado_dbz: np.ndarray, fi_passo: int) -> dict:
    """
    avalia cada backend no passo fi_passo contra o observado, mais a
    persistência (o último CAPPI de entrada usado como "previsão").

    :param fdct_previsoes: nome do backend -> previsão (passos, linhas,
                           colunas) em dBZ
    :param fnp_ultima_entrada_dbz: último CAPPI de entrada (persistência)
    :param fnp_observado_dbz: CAPPI observado do target
    :param fi_passo: passo a avaliar (1 = primeiro passo previsto)

    :returns: {"lead": fi_passo, "backends": {nome: métricas},
               "persistencia": métricas}
    """
    ldct_resultado = {"lead": fi_passo, "backends": {}}

    for ls_nome_backend, lnp_previsao_dbz in fdct_previsoes.items():
        li_num_passos_previstos = lnp_previsao_dbz.shape[0]

        if fi_passo > li_num_passos_previstos:
            M_LOG.warning(f"[!] {ls_nome_backend}: passo {fi_passo} além do horizonte "
                          f"({li_num_passos_previstos} passos); não verificado.")
            continue

        # o passo N está na posição N-1 do array
        ldct_resultado["backends"][ls_nome_backend] = calcula_metricas(
            lnp_previsao_dbz[fi_passo - 1], fnp_observado_dbz)

    ldct_resultado["persistencia"] = calcula_metricas(fnp_ultima_entrada_dbz, fnp_observado_dbz)

    return ldct_resultado

# -------------------------------------------------------------------------------------
def verifica_lead(fnp_previsao_dbz: np.ndarray, fnp_observado_dbz: np.ndarray,
                  fi_passo: int) -> typing.Optional[dict]:
    """
    métricas de um passo de uma previsão (avaliação multi-lead do untEval).

    :param fnp_previsao_dbz: previsão (passos, linhas, colunas) em dBZ
    :param fnp_observado_dbz: CAPPI observado do horário do passo
    :param fi_passo: passo a avaliar (1 = primeiro passo previsto)

    :returns: métricas (calcula_metricas), ou None se o passo está além do horizonte
    """
    if fi_passo > fnp_previsao_dbz.shape[0]:
        return None

    return calcula_metricas(fnp_previsao_dbz[fi_passo - 1], fnp_observado_dbz)

# -------------------------------------------------------------------------------------
def _formata_valor(fx_valor: typing.Optional[float], fs_formato: str) -> str:
    """
    formata um número, ou "-" se ele for None (métrica indefinida).
    """
    if fx_valor is None:
        return "-"

    return format(fx_valor, fs_formato)

# -------------------------------------------------------------------------------------
def _fss_de_resumo(fdct_metricas: dict) -> typing.Optional[float]:
    """
    o FSS mostrado no resumo do log: limiar 30 dBZ, escala 10 km (None se ausente).
    """
    ldct_fss = fdct_metricas.get("fss", {})

    return ldct_fss.get("30", {}).get("10")

# -------------------------------------------------------------------------------------
def formata_resumo(fdct_resultado: dict) -> list:
    """
    monta uma linha legível por backend (e a persistência) para o log, ex.:

        rainnet       +1p CSI20=0.61 CSI30=0.48 CSI36=0.35 MAE=4.2dBZ FSS30@10=0.72

    :param fdct_resultado: saída de verifica_previsoes

    :returns: lista de linhas
    """
    # backends primeiro, persistência por último
    llst_avaliados = list(fdct_resultado["backends"].items())
    llst_avaliados.append(("persistencia", fdct_resultado["persistencia"]))

    llst_linhas = []

    for ls_nome, ldct_metricas in llst_avaliados:
        llst_textos_csi = []

        for ls_limiar, ldct_limiar in ldct_metricas["limiares"].items():
            llst_textos_csi.append(f"CSI{ls_limiar}={_formata_valor(ldct_limiar['csi'], '.2f')}")

        ls_textos_csi = " ".join(llst_textos_csi)
        ls_mae = _formata_valor(ldct_metricas["mae_dbz"], ".1f")
        ls_fss = _formata_valor(_fss_de_resumo(ldct_metricas), ".2f")

        llst_linhas.append(f"{ls_nome:<13} +{fdct_resultado['lead']}p {ls_textos_csi} "
                           f"MAE={ls_mae}dBZ FSS30@10={ls_fss}")

    return llst_linhas

# -------------------------------------------------------------------------------------
def grava_json(fdct_resultado: dict, fpth_json: typing.Union[str, pathlib.Path]) -> None:
    """
    grava o resultado da verificação em JSON (indentado, UTF-8 legível).

    :param fdct_resultado: saída de verifica_previsoes (+ dados da execução)
    :param fpth_json: caminho do arquivo
    """
    ls_json = json.dumps(fdct_resultado, indent=2, ensure_ascii=False)

    pathlib.Path(fpth_json).write_text(ls_json)

# < the end >----------------------------------------------------------------------------
