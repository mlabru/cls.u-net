# -*- coding: utf-8 -*-
"""
Centro Logístico de Simulação Meteorológica
Copyright (c) 1995-2026, Instituto de Controle do Espaço Aéreo

unt_avaliacao

agregação das métricas de verificação (unt_verify) ao longo de várias janelas e passos,
comum à avaliação comparativa (untEval) e à seleção de época no treino (unt_treino):
o treino escolhe o modelo pelo MESMO critério com que ele é julgado depois.

agregação por (rótulo, passo):
  - categóricas: acertos/perdas/falsos SOMADOS em todas as janelas, e só então
    POD/FAR/CSI/bias (o CSI médio por janela exagera janelas com pouco eco);
  - FSS: termos (erro e referência) SOMADOS em todas as janelas, e só então
    1 - erro / referência (Roberts & Lean 2008; pysteps). A média por janela exagera
    janelas com pouco eco (FSS ~0); ela vai junto, só como referência, em
    "fss_media_janelas". Verificações sem "fss_somas" (verify.jsonl antigos) caem na
    média por janela;
  - MAE/bias: média ponderada pelo número de pixels com eco.

pontuacao(): um escalar para a seleção de época -- média do CSI e do FSS (limiar e
escala do [promocao], os do portão do untPromove) nos passos em que são definidos.

2026.out  mlabru  janelas_com_evento (amostra da métrica de validação)
2026.out  mlabru  somas() por acumulador e por conjunto (bootstrap por bloco)
2026.out  mlabru  FSS agregado pelas somas dos termos (fss_somas), não pela média das
                  janelas
2026.out  mlabru  acumula soma em vários conjuntos de acumuladores (agregado e por par
                  radar/alcance do untEval) com uma só verificação por passo
2026.out  mlabru  initial version (Linux/Python), extraído do untEval
"""
# < imports >----------------------------------------------------------------------------

# python library
import logging
import typing

# numPy
import numpy as np

# model
import work.unt_dataset as uds
import work.unt_inferencia as ui
import work.unt_verify as ov

# < logging >----------------------------------------------------------------------------

# logger
M_LOG = logging.getLogger(__name__)
M_LOG.setLevel(logging.INFO)

# < acumuladores >-----------------------------------------------------------------------

def _razao(ff_numerador: float, ff_denominador: float) -> typing.Optional[float]:
    """
    divisão que devolve None com denominador zero.
    """
    if 0 == ff_denominador:
        return None

    return float(ff_numerador) / float(ff_denominador)

# -------------------------------------------------------------------------------------
class AcumuladorLead:
    """
    soma as métricas de um (rótulo, passo) ao longo das janelas.
    """
    def __init__(self):
        self.janelas = 0
        self.categoricas = {}
        self.fss = {}
        self.fss_somas = {}
        self.fss_sem_somas = 0
        self.soma_erro_abs = 0.0
        self.soma_erro = 0.0
        self.pixels_eco = 0

    def adiciona(self, fdct_metricas: dict) -> None:
        """
        :param fdct_metricas: saída de unt_verify.calcula_metricas
        """
        self.janelas += 1

        for ls_limiar, ldct in fdct_metricas["limiares"].items():
            ldct_soma = self.categoricas.setdefault(ls_limiar, {"hits": 0, "misses": 0,
                                                                "false_alarms": 0})
            for ls_chave in ldct_soma:
                ldct_soma[ls_chave] += ldct[ls_chave]

        for ls_limiar, ldct_escalas in fdct_metricas.get("fss", {}).items():
            for ls_escala, lf_fss in ldct_escalas.items():
                if lf_fss is None:
                    continue
                self.fss.setdefault(ls_limiar, {}).setdefault(ls_escala, []).append(lf_fss)

        self._adiciona_somas_fss(fdct_metricas)

        li_pixels_eco = fdct_metricas["pixels_eco"]

        if li_pixels_eco and fdct_metricas["mae_dbz"] is not None:
            self.soma_erro_abs += fdct_metricas["mae_dbz"] * li_pixels_eco
            self.soma_erro += fdct_metricas["bias_dbz"] * li_pixels_eco
            self.pixels_eco += li_pixels_eco

    def _adiciona_somas_fss(self, fdct_metricas: dict) -> None:
        """
        soma os termos do FSS; conta a verificação sem eles (formato antigo).
        """
        if "fss_somas" not in fdct_metricas:
            self.fss_sem_somas += int(bool(fdct_metricas.get("fss")))
            return

        for ls_limiar, ldct_escalas in fdct_metricas["fss_somas"].items():
            for ls_escala, ldct_termos in ldct_escalas.items():
                ldct_soma = self.fss_somas.setdefault(ls_limiar, {}).setdefault(
                    ls_escala, {"erro": 0.0, "referencia": 0.0})
                ldct_soma["erro"] += ldct_termos["erro"]
                ldct_soma["referencia"] += ldct_termos["referencia"]

    def somas(self) -> dict:
        """
        só as somas (contingências e termos do FSS), sem as razões: o que o bootstrap
        por bloco (unt_bootstrap) precisa para reagregar blocos sorteados.
        """
        return {"janelas": self.janelas,
                "limiares": {ls_limiar: dict(ldct)
                             for ls_limiar, ldct in self.categoricas.items()},
                "fss_somas": {ls_limiar: {ls_escala: dict(ldct)
                                          for ls_escala, ldct in ldct_escalas.items()}
                              for ls_limiar, ldct_escalas in self.fss_somas.items()}}

    def _fss_agregado(self, fs_limiar: str, fs_escala: str) -> typing.Optional[float]:
        """
        FSS das somas dos termos; média das janelas se alguma veio sem os termos.
        """
        llst_por_janela = self.fss.get(fs_limiar, {}).get(fs_escala, [])
        ldct_soma = self.fss_somas.get(fs_limiar, {}).get(fs_escala)

        if self.fss_sem_somas or ldct_soma is None:
            return float(np.mean(llst_por_janela)) if llst_por_janela else None

        return ov.fss_das_somas(ldct_soma["erro"], ldct_soma["referencia"])

    def resumo(self) -> dict:
        """
        métricas agregadas (None quando indefinidas).
        """
        ldct_limiares = {}

        for ls_limiar, ldct in self.categoricas.items():
            li_h, li_m, li_f = ldct["hits"], ldct["misses"], ldct["false_alarms"]

            ldct_limiares[ls_limiar] = dict(ldct)
            ldct_limiares[ls_limiar].update({
                "pod": _razao(li_h, li_h + li_m),
                "far": _razao(li_f, li_h + li_f),
                "csi": _razao(li_h, li_h + li_m + li_f),
                "freq_bias": _razao(li_h + li_f, li_h + li_m)})

        ldct_fss = {}
        ldct_fss_media = {}
        for ls_limiar in sorted(set(self.fss) | set(self.fss_somas)):
            lset_escalas = (set(self.fss.get(ls_limiar, {})) |
                            set(self.fss_somas.get(ls_limiar, {})))
            ldct_fss[ls_limiar] = {ls_escala: self._fss_agregado(ls_limiar, ls_escala)
                                   for ls_escala in sorted(lset_escalas, key=int)}

        for ls_limiar, ldct_escalas in self.fss.items():
            ldct_fss_media[ls_limiar] = {ls_escala: float(np.mean(llst))
                                         for ls_escala, llst in ldct_escalas.items()}

        return {"janelas": self.janelas,
                "limiares": ldct_limiares,
                "fss": ldct_fss,
                "fss_media_janelas": ldct_fss_media,
                "pixels_eco": self.pixels_eco,
                "mae_dbz": _razao(self.soma_erro_abs, self.pixels_eco),
                "bias_dbz": _razao(self.soma_erro, self.pixels_eco)}

# -------------------------------------------------------------------------------------
def acumula(fdct_previsoes: dict, fnp_alvos: np.ndarray, *flst_acumuladores: dict) -> None:
    """
    compara cada passo de cada previsão com o alvo observado e soma nos acumuladores
    (cada métrica é calculada uma vez e somada em todos os conjuntos dados).

    :param fdct_previsoes: rótulo -> previsão (passos, H, W) em dBZ
    :param fnp_alvos: alvos observados (passos, H, W) em dBZ
    :param flst_acumuladores: um ou mais rótulo -> passo -> AcumuladorLead (atualizados
                              aqui), ex.: o agregado e o do par radar/alcance da janela
    """
    li_num_passos = fnp_alvos.shape[0]

    for ls_rotulo, lnp_previsao in fdct_previsoes.items():
        for li_passo in range(1, li_num_passos + 1):
            ldct_metricas = ov.verifica_lead(lnp_previsao, fnp_alvos[li_passo - 1], li_passo)

            if ldct_metricas is None:
                continue

            for ldct_acumuladores in flst_acumuladores:
                lacumulador = ldct_acumuladores.setdefault(ls_rotulo, {}).setdefault(
                    li_passo, AcumuladorLead())
                lacumulador.adiciona(ldct_metricas)

# -------------------------------------------------------------------------------------
def somas(fdct_acumuladores: dict) -> dict:
    """
    rótulo -> passo (texto) -> somas (AcumuladorLead.somas), para o bootstrap.
    """
    return {ls_rotulo: {str(li_passo): lacum.somas()
                        for li_passo, lacum in sorted(ldct_por_passo.items())}
            for ls_rotulo, ldct_por_passo in fdct_acumuladores.items()}

# -------------------------------------------------------------------------------------
def resume(fdct_acumuladores: dict) -> dict:
    """
    rótulo -> passo (texto) -> resumo, como no comparacao.json.
    """
    ldct_resultados = {}

    for ls_rotulo, ldct_por_passo in fdct_acumuladores.items():
        ldct_resultados[ls_rotulo] = {str(li_passo): lacum.resumo()
                                      for li_passo, lacum in sorted(ldct_por_passo.items())}

    return ldct_resultados

# < seleção de época >-------------------------------------------------------------------

def pontuacao(fdct_por_passo: dict, fs_limiar: str, fs_escala: str) -> typing.Optional[float]:
    """
    escalar para a seleção de época (maior = melhor): média de CSI e FSS (no limiar e
    escala dados) nos passos em que são definidos.

    :param fdct_por_passo: passo (texto) -> resumo, de um rótulo
    :param fs_limiar: limiar em dBZ, como chave ("30")
    :param fs_escala: escala do FSS em km, como chave ("20")

    :returns: pontuação, ou None se nenhuma métrica foi definida (nenhum evento)
    """
    llst_csi = []
    llst_fss = []

    for ldct_passo in fdct_por_passo.values():
        lf_csi = ldct_passo.get("limiares", {}).get(fs_limiar, {}).get("csi")
        lf_fss = ldct_passo.get("fss", {}).get(fs_limiar, {}).get(fs_escala)

        if lf_csi is not None:
            llst_csi.append(lf_csi)

        if lf_fss is not None:
            llst_fss.append(lf_fss)

    llst_medias = [float(np.mean(llst)) for llst in (llst_csi, llst_fss) if llst]

    if not llst_medias:
        return None

    return float(np.mean(llst_medias))

# -------------------------------------------------------------------------------------
def amostra_janelas(flst_janelas: list, fi_maximo: int) -> list:
    """
    no máximo fi_maximo janelas, igualmente espaçadas no tempo (determinístico).
    0 = todas.
    """
    if fi_maximo <= 0 or len(flst_janelas) <= fi_maximo:
        return list(flst_janelas)

    lnp_indices = np.linspace(0, len(flst_janelas) - 1, fi_maximo).round().astype(int)

    return [flst_janelas[li] for li in sorted(set(lnp_indices.tolist()))]

# -------------------------------------------------------------------------------------
def janelas_com_evento(flst_janelas: list, fs_cappi_dir: str, ff_limiar_dbz: float,
                       fi_min_pixels: int, fi_num_entradas: int) -> list:
    """
    só as janelas com ao menos fi_min_pixels pixels >= ff_limiar_dbz em algum alvo.
    """
    llst_com_evento = []

    for ljanela in flst_janelas:
        lnp_alvos = uds.carrega_quadros(fs_cappi_dir, ljanela)[fi_num_entradas:]

        with np.errstate(invalid="ignore"):
            lnp_pixels = (lnp_alvos >= ff_limiar_dbz).sum(axis=(-2, -1))

        if lnp_pixels.max() >= fi_min_pixels:
            llst_com_evento.append(ljanela)

    return llst_com_evento

# -------------------------------------------------------------------------------------
def avalia_modelo(fmodelo, flst_janelas: list, fs_cappi_dir: str) -> dict:
    """
    previsão completa (todos os passos, grade inteira, rollout) de um modelo em
    memória sobre janelas, agregada por passo.

    :param fmodelo: unt_modelos.SModeloCarregado (rede em modo eval)
    :param flst_janelas: unt_janelas.SJanela
    :param fs_cappi_dir: raiz dos CAPPIs

    :returns: passo (texto) -> resumo
    """
    ldct_acumuladores = {}

    for ljanela in flst_janelas:
        li_num_entradas = len(ljanela.entradas)

        lnp_quadros = uds.carrega_quadros(fs_cappi_dir, ljanela)
        lnp_alvos = lnp_quadros[li_num_entradas:]

        lnp_previsao = ui.preve_dbz(fmodelo, lnp_quadros[:li_num_entradas], lnp_alvos.shape[0])
        acumula({"modelo": lnp_previsao}, lnp_alvos, ldct_acumuladores)

    return resume(ldct_acumuladores).get("modelo", {})

# < the end >----------------------------------------------------------------------------
