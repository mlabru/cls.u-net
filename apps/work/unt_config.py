# -*- coding: utf-8 -*-
"""
Centro Logístico de Simulação Meteorológica
Copyright (c) 1995-2026, Instituto de Controle do Espaço Aéreo

unt_config

configuração de "como" o algoritmo roda (janelas, treino, cada backend), lida do
config.ini. Os caminhos ("onde": catálogo, CAPPIs, modelos) NÃO moram aqui: vêm da
CLI e das variáveis UNT_* do unt_cfg.sh, como no 25.oflow/27.nowcast.

seções:
  [janelas]  montagem das janelas de treino (untJanelas);
  [treino]   hiperparâmetros comuns (untTrain);
  [pytorch] [rainnet] [smaat]  sobrescrevem chaves do [treino] para aquele backend
             e trazem as chaves próprias da arquitetura;
  [promocao] portão do untPromove (candidato x atual x persistência);
  [avaliacao] bootstrap por bloco do untEval (intervalos de confiança);
  [monitor]  alarme do untMonitor (skill em operação x persistência).

sem config.ini (ou sem uma seção/opção), valem os defaults abaixo -- os testes e o
--help funcionam sem arquivo. Precedência: CLI > config.ini > default.

2026.out  mlabru  [treino] passos_colapso, tentativas_colapso
2026.out  mlabru  [treino] media_metrica, limites_peso_dbz, peso_dbz (perda de regressão
                  ponderada pela intensidade do alvo); lista_reais vazia = []
2026.out  mlabru  [treino] max_nao_finitos
2026.out  mlabru  [treino] min_pixels_evento_val, passos_rollout
2026.out  mlabru  seção [avaliacao] (bootstrap); [promocao] significancia, min_blocos
2026.out  mlabru  [promocao] min_janelas_par (alertas por par do untPromove)
2026.out  mlabru  seção [monitor] (untMonitor)
2026.out  mlabru  [treino] selecao, val_cada, max_janelas_val, clip_grad; bins com 36 dBZ;
                  peso_bins = auto
2026.out  mlabru  seção [promocao] (untPromove)
2026.out  mlabru  initial version (Linux/Python)
"""
# < imports >----------------------------------------------------------------------------

# python library
import configparser
import logging
import pathlib
import typing

# < constants >--------------------------------------------------------------------------

# arquivo padrão (relativo ao diretório corrente, como no 27.nowcast)
CS_ARQUIVO_CONFIG = "config.ini"

# defaults de cada seção (valores como texto, convertidos por quem lê)
MDCT_DEFAULTS = {
    "janelas": {
        # cadência nominal e tolerância da associação encadeada (s)
        "intervalo_s": "600",
        "tolerancia_s": "300",
        # entradas e alvos de cada janela (+10..+60 min)
        "num_entradas": "4",
        "num_alvos": "6",
        # filtros (mesmo critério do nwcSequences)
        "min_valid_ratio": "0.95",
        "min_rain_fraction": "0.0",
        # split temporal por dia UTC (o resto é teste)
        "frac_treino": "0.70",
        "frac_validacao": "0.15",
        # bloco do split (s): um dia UTC; menor só para testes com pouco dado
        "bloco_split_s": "86400",
    },
    "treino": {
        "device": "auto",
        "epocas": "50",
        "batch": "8",
        "lr": "3e-4",
        "paciencia": "8",
        # lado do recorte aleatório (px); 0 = grade inteira
        "recorte_px": "256",
        # 64 = largura dos artigos; 16/32 para treino em CPU
        "largura_base": "64",
        "workers": "2",
        "semente": "42",
        "amp": "true",
        "augment": "true",
        # recorte aceito se ao menos esta fração de pixels tem eco (>= 10 dBZ)
        "min_eco_recorte": "0.01",
        # tentativas de achar um recorte com eco antes de aceitar qualquer um
        "tentativas_recorte": "10",
        # janelas de treino por época (0 = todas); o mesmo número de ITENS em todos os
        # backends -- no pytorch (um item por janela x lead) é sorteado um lead por item
        "amostras_por_epoca": "0",
        # escolha da melhor época e early stopping: "metrica" (CSI/FSS do rollout
        # completo nas janelas de validação, limiar/escala do [promocao]) ou "perda"
        "selecao": "metrica",
        # a métrica é calculada a cada val_cada épocas, em no máximo max_janelas_val
        # janelas de validação igualmente espaçadas (0 = todas)
        "val_cada": "1",
        "max_janelas_val": "32",
        # só janelas de validação com evento entram na métrica: ao menos este número de
        # pixels >= limiar do [promocao] em algum alvo (0 = sem filtro). Sem evento, a
        # janela não muda o CSI/FSS e só tira janelas que importam da amostra
        "min_pixels_evento_val": "0",
        # norma máxima do gradiente (0 = sem corte)
        "clip_grad": "1.0",
        # fração máxima de lotes com perda não finita (NaN/inf) numa época, no treino ou
        # na validação; acima dela o treino para com erro (1 = nunca para)
        "max_nao_finitos": "0.01",
        # rainnet/smaat: saída da rede constante (desvio ~0) depois destes passos da 1ª
        # época = rede colapsada (ReLUs mortas); o untTrain recomeça com a semente
        # seguinte, até tentativas_colapso vezes (0 passos = sem verificação). 2026-10:
        # o rainnet colapsou com 3 de 9 sementes nos primeiros 50 passos
        "passos_colapso": "200",
        "tentativas_colapso": "5",
        # modo recursivo: passos t+1..t+K realimentados também no treino, perda média
        # nos K passos (1 = só t+1, como nos artigos). Com 1, a rede nunca vê a própria
        # saída como entrada e o erro se acumula na previsão (rainnet atenua, smaat infla)
        "passos_rollout": "1",
        # seleção pela métrica: média móvel das últimas N métricas de validação (1 = a
        # métrica da época, sem média). Com a métrica ruidosa entre épocas vizinhas, o
        # early stopping escolhia um pico isolado (2026-10: rainnet 0,18 -> 0,39 -> 0,25)
        "media_metrica": "1",
        # rainnet/smaat: perda de regressão ponderada pela intensidade do ALVO. Limites
        # em dBZ (crescentes) e um peso por faixa (len(limites) + 1 valores); vazio =
        # sem peso, como nos artigos
        "limites_peso_dbz": "",
        "peso_dbz": "",
    },
    "pytorch": {
        "lr": "1e-3",
        # limites (dBZ) dos bins de intensidade da entropia cruzada; incluem os limiares
        # da verificação (20, 30, 36), para que o centro do bin não cruze um limiar
        "bins_dbz": "10,20,25,30,36,40,45,50,55",
        # peso de cada classe (len(bins_dbz) + 1 valores), ou "auto": raiz do inverso
        # da frequência da classe nos alvos de treino (calculado pelo untTrain)
        "peso_bins": "1,1,2,2,3,4,5,6,8,10",
        "blocos_por_nivel": "2",
    },
    "rainnet": {
        "lr": "3e-4",
    },
    "smaat": {
        "lr": "1e-3",
        "kernels_per_layer": "2",
        "reduction_ratio": "16",
        # recursivo (t+1, como o rainnet) ou direto (6 saídas)
        "modo": "recursivo",
    },
    "promocao": {
        # passos (1 = +10 min) em que o candidato é julgado
        "passos": "3,6",
        # limiar (dBZ) do CSI e do FSS, e escala do FSS (km); precisam existir no
        # comparacao.json (unt_verify.CT_LIMIARES_DBZ / CT_ESCALAS_FSS_KM)
        "limiar_dbz": "30",
        "escala_fss_km": "20",
        # quanto o candidato pode ficar ABAIXO do atual (CSI/FSS absolutos)
        "tolerancia": "0.01",
        # quanto o candidato precisa ficar ACIMA da persistência
        "margem_persistencia": "0.0",
        # conjunto do manifesto exigido na avaliação
        "split": "teste",
        # o mesmo critério é aplicado a cada par radar/alcance, só como alerta (não
        # bloqueia); pares com menos janelas avaliadas que isto não são julgados
        "min_janelas_par": "20",
        # exige também que o intervalo de confiança (bootstrap por bloco, [avaliacao])
        # da diferença candidato - persistência fique acima da margem, em cada passo e
        # métrica; precisa de somas_por_bloco no comparacao.json e de ao menos
        # min_blocos blocos (com poucos blocos o intervalo não significa nada)
        "significancia": "false",
        "min_blocos": "5",
    },
    "avaliacao": {
        # bootstrap por bloco do split (untEval, e o untPromove com significancia)
        "reamostras": "2000",
        # nível do intervalo percentil bilateral
        "nivel_confianca": "0.90",
        "semente": "42",
    },
    "monitor": {
        # janela móvel (h) das verificações em operação (verify.jsonl)
        "janela_h": "24",
        # mínimo de verificações COM evento (no limiar do [promocao]) para julgar
        "min_verificacoes": "12",
        # alarme quando a pontuação do backend fica abaixo da persistência - margem
        "margem": "0.0",
    },
}

# < logging >----------------------------------------------------------------------------

# logger
M_LOG = logging.getLogger(__name__)
M_LOG.setLevel(logging.INFO)

# < tipos >------------------------------------------------------------------------------

class SConfig:
    """
    configuração lida: acesso tipado por seção e opção.

    num backend, uma opção é procurada primeiro na seção do backend e depois no [treino]
    (ver opcao_do_backend).
    """
    def __init__(self, fparser: configparser.ConfigParser):
        self._parser = fparser

    # ---------------------------------------------------------------------------------
    def texto(self, fs_secao: str, fs_opcao: str) -> str:
        """
        valor bruto (texto) de uma opção.

        :raises KeyError: se a opção não existe nem no arquivo nem nos defaults
        """
        if self._parser.has_option(fs_secao, fs_opcao):
            return self._parser.get(fs_secao, fs_opcao).strip()

        return MDCT_DEFAULTS[fs_secao][fs_opcao]

    # ---------------------------------------------------------------------------------
    def inteiro(self, fs_secao: str, fs_opcao: str) -> int:
        return _converte(int, self.texto(fs_secao, fs_opcao), fs_secao, fs_opcao)

    # ---------------------------------------------------------------------------------
    def real(self, fs_secao: str, fs_opcao: str) -> float:
        return _converte(float, self.texto(fs_secao, fs_opcao), fs_secao, fs_opcao)

    # ---------------------------------------------------------------------------------
    def booleano(self, fs_secao: str, fs_opcao: str) -> bool:
        ls_valor = self.texto(fs_secao, fs_opcao).lower()

        if ls_valor in ("1", "true", "sim", "yes", "on"):
            return True

        if ls_valor in ("0", "false", "nao", "não", "no", "off"):
            return False

        raise ValueError(f"[{fs_secao}] {fs_opcao}: booleano inválido ({ls_valor})")

    # ---------------------------------------------------------------------------------
    def lista_reais(self, fs_secao: str, fs_opcao: str) -> list:
        """
        valores separados por vírgula; texto vazio = lista vazia.
        """
        llst_valores = []

        if not self.texto(fs_secao, fs_opcao):
            return llst_valores

        for ls_item in self.texto(fs_secao, fs_opcao).split(","):
            llst_valores.append(_converte(float, ls_item.strip(), fs_secao, fs_opcao))

        return llst_valores

    # ---------------------------------------------------------------------------------
    def secao_do_backend(self, fs_backend: str, fs_opcao: str) -> str:
        """
        seção de onde vem a opção de um backend: a do backend, se ela tem a opção (no
        arquivo ou nos defaults), senão o [treino].
        """
        lb_no_arquivo = self._parser.has_option(fs_backend, fs_opcao)
        lb_no_default = fs_opcao in MDCT_DEFAULTS.get(fs_backend, {})

        if lb_no_arquivo or lb_no_default:
            return fs_backend

        return "treino"

# < leitura >----------------------------------------------------------------------------

def chave_numerica(fs_valor: str) -> str:
    """
    número como chave dos dicionários de métricas: "30", "30.0" -> "30"; "2.5" -> "2.5".
    """
    lf_valor = float(fs_valor)

    if lf_valor.is_integer():
        return str(int(lf_valor))

    return str(lf_valor)

# -------------------------------------------------------------------------------------

def _converte(ff_tipo, fs_valor: str, fs_secao: str, fs_opcao: str):
    """
    converte um valor, com mensagem que cita a seção e a opção.

    :raises ValueError: valor inválido para o tipo
    """
    try:
        return ff_tipo(fs_valor)

    except ValueError:
        raise ValueError(f"[{fs_secao}] {fs_opcao}: valor inválido ({fs_valor})") from None

# -------------------------------------------------------------------------------------
def carrega(fs_arquivo: typing.Optional[str]=None) -> SConfig:
    """
    lê o config.ini (se existir) e devolve a configuração.

    :param fs_arquivo: caminho do arquivo (None = ./config.ini)

    :returns: SConfig (só com os defaults se o arquivo não existe)
    """
    lpth_arquivo = pathlib.Path(fs_arquivo or CS_ARQUIVO_CONFIG)
    lparser = configparser.ConfigParser()

    if lpth_arquivo.is_file():
        lparser.read(lpth_arquivo, encoding="utf-8")

    elif fs_arquivo is not None:
        raise FileNotFoundError(f"config não encontrado: {lpth_arquivo}")

    return SConfig(lparser)

# < the end >----------------------------------------------------------------------------
