# -*- coding: utf-8 -*-
"""
Centro Logístico de Simulação Meteorológica
Copyright (c) 1995-2026, Instituto de Controle do Espaço Aéreo

unt_smaat

backend "smaat" do nowcast por Deep Learning:
SmaAt-UNet (Trebing et al., 2021; smaat/): U-Net com convoluções separáveis em
profundidade e atenção CBAM sobre R normalizado, previsão recursiva (ou direta,
[smaat] modo).

o pré-processamento, a rede e a estratégia vêm do checkpoint ($UNT_MODELOS_DIR/smaat/
atual.pt, gravado pelo untTrain); o fluxo comum está em
unt_inferencia.gera_nowcast_do_backend.

2026.out  mlabru  initial version (Linux/Python)
"""
# < imports >----------------------------------------------------------------------------

# numPy
import numpy as np

# pyTorch (sem ele o untMain descarta este backend)
import torch  # noqa: F401

# model
import work.unt_backend_comum as obc
import work.unt_inferencia as ui

# < constants >--------------------------------------------------------------------------

# nome do backend (título dos frames, diretório de modelos, subdiretório de saída)
CS_NOME_BACKEND = "smaat"

# -------------------------------------------------------------------------------------
def gera_nowcast(fnp_refletividade_dbz: np.ndarray, ft_radar: obc.GeometriaRadar,
                 fi_num_passos: int=6, fi_passo_minutos: int=10,
                 fs_prefixo_saida: str="nowcast") -> obc.ResultadoNowcast:
    """
    gera o nowcast (contrato de unt_backend_comum).

    :param fnp_refletividade_dbz: (quadros, linhas, colunas) em dBZ, NaN fora da
                                  cobertura, do mais antigo ao mais novo
    :param ft_radar: geometria do radar (mapa)
    :param fi_num_passos: passos a prever
    :param fi_passo_minutos: minutos entre passos (título dos frames)
    :param fs_prefixo_saida: prefixo dos frames <prefixo>_NN.png

    :returns: ResultadoNowcast (previsões None se não há chuva)

    :raises ValueError: entrada sem nenhum valor finito
    :raises FileNotFoundError: backend ainda sem checkpoint treinado
    """
    return ui.gera_nowcast_do_backend(CS_NOME_BACKEND, fnp_refletividade_dbz, ft_radar,
                                      fi_num_passos, fi_passo_minutos, fs_prefixo_saida)

# < the end >----------------------------------------------------------------------------
