# -*- coding: utf-8 -*-
"""
Centro Logístico de Simulação Meteorológica
Copyright (c) 1995-2026, Instituto de Controle do Espaço Aéreo

unt_inferencia

previsão em dBZ, na grade inteira, a partir de um checkpoint -- o núcleo comum dos
três backends (untMain) e da avaliação comparativa (untEval). Cada estratégia de
previsão segue o artigo da rede:

  - recursivo (rainnet, smaat): prevê t+1, realimenta a previsão como a entrada mais
    recente e repete (RainNet README/RainNet2024);
  - direto (smaat opcional):    uma passada dá os N passos;
  - lead (pytorch):             uma passada por passo, com o lead como entrada.

a grade recebe padding refletido até múltiplo de 32 e é recortada de volta; os pixels
fora da cobertura da ÚLTIMA entrada saem NaN (a rede não tem como prever ali).

carrega_modelo_atual() guarda o checkpoint carregado por processo (os backends rodam
um por processo no untMain; o untEval reaproveita o mesmo modelo em todas as janelas).

2026.out  mlabru  initial version (Linux/Python)
"""
# < imports >----------------------------------------------------------------------------

# python library
import functools
import logging
import os
import typing

# numPy
import numpy as np

# pyTorch
import torch

# model
import work.unt_backend_comum as obc
import work.unt_dados as ud
import work.unt_device as ue
import work.unt_modelos as um

# < logging >----------------------------------------------------------------------------

# logger
M_LOG = logging.getLogger(__name__)
M_LOG.setLevel(logging.INFO)

# < carga do modelo >--------------------------------------------------------------------

@functools.lru_cache(maxsize=None)
def carrega_modelo_atual(fs_backend: str, fs_modelos_dir: str,
                         fs_device: str="auto") -> um.SModeloCarregado:
    """
    carrega <modelos>/<backend>/atual.pt uma vez por processo.

    :param fs_backend: nome do backend
    :param fs_modelos_dir: raiz dos modelos
    :param fs_device: "auto", "cpu", "cuda"...

    :returns: SModeloCarregado

    :raises FileNotFoundError: sem atual.pt (backend ainda não treinado)
    """
    ldevice = ue.escolhe_device(fs_device)
    lpth_checkpoint = um.caminho_checkpoint_atual(fs_modelos_dir, fs_backend)

    lmodelo = um.carrega_checkpoint(lpth_checkpoint, ldevice)

    if lmodelo.backend != fs_backend:
        raise ValueError(f"{lpth_checkpoint}: checkpoint do backend {lmodelo.backend}, "
                         f"não {fs_backend}")

    M_LOG.info(f"[+] {fs_backend}: {lpth_checkpoint.resolve().name} em {ue.descreve(ldevice)}")

    return lmodelo

# -------------------------------------------------------------------------------------
def device_do_modelo(fmodelo: um.SModeloCarregado) -> torch.device:
    """
    device onde estão os pesos da rede.
    """
    return next(fmodelo.rede.parameters()).device

# < estratégias >------------------------------------------------------------------------

def _executa_rede(fmodelo: um.SModeloCarregado, fnp_entrada: np.ndarray,
                  flead: typing.Optional[int]=None) -> np.ndarray:
    """
    uma passada da rede para UM exemplo.

    :param fnp_entrada: (canais, H, W) já transformado e com padding
    :param flead: índice do passo (só o backend pytorch)

    :returns: saída (canais, H, W) em numpy float32
    """
    ldevice = device_do_modelo(fmodelo)
    lx = torch.from_numpy(np.ascontiguousarray(fnp_entrada))[None].to(ldevice)

    with torch.inference_mode():
        if flead is None:
            lsaida = fmodelo.rede(lx)
        else:
            llead = torch.tensor([flead], dtype=torch.int64, device=ldevice)
            lsaida = torch.softmax(fmodelo.rede(lx, llead).float(), dim=1)

    return lsaida[0].float().cpu().numpy()

# -------------------------------------------------------------------------------------
def _preve_recursivo(fmodelo: um.SModeloCarregado, fnp_entradas_dbz: np.ndarray,
                     fi_num_passos: int) -> np.ndarray:
    """
    t+1 realimentado fi_num_passos vezes (na representação da rede, sem voltar a dBZ).

    :returns: (passos, H, W) em dBZ (com padding)
    """
    ltransf = fmodelo.transformacao
    lnp_janela = ltransf.entrada(fnp_entradas_dbz)

    llst_passos = []

    for _ in range(fi_num_passos):
        lnp_saida = _executa_rede(fmodelo, lnp_janela)[:1]
        llst_passos.append(ltransf.saida_para_dbz(lnp_saida[0]))

        lnp_janela = np.concatenate([lnp_janela[1:], ltransf.realimenta(lnp_saida)])

    return np.stack(llst_passos)

# -------------------------------------------------------------------------------------
def _preve_direto(fmodelo: um.SModeloCarregado, fnp_entradas_dbz: np.ndarray,
                  fi_num_passos: int) -> np.ndarray:
    """
    uma passada com N saídas (no máximo os passos treinados).

    :returns: (passos, H, W) em dBZ (com padding)
    """
    ltransf = fmodelo.transformacao
    lnp_saida = _executa_rede(fmodelo, ltransf.entrada(fnp_entradas_dbz))

    llst_passos = []
    for li_passo in range(min(fi_num_passos, lnp_saida.shape[0])):
        llst_passos.append(ltransf.saida_para_dbz(lnp_saida[li_passo]))

    return np.stack(llst_passos)

# -------------------------------------------------------------------------------------
def _preve_por_lead(fmodelo: um.SModeloCarregado, fnp_entradas_dbz: np.ndarray,
                    fi_num_passos: int) -> np.ndarray:
    """
    uma passada por passo, com o lead como entrada (no máximo os leads treinados).

    :returns: (passos, H, W) em dBZ (com padding)
    """
    ltransf = fmodelo.transformacao
    lnp_entrada = ltransf.entrada(fnp_entradas_dbz)
    li_passos = min(fi_num_passos, fmodelo.hparams["num_alvos"])

    llst_passos = []
    for li_lead in range(li_passos):
        lnp_probabilidades = _executa_rede(fmodelo, lnp_entrada, li_lead)
        llst_passos.append(ltransf.probabilidades_para_dbz(lnp_probabilidades))

    return np.stack(llst_passos)

# -------------------------------------------------------------------------------------
MDCT_ESTRATEGIAS = {um.CS_MODO_RECURSIVO: _preve_recursivo,
                    um.CS_MODO_DIRETO: _preve_direto,
                    um.CS_MODO_LEAD: _preve_por_lead}

# < previsão >---------------------------------------------------------------------------

def preve_dbz(fmodelo: um.SModeloCarregado, fnp_entradas_dbz: np.ndarray,
              fi_num_passos: int) -> np.ndarray:
    """
    previsão em dBZ para a grade inteira.

    :param fmodelo: checkpoint carregado
    :param fnp_entradas_dbz: (num_entradas, H, W) em dBZ, NaN fora da cobertura, do mais
                             antigo ao mais novo
    :param fi_num_passos: passos a prever

    :returns: (passos, H, W) float32 em dBZ (-32 = sem eco, NaN fora da cobertura)

    :raises ValueError: número de entradas diferente do treinado
    """
    li_esperado = fmodelo.hparams["num_entradas"]

    if fnp_entradas_dbz.shape[0] != li_esperado:
        raise ValueError(f"{fmodelo.backend}: {fnp_entradas_dbz.shape[0]} entradas; "
                         f"o modelo foi treinado com {li_esperado}")

    lt_padding = ud.calcula_padding(fnp_entradas_dbz.shape)
    lnp_entradas = ud.aplica_padding(fnp_entradas_dbz, lt_padding)

    lnp_previsao = MDCT_ESTRATEGIAS[fmodelo.hparams["modo"]](fmodelo, lnp_entradas,
                                                            fi_num_passos)
    lnp_previsao = ud.remove_padding(lnp_previsao, lt_padding).astype(np.float32)

    # fora da cobertura da última entrada: sem previsão
    lnp_previsao[:, ~ud.mascara_valida(fnp_entradas_dbz[-1])] = np.nan

    return lnp_previsao

# -------------------------------------------------------------------------------------
def threads_por_backend(fi_num_backends: int) -> int:
    """
    threads do torch em cpu quando N backends rodam em paralelo (CPUs / N, mínimo 1).
    """
    return max(1, (os.cpu_count() or 1) // max(1, fi_num_backends))

# < entrada comum dos backends (untMain) >-----------------------------------------------

def _completa_entradas(fnp_pilha_dbz: np.ndarray, fi_num_entradas: int) -> np.ndarray:
    """
    as fi_num_entradas mais recentes; com menos quadros, repete o mais antigo à esquerda.
    """
    if fnp_pilha_dbz.shape[0] >= fi_num_entradas:
        return fnp_pilha_dbz[-fi_num_entradas:]

    li_faltam = fi_num_entradas - fnp_pilha_dbz.shape[0]
    M_LOG.warning(f"[!] {fnp_pilha_dbz.shape[0]} quadros de entrada; o mais antigo foi "
                  f"repetido {li_faltam}x.")

    return np.concatenate([np.repeat(fnp_pilha_dbz[:1], li_faltam, axis=0), fnp_pilha_dbz])

# -------------------------------------------------------------------------------------
def gera_nowcast_do_backend(fs_backend: str, fnp_refletividade_dbz: np.ndarray,
                            ft_radar: obc.GeometriaRadar, fi_num_passos: int,
                            fi_passo_minutos: int, fs_prefixo_saida: str) -> obc.ResultadoNowcast:
    """
    o gera_nowcast() dos três backends (contrato de unt_backend_comum).

    o checkpoint vem de $UNT_MODELOS_DIR/<backend>/atual.pt e o device de $UNT_DEVICE
    (padrão auto); o untMain define as duas antes de criar os processos.

    fluxo:
      1. garante uma sequência 3D e valida (tudo NaN -> ValueError);
      2. sem chuva nenhuma -> (None, None, metadados), como no 25.oflow;
      3. carrega o checkpoint (uma vez por processo) e prevê em dBZ (preve_dbz);
      4. grava <prefixo>_NN.png e devolve dBZ e mm/h.

    :raises ValueError: entrada sem nenhum valor finito
    :raises FileNotFoundError: backend sem checkpoint
    """
    lnp_refletividade = obc.converte_frame_unico_em_sequencia(fnp_refletividade_dbz,
                                                             fi_num_copias=4)
    ldct_metadados = obc.monta_metadados(ft_radar, lnp_refletividade)

    obc.valida_refletividade(lnp_refletividade)

    lnp_chuva_mm_h = ud.dbz_para_mm_h(lnp_refletividade)

    if not obc.existe_chuva(lnp_chuva_mm_h):
        return obc.ResultadoNowcast(None, None, ldct_metadados)

    ls_device = os.environ.get("UNT_DEVICE", "auto")
    lmodelo = carrega_modelo_atual(fs_backend, um.diretorio_modelos_padrao(), ls_device)

    li_threads = int(os.environ.get("UNT_THREADS", "0") or 0)
    ue.configura_threads_cpu(device_do_modelo(lmodelo), li_threads)

    lnp_entradas = _completa_entradas(lnp_refletividade, lmodelo.hparams["num_entradas"])
    lnp_previsao_dbz = preve_dbz(lmodelo, lnp_entradas, fi_num_passos)
    lnp_previsao_mm_h = ud.dbz_para_mm_h(lnp_previsao_dbz)

    obc.salva_frames_da_previsao(lnp_previsao_dbz, ft_radar, ldct_metadados, fs_backend,
                                 fi_passo_minutos, fs_prefixo_saida)

    return obc.ResultadoNowcast(lnp_previsao_dbz, lnp_previsao_mm_h, ldct_metadados)

# < the end >----------------------------------------------------------------------------
