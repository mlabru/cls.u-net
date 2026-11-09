# -*- coding: utf-8 -*-
"""
Centro Logístico de Simulação Meteorológica
Copyright (c) 1995-2026, Instituto de Controle do Espaço Aéreo

unt_device

escolha do device do PyTorch. O treino de verdade roda num host com GPU, mas tudo
precisa funcionar só com CPU (o host de desenvolvimento não tem GPU):

  - "auto": cuda se disponível, senão mps (Apple), senão cpu;
  - AMP e pin_memory só em cuda; a precisão mista é bfloat16 quando a GPU suporta
    (mesma faixa do float32, sem GradScaler), senão float16;
  - em cpu, o número de threads do torch é limitado (fi_threads) para não disputar
    os núcleos com os workers do DataLoader ou com outros backends em paralelo.

2026.out  mlabru  dtype_amp: bfloat16 quando a GPU suporta (o float16 estourava no
                  rainnet: até 17 mil lotes NaN por época)
2026.out  mlabru  initial version (Linux/Python)
"""
# < imports >----------------------------------------------------------------------------

# python library
import logging
import os

# pyTorch
import torch

# < logging >----------------------------------------------------------------------------

# logger
M_LOG = logging.getLogger(__name__)
M_LOG.setLevel(logging.INFO)

# -------------------------------------------------------------------------------------
def _mps_disponivel() -> bool:
    """
    backend Metal (Apple) disponível nesta instalação do torch.
    """
    lbackend_mps = getattr(torch.backends, "mps", None)

    if lbackend_mps is None:
        return False

    return bool(lbackend_mps.is_available())

# -------------------------------------------------------------------------------------
def escolhe_device(fs_pedido: str="auto") -> torch.device:
    """
    resolve o device pedido.

    :param fs_pedido: "auto", "cpu", "cuda", "cuda:N" ou "mps"

    :returns: torch.device

    :raises RuntimeError: device explícito indisponível (ex.: "cuda" sem GPU)
    """
    ls_pedido = (fs_pedido or "auto").lower()

    if "auto" == ls_pedido:
        if torch.cuda.is_available():
            return torch.device("cuda")

        if _mps_disponivel():
            return torch.device("mps")

        return torch.device("cpu")

    if ls_pedido.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"device {fs_pedido} pedido, mas não há GPU CUDA disponível "
                           f"(use --device cpu ou auto)")

    if "mps" == ls_pedido and not _mps_disponivel():
        raise RuntimeError("device mps pedido, mas não está disponível")

    return torch.device(ls_pedido)

# -------------------------------------------------------------------------------------
def usa_amp(fdevice: torch.device, fb_pedido: bool) -> bool:
    """
    precisão mista só em cuda (em cpu/mps fica em float32).
    """
    return bool(fb_pedido) and "cuda" == fdevice.type

# -------------------------------------------------------------------------------------
def dtype_amp(fdevice: torch.device) -> torch.dtype:
    """
    tipo da precisão mista: bfloat16 quando a GPU o suporta nativamente (faixa do
    float32; dispensa o GradScaler), senão float16 (máximo 65.504; precisa do
    GradScaler).
    """
    if "cuda" == fdevice.type and torch.cuda.is_bf16_supported(including_emulation=False):
        return torch.bfloat16

    return torch.float16

# -------------------------------------------------------------------------------------
def usa_pin_memory(fdevice: torch.device) -> bool:
    """
    pin_memory do DataLoader só ajuda (e só funciona bem) com cuda.
    """
    return "cuda" == fdevice.type

# -------------------------------------------------------------------------------------
def configura_threads_cpu(fdevice: torch.device, fi_threads: int=0) -> int:
    """
    limita as threads do torch em cpu.

    :param fdevice: device em uso
    :param fi_threads: threads desejadas (0 = número de CPUs)

    :returns: threads configuradas (0 se o device não é cpu)
    """
    if "cpu" != fdevice.type:
        return 0

    li_threads = fi_threads if fi_threads > 0 else (os.cpu_count() or 1)
    torch.set_num_threads(max(1, li_threads))

    # sem NNPACK: em CPUs sem suporte ele só imprime "Could not initialize NNPACK" a
    # cada convolução (o torch cai no caminho padrão de qualquer forma)
    lnnpack = getattr(torch.backends, "nnpack", None)
    if lnnpack is not None:
        lnnpack.set_flags(False)

    return torch.get_num_threads()

# -------------------------------------------------------------------------------------
def descreve(fdevice: torch.device) -> str:
    """
    texto para o log: device, nome da GPU (se houver) e threads.
    """
    if "cuda" == fdevice.type:
        return f"cuda ({torch.cuda.get_device_name(fdevice)})"

    if "cpu" == fdevice.type:
        return f"cpu ({torch.get_num_threads()} threads)"

    return str(fdevice)

# < the end >----------------------------------------------------------------------------
