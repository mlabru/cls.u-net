# -*- coding: utf-8 -*-
"""
[25.u-net] porte em PyTorch do RainNet v1.0 (Ayzel et al., 2020), camada por camada a partir
de rainnet_keras.py (hydrogo/rainnet, MIT, commit 7ce02d8; ver LEIA-ME.md).

diferenças em relação ao original, todas marcadas [25.u-net]:
  - layout canais-primeiro do PyTorch: entrada (B, C, H, W) em vez de (B, H, W, C);
  - largura_base (64 = artigo) para redes estreitas no treino em CPU;
  - n_saidas (1 = artigo) para o modo direto opcional.

H e W precisam ser múltiplos de 16 (quatro max-poolings 2x2); o chamador faz o padding.
"""
# < imports >----------------------------------------------------------------------------

# pyTorch
import torch
from torch import nn

# < blocos >-----------------------------------------------------------------------------

def _conv3x3(fi_entrada: int, fi_saida: int) -> nn.Conv2d:
    """
    Conv2D(f, 3, padding='same', kernel_initializer='he_normal') do Keras.
    """
    lconv = nn.Conv2d(fi_entrada, fi_saida, kernel_size=3, padding=1)
    nn.init.kaiming_normal_(lconv.weight, nonlinearity="relu")
    nn.init.zeros_(lconv.bias)

    return lconv

# -------------------------------------------------------------------------------------
class _DuplaConv(nn.Sequential):
    """
    convNf -> relu -> convNs -> relu (um estágio do RainNet).
    """
    def __init__(self, fi_entrada: int, fi_saida: int):
        super().__init__(_conv3x3(fi_entrada, fi_saida), nn.ReLU(inplace=True),
                         _conv3x3(fi_saida, fi_saida), nn.ReLU(inplace=True))

# < modelo >-----------------------------------------------------------------------------

class RainNet(nn.Module):
    """
    RainNet v1.0: U-Net de 5 níveis, sem BatchNorm, upsampling por vizinho mais próximo,
    dropout 0.5 nos dois níveis mais profundos.

    modo "regression" (saída linear) ou "segmentation" (sigmóide), como no original.
    """
    def __init__(self, fi_canais_entrada: int=4, fs_modo: str="regression",
                 fi_largura_base: int=64, fi_n_saidas: int=1):
        super().__init__()

        if fs_modo not in ("regression", "segmentation"):
            raise ValueError(f"modo inválido: {fs_modo}")

        self.modo = fs_modo

        # [25.u-net] larguras derivadas de largura_base (artigo: 64, 128, 256, 512, 1024)
        l1, l2, l3, l4, l5 = (fi_largura_base * m for m in (1, 2, 4, 8, 16))

        self.conv1 = _DuplaConv(fi_canais_entrada, l1)
        self.conv2 = _DuplaConv(l1, l2)
        self.conv3 = _DuplaConv(l2, l3)
        self.conv4 = _DuplaConv(l3, l4)
        self.conv5 = _DuplaConv(l4, l5)

        self.pool = nn.MaxPool2d(2)
        self.drop = nn.Dropout(0.5)
        self.up = nn.Upsample(scale_factor=2, mode="nearest")

        self.conv6 = _DuplaConv(l5 + l4, l4)
        self.conv7 = _DuplaConv(l4 + l3, l3)
        self.conv8 = _DuplaConv(l3 + l2, l2)
        self.conv9 = _DuplaConv(l2 + l1, l1)

        self.conv9_extra = _conv3x3(l1, 2)
        self.saida = nn.Conv2d(2, fi_n_saidas, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        conv1s = self.conv1(x)
        conv2s = self.conv2(self.pool(conv1s))
        conv3s = self.conv3(self.pool(conv2s))

        # o skip usa conv4s (antes do dropout), como no original
        conv4s = self.conv4(self.pool(conv3s))
        drop4 = self.drop(conv4s)

        drop5 = self.drop(self.conv5(self.pool(drop4)))

        conv6 = self.conv6(torch.cat([self.up(drop5), conv4s], dim=1))
        conv7 = self.conv7(torch.cat([self.up(conv6), conv3s], dim=1))
        conv8 = self.conv8(torch.cat([self.up(conv7), conv2s], dim=1))
        conv9 = self.conv9(torch.cat([self.up(conv8), conv1s], dim=1))
        conv9 = torch.relu(self.conv9_extra(conv9))

        lsaida = self.saida(conv9)

        if "segmentation" == self.modo:
            return torch.sigmoid(lsaida)

        return lsaida

# < the end >----------------------------------------------------------------------------
