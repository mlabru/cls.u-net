# -*- coding: utf-8 -*-
"""
Centro Logístico de Simulação Meteorológica
Copyright (c) 1995-2026, Instituto de Controle do Espaço Aéreo

unt_rede_icea

rede própria do backend "pytorch" (linha de pesquisa ICEA-Google, issue #231), com os
princípios das sessões de capacitação (relatório, seção 7.3.3):
  (i)   U-Net: downsampling em 4 níveis + upsampling com skip connections (os blocos
        Up/OutConv vêm do pytorch_unet vendorizado);
  (ii)  blocos residuais em cada nível (fluxo do gradiente);
  (iii) lead time como entrada: one-hot dos passos em canais constantes empilhados
        ("codificação temporal por canais empilhados", 2ª sessão), então UM modelo
        cobre +10..+60 min, previsão direta (sem recursão);
  (iv)  saída em logits de bins de intensidade, treinada com entropia cruzada
        (unt_treino) em vez de MSE;
  (v)   cabeças de saída por variável: por ora só "refletividade"; descargas
        elétricas entram com a issue #226 (canais_extras na entrada idem).

PROTÓTIPO para a avaliação comparativa: a arquitetura final será consolidada depois
das próximas sessões da parceria.

2026.out  mlabru  initial version (Linux/Python)
"""
# < imports >----------------------------------------------------------------------------

# pyTorch
import torch
from torch import nn

# blocos do U-Net (milesial/Pytorch-UNet, vendorizado)
from pytorch_unet.unet_parts import OutConv, Up

# < blocos >-----------------------------------------------------------------------------

class BlocoResidual(nn.Module):
    """
    conv3x3-BN-ReLU-conv3x3-BN + atalho (1x1 se o número de canais muda), ReLU no fim.
    """
    def __init__(self, fi_entrada: int, fi_saida: int):
        super().__init__()

        self.ramo = nn.Sequential(
            nn.Conv2d(fi_entrada, fi_saida, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(fi_saida),
            nn.ReLU(inplace=True),
            nn.Conv2d(fi_saida, fi_saida, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(fi_saida))

        if fi_entrada == fi_saida:
            self.atalho = nn.Identity()
        else:
            self.atalho = nn.Sequential(nn.Conv2d(fi_entrada, fi_saida, kernel_size=1, bias=False),
                                        nn.BatchNorm2d(fi_saida))

        self.ativacao = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.ativacao(self.ramo(x) + self.atalho(x))

# -------------------------------------------------------------------------------------
def _pilha_residual(fi_entrada: int, fi_saida: int, fi_blocos: int) -> nn.Sequential:
    """
    fi_blocos blocos residuais; o primeiro ajusta o número de canais.
    """
    llst_blocos = [BlocoResidual(fi_entrada, fi_saida)]

    for _ in range(fi_blocos - 1):
        llst_blocos.append(BlocoResidual(fi_saida, fi_saida))

    return nn.Sequential(*llst_blocos)

# < rede >-------------------------------------------------------------------------------

class RedeICEA(nn.Module):
    """
    U-Net residual condicionada ao lead time.

    forward(x, lead): x (B, quadros + 1 máscara + canais_extras, H, W); lead (B,) int64
    em [0, num_leads). Devolve logits (B, num_classes, H, W). H e W múltiplos de 16.
    """
    def __init__(self, fi_canais_entrada: int=5, fi_num_leads: int=6, fi_num_classes: int=10,
                 fi_largura_base: int=32, fi_blocos_por_nivel: int=2, fi_canais_extras: int=0):
        super().__init__()

        self.num_leads = fi_num_leads

        l1, l2, l3, l4, l5 = (fi_largura_base * m for m in (1, 2, 4, 8, 16))
        li_entrada = fi_canais_entrada + fi_canais_extras + fi_num_leads

        self.inicio = _pilha_residual(li_entrada, l1, fi_blocos_por_nivel)

        self.pool = nn.MaxPool2d(2)
        self.desce1 = _pilha_residual(l1, l2, fi_blocos_por_nivel)
        self.desce2 = _pilha_residual(l2, l3, fi_blocos_por_nivel)
        self.desce3 = _pilha_residual(l3, l4, fi_blocos_por_nivel)
        # upsampling bilinear (Up(bilinear=True)) => o fundo tem metade dos canais
        self.desce4 = _pilha_residual(l4, l5 // 2, fi_blocos_por_nivel)

        self.sobe1 = Up(l5, l4 // 2, bilinear=True)
        self.sobe2 = Up(l4, l3 // 2, bilinear=True)
        self.sobe3 = Up(l3, l2 // 2, bilinear=True)
        self.sobe4 = Up(l2, l1, bilinear=True)
        self.refina = _pilha_residual(l1, l1, 1)

        # cabeças por variável prevista
        self.cabecas = nn.ModuleDict({"refletividade": OutConv(l1, fi_num_classes)})

    def _canais_de_lead(self, flead: torch.Tensor, ft_shape: tuple) -> torch.Tensor:
        """
        one-hot do lead em canais constantes: (B, num_leads, H, W).
        """
        lone_hot = nn.functional.one_hot(flead, self.num_leads).to(dtype=torch.float32)

        return lone_hot[:, :, None, None].expand(-1, -1, ft_shape[-2], ft_shape[-1])

    def forward(self, x: torch.Tensor, flead: torch.Tensor) -> torch.Tensor:
        lcanais_lead = self._canais_de_lead(flead, x.shape).to(dtype=x.dtype)
        x = torch.cat([x, lcanais_lead], dim=1)

        x1 = self.inicio(x)
        x2 = self.desce1(self.pool(x1))
        x3 = self.desce2(self.pool(x2))
        x4 = self.desce3(self.pool(x3))
        x5 = self.desce4(self.pool(x4))

        x = self.sobe1(x5, x4)
        x = self.sobe2(x, x3)
        x = self.sobe3(x, x2)
        x = self.refina(self.sobe4(x, x1))

        return self.cabecas["refletividade"](x)

# < the end >----------------------------------------------------------------------------
