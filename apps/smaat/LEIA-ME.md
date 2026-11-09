# smaat (vendorizado, provisório)

- Origem: https://github.com/HansBambel/SmaAt-UNet, commit `3e48dcca3e88d9f154bed83268a0e9ec4355371a`
  (`models/SmaAt_UNet.py`, `layers.py`, `unet_parts.py`, `unet_parts_depthwise_separable.py`).
- Artigo: Trebing, Stańczyk & Mehrkanoon (2021), *SmaAt-UNet: Precipitation nowcasting using a small
  attention-UNet architecture*, Pattern Recognition Letters 145.
- **Licença:** o repositório não tem arquivo LICENSE (todos os direitos reservados por padrão);
  `unet_parts*.py` derivam do milesial/Pytorch-UNet (GPL-3.0). Uso interno para a avaliação
  comparativa (issue #230). A versão final será reescrita depois da escolha da arquitetura.

Alterações locais (marcadas `[25.u-net]`):
- imports `models.` → `smaat.` (namespace package em `apps/`);
- `SmaAt_UNet(..., largura_base=64)`: larguras derivadas de `largura_base` (64 = artigo);
- `ChannelAttention`: camada oculta com no mínimo 1 neurônio (redes estreitas).
