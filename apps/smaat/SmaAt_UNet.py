from torch import nn
from smaat.unet_parts import OutConv
from smaat.unet_parts_depthwise_separable import DoubleConvDS, UpDS, DownDS
from smaat.layers import CBAM


class SmaAt_UNet(nn.Module):
    def __init__(
        self,
        n_channels,
        n_classes,
        kernels_per_layer=2,
        bilinear=True,
        reduction_ratio=16,
        largura_base=64,
    ):
        # [25.u-net] largura_base (64 = artigo) permite redes estreitas para treino em CPU
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        kernels_per_layer = kernels_per_layer
        self.bilinear = bilinear
        reduction_ratio = reduction_ratio
        # [25.u-net] larguras derivadas de largura_base (artigo: 64, 128, 256, 512, 1024)
        w1, w2, w3, w4, w5 = (largura_base * m for m in (1, 2, 4, 8, 16))

        self.inc = DoubleConvDS(self.n_channels, w1, kernels_per_layer=kernels_per_layer)
        self.cbam1 = CBAM(w1, reduction_ratio=reduction_ratio)
        self.down1 = DownDS(w1, w2, kernels_per_layer=kernels_per_layer)
        self.cbam2 = CBAM(w2, reduction_ratio=reduction_ratio)
        self.down2 = DownDS(w2, w3, kernels_per_layer=kernels_per_layer)
        self.cbam3 = CBAM(w3, reduction_ratio=reduction_ratio)
        self.down3 = DownDS(w3, w4, kernels_per_layer=kernels_per_layer)
        self.cbam4 = CBAM(w4, reduction_ratio=reduction_ratio)
        factor = 2 if self.bilinear else 1
        self.down4 = DownDS(w4, w5 // factor, kernels_per_layer=kernels_per_layer)
        self.cbam5 = CBAM(w5 // factor, reduction_ratio=reduction_ratio)
        self.up1 = UpDS(w5, w4 // factor, self.bilinear, kernels_per_layer=kernels_per_layer)
        self.up2 = UpDS(w4, w3 // factor, self.bilinear, kernels_per_layer=kernels_per_layer)
        self.up3 = UpDS(w3, w2 // factor, self.bilinear, kernels_per_layer=kernels_per_layer)
        self.up4 = UpDS(w2, w1, self.bilinear, kernels_per_layer=kernels_per_layer)

        self.outc = OutConv(w1, self.n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x1Att = self.cbam1(x1)
        x2 = self.down1(x1)
        x2Att = self.cbam2(x2)
        x3 = self.down2(x2)
        x3Att = self.cbam3(x3)
        x4 = self.down3(x3)
        x4Att = self.cbam4(x4)
        x5 = self.down4(x4)
        x5Att = self.cbam5(x5)
        x = self.up1(x5Att, x4Att)
        x = self.up2(x, x3Att)
        x = self.up3(x, x2Att)
        x = self.up4(x, x1Att)
        logits = self.outc(x)
        return logits
