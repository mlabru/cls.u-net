# pytorch_unet (vendorizado, provisório)

- Origem: https://github.com/milesial/Pytorch-UNet, commit `21d7850f2af30a9695bbeea75f3136aa538cfc4a`
  (`unet/unet_parts.py`, `unet/unet_model.py`, `LICENSE`).
- **Licença: GPL-3.0** (ver `LICENSE`). Uso interno na avaliação comparativa; o modelo próprio
  ICEA-Google (issue #231) será reescrito sem esta dependência depois da escolha da arquitetura.

Usado como base do backend `pytorch` (`work/unt_modelos.py`): `Up`/`OutConv` e o padrão
encoder/decoder. Sem alterações locais.
