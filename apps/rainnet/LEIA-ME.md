# rainnet (vendorizado)

- Origem: https://github.com/hydrogo/rainnet, commit `7ce02d8526bcc848e632b9e4a25ff7294a2f70f9`,
  licença MIT (`LICENSE`).
- Artigo: Ayzel, Scheffer & Heistermann (2020), *RainNet v1.0: a convolutional neural network for
  radar-based precipitation nowcasting*, GMD 13, 2631–2644.
- `rainnet_keras.py`: o original (Keras), mantido como referência e **não importado**.
- `rainnet.py`: porte em PyTorch camada por camada (o `~/.venv` não tem TensorFlow). Diferenças
  marcadas `[25.u-net]`: layout (B, C, H, W), `largura_base` (64 = artigo) e `n_saidas`.

Pré-processamento do artigo (aplicado em `work/unt_dados.py`): mm/h → `log(x + 0.01)`; perda
`log_cosh`; previsão recursiva de 6 passos.
