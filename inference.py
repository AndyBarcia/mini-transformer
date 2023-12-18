from transformer import *    

m = Transformer(
    vocab_size=len(sp),
    window_size=(16,8,4,2),
    window_stride=(1,2,4,8),
    n_embed=64,
    n_layers=4,
    n_head=4,
    encoding="relative",
    dropout=0.0
).to(device)
m.cargar_parametros("checkpoints/ultra_long_training_3/checkpoint.pth")
_ = summary(m, input_size=[64,256,m.n_embed], mode="train", verbose=1)

text_iter = tqdm(m.generate("hola:", 64, temperature=0.4, repetition_penalty=2.1, speculate=False))
for text in text_iter:
    text_iter.clear()
    text_iter.write(decode(text.tolist())[0])

