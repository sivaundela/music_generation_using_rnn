import os
import numpy as np
import argparse
from keras.models import Sequential, load_model
from keras.layers import LSTM, Dropout, TimeDistributed, Dense, Activation, Embedding
# from train import epoch, vocab_size, idx_to_char
from model import build_model, save_weights, load_weights

MODEL_DIR = './model'
# model2 = Sequential()
# model2.add(Embedding(vocab_size, 512, batch_input_shape=(1,1)))
# for i in range(3):
#     model2.add(LSTM(256, return_sequences=True, stateful=True))
#     model2.add(Dropout(0.2))

# model2.add(TimeDistributed(Dense(vocab_size))) 
# model2.add(Activation('softmax'))

x = model.load_weights(os.path.join(MODEL_DIR, 'weights.100.h5'))
x.summary()

sampled = []
for i in range(1024):
    batch = np.zeros((1, 1))
    if sampled:
        batch[0, 0] = sampled[-1]
    else:
        batch[0, 0] = np.random.randint(vocab_size)
    result = model2.predict_on_batch(batch).ravel()
    sample = np.random.choice(range(vocab_size), p=result)
    sampled.append(sample)

print("sampled")
print(sampled)
print(''.join(idx_to_char[c] for c in sampled))
