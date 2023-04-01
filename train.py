import os
import numpy as np
import json
import argparse
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential, load_model
from model import build_model, save_weights, load_weights
from keras.layers import LSTM, Dropout, TimeDistributed, Dense, Activation, Embedding


LOG_DIR = './logs'
DATA_DIR = './data'

BATCH_SIZE = 16
SEQ_LENGTH = 64

class TrainLogger(object):
    def __init__(self, file):
        self.file = os.path.join(LOG_DIR, file)
        self.epochs = 0
        with open(self.file, 'w') as f:
            f.write('epoch,loss,acc\n')

    def add_entry(self, loss, acc):
        self.epochs += 1
        s = '{},{},{}\n'.format(self.epochs, loss, acc)
        with open(self.file, 'a') as f:
            f.write(s)

def read_batches(T, vocab_size):
    length = T.shape[0]; #129,665
    batch_chars = int(length / BATCH_SIZE); # 8,104

    for start in range(0, batch_chars - SEQ_LENGTH, SEQ_LENGTH): # (0, 8040, 64)
        X = np.zeros((BATCH_SIZE, SEQ_LENGTH)) # 16X64
        Y = np.zeros((BATCH_SIZE, SEQ_LENGTH, vocab_size)) # 16X64X86
        for batch_idx in range(0, BATCH_SIZE): # (0,16)
            for i in range(0, SEQ_LENGTH): #(0,64)
                X[batch_idx, i] = T[batch_chars * batch_idx + start + i] # 
                Y[batch_idx, i, T[batch_chars * batch_idx + start + i + 1]] = 1
        yield X, Y

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train the model on some text.")
    parser.add_argument('--input', default='jigs.txt', help='name of the text file to train from')
    parser.add_argument('--epochs', type=int, default=100, help='number of epochs to train on')
    parser.add_argument('--freq', type=int, default=10, help='checkpoint to save frequency')
    args = parser.parse_args()

    if not os.path.exists(LOG_DIR):
        os.makedirs(LOG_DIR)

    epochs = args.epochs
    freq = args.freq
    input = open(os.path.join(DATA_DIR, args.input)).read()

    print("processing")

    char_to_idx = {
        ch:i for (i, ch) in enumerate(sorted(list(set(input))))
    }
    print("Number of unique charaters:" + str(len(char_to_idx)))

    idx_to_char = {
        i: ch for (ch, i) in char_to_idx.items()
    }
    vocab_size = len(char_to_idx)
    print(vocab_size)
    print("processing done")

    model = Sequential()
    model.add(Embedding(vocab_size, 512, batch_input_shape=(BATCH_SIZE, SEQ_LENGTH)))
    for i in range(3):
        model.add(LSTM(256, return_sequences=True, stateful=True))
        model.add(Dropout(0.2))

    model.add(TimeDistributed(Dense(vocab_size))) 
    model.add(Activation('softmax'))
    print("model created")

    model.summary()
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    #Train data generation
    print("training data")
    T = np.asarray([char_to_idx[c] for c in input], dtype=np.int32) #convert complete input text into numerical indices

    print("Length of input text:" + str(T.size)) #129,665

    steps_per_epoch = (len(input) / BATCH_SIZE - 1) / SEQ_LENGTH  

    log = TrainLogger('training_log.csv')

    for epoch in range(epochs):
        print('\nEpoch {}/{}'.format(epoch + 1, epochs))
            
        losses, accs = [], []

        for i, (X, Y) in enumerate(read_batches(T, vocab_size)):
                
            print(X)

            loss, acc = model.train_on_batch(X, Y)
            print('Batch {}: loss = {}, acc = {}'.format(i + 1, loss, acc))
            losses.append(loss)
            accs.append(acc)

        log.add_entry(np.average(losses), np.average(accs))
        
        if (epoch + 1) % freq == 0:
                save_weights(epoch + 1, model)
                print('Saved checkpoint to', 'weights.{}.h5'.format(epoch + 1))

    print("training done...........")