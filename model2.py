# https://www.thepythoncode.com/article/text-generation-keras-python
from __future__ import annotations
from itertools import chain
from functools import cache
from pathlib import Path
import tensorflow as tf
import tqdm
import numpy as np
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from dataclasses import dataclass, field
# from string import punctuation


@dataclass(frozen=True)
class ModelParams:
    sequence_length: int = 100
    batch_size: int = 1048
    epochs: int = 30
    data_files: tuple[Path, ...] = field(default_factory=lambda: (
        Path('./kjv.txt'),
        # Path('./code-text.txt')
    ))

    @property
    def model_weights_path(self) -> Path:
        fname = f"{','.join([p.name for p in self.data_files])}-{self.sequence_length}-{self.epochs}-{self.batch_size}.h5"  # noqa: E501
        p = Path("results", fname)
        p.parent.mkdir(exist_ok=True, parents=True)
        return p

    @property
    def text(self) -> str:
        data = self.read_data().lower()
        punctuation = "'()-;"
        return data.translate(str.maketrans("", "", punctuation))

    @property
    def n_unique_chars(self) -> int:
        return len(self.vocab)

    @cache
    def read_data(self) -> str:
        return '\n'.join(p.read_text() for p in self.data_files)

    def make_model(self) -> Sequential:
        return Sequential([
            LSTM(256,
                 input_shape=(self.sequence_length, self.n_unique_chars),
                 return_sequences=True),
            Dropout(0.3),
            LSTM(256),
            Dense(self.n_unique_chars, activation="softmax"),
        ])

    @property
    def vocab(self) -> str:
        return ''.join(sorted(set(self.text)))

    def char2int(self) -> dict[str, int]:
        return {c: i for i, c in enumerate(self.vocab)}

    def int2char(self) -> dict[int, str]:
        return {i: c for i, c in enumerate(self.vocab)}

    def train(self) -> None:
        # print some stats
        n_chars = len(self.text)
        print("unique_chars:", self.vocab)
        print("Number of characters:", n_chars)
        print("Number of unique characters:", self.n_unique_chars)

        # dictionary that converts characters to integers
        char2int = self.char2int()
        # dictionary that converts integers to characters
        int2char = self.int2char()

        encoded_text = np.array([char2int[c] for c in self.text])
        # construct tf.data.Dataset object
        char_dataset = tf.data.Dataset.from_tensor_slices(encoded_text)
        for char in char_dataset.take(8):
            print(char.numpy(), int2char[char.numpy()])

        sequences = char_dataset.batch(2 * self.sequence_length + 1,
                                       drop_remainder=True)
        for sequence in sequences.take(2):
            print(''.join([int2char[i] for i in sequence.numpy()]))

        def split_sample(sample):
            # example :
            # SEQUENCE_LENGTH is 10
            # sample is "python is a great pro" (21 length)
            # ds will equal to ('python is ', 'a') encoded as integers
            ds = tf.data.Dataset.from_tensors(
                (sample[:self.sequence_length], sample[self.sequence_length]))
            for i in range(1, (len(sample) - 1) // 2):
                # first (input_, target) will be ('ython is a', ' ')
                # second (input_, target) will be ('thon is a ', 'g')
                # third (input_, target) will be ('hon is a g', 'r')
                # and so on
                input_ = sample[i:i + self.sequence_length]
                target = sample[i + self.sequence_length]
                # extend the dataset with these samples by concatenate() method
                other_ds = tf.data.Dataset.from_tensors((input_, target))
                ds = ds.concatenate(other_ds)
            return ds

        # prepare inputs and targets
        dataset = sequences.flat_map(split_sample)

        def one_hot_samples(input_, target):
            # onehot encode the inputs and the targets
            # Example:
            # if character 'd' is encoded as 3 and n_unique_chars = 5
            # result should be the vector: [0, 0, 0, 1, 0], since 'd' is the 4th character
            return tf.one_hot(input_, self.n_unique_chars), tf.one_hot(
                target, self.n_unique_chars)

        dataset = dataset.map(one_hot_samples)

        for element in dataset.take(2):
            print(
                "Input:", ''.join([
                    int2char[np.argmax(char_vector)]
                    for char_vector in element[0].numpy()
                ]))
            print("Target:", int2char[np.argmax(element[1].numpy())])
            print("Input shape:", element[0].shape)
            print("Target shape:", element[1].shape)
            print("=" * 50, "\n")

        ds = dataset.repeat().shuffle(1024).batch(self.batch_size,
                                                  drop_remainder=True)
        model = self.make_model()
        model.summary()
        model.compile(loss="categorical_crossentropy",
                      optimizer="adam",
                      metrics=["accuracy"])
        model.fit(ds,
                  steps_per_epoch=(len(encoded_text) - self.sequence_length) //
                  self.batch_size,
                  epochs=self.epochs)
        model.save(str(self.model_weights_path))

    def generate(self) -> None:
        seed = "ge1:"
        s = seed
        char2int = self.char2int()
        print(char2int)
        vocab_size = len(char2int)
        model = self.make_model()
        model.load_weights(self.model_weights_path)
        n_chars = 400
        # generate 400 characters
        generated = ""
        int2char = self.int2char()
        for i in tqdm.tqdm(range(n_chars), "Generating text"):
            # make the input sequence
            X = np.zeros((1, self.sequence_length, vocab_size))
            for t, char in enumerate(seed):
                X[0, (self.sequence_length - len(seed)) + t,
                  char2int[char]] = 1
            # predict the next character
            predicted = model.predict(X, verbose=0)[0]
            # converting the vector to an integer
            next_index = np.argmax(predicted)
            # converting the integer to a character
            next_char = int2char[next_index]
            # add the character to results
            generated += next_char
            # shift seed and the predicted character
            seed = seed[1:] + next_char

        print("Seed:", s)
        print("Generated text:")
        print(generated)


if __name__ == "__main__":
    modelparams = ModelParams(epochs=15)
    if not modelparams.model_weights_path.exists():
        modelparams.train()
    modelparams.generate()
