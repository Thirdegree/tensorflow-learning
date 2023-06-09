import tensorflow as tf
import numpy as np
from itertools import chain
from pathlib import Path

import os
import time

def read_data() -> str:
    return ''.join(
        chain(
            Path('./kjv.txt').read_text(),
            # Path('./code-text.txt').read_text()
        ))
    # return Path('./shakespeare.txt').read_text()


def main() -> None:
    text = read_data()

    print('building vocab')
    vocab: list[str] = sorted(set(text))

    print('building chars/string lookups')
    chars = tf.strings.unicode_split(text, input_encoding='UTF-8')
    ids_from_chars = tf.keras.layers.StringLookup(vocabulary=list(vocab),
                                                  mask_token=None)
    chars_from_ids = tf.keras.layers.StringLookup(
        vocabulary=ids_from_chars.get_vocabulary(),
        invert=True,
        mask_token=None)

    def text_from_ids(ids):
        return tf.strings.reduce_join(chars_from_ids(ids), axis=-1)

    print('building dataset')
    all_ids = ids_from_chars(tf.strings.unicode_split(read_data(), input_encoding='UTF-8'))
    ids_dataset = tf.data.Dataset.from_tensor_slices(all_ids)
    seq_length = 100
    print('building sequences')
    sequences = ids_dataset.batch(seq_length + 1, drop_remainder=True)
    dataset = sequences.map(split_input_target)
    # Batch size
    BATCH_SIZE = 64

    # Buffer size to shuffle the dataset
    # (TF data is designed to work with possibly infinite sequences,
    # so it doesn't attempt to shuffle the entire sequence in memory. Instead,
    # it maintains a buffer in which it shuffles elements).
    BUFFER_SIZE = 10000

    dataset = (dataset.shuffle(BUFFER_SIZE).batch(
        BATCH_SIZE,
        drop_remainder=True).prefetch(tf.data.experimental.AUTOTUNE))

    # Length of the vocabulary in StringLookup Layer
    vocab_size = len(ids_from_chars.get_vocabulary())

    # The embedding dimension
    embedding_dim = 256

    # Number of RNN units
    rnn_units = 1024

    model = MyModel(vocab_size=vocab_size,
                    embedding_dim=embedding_dim,
                    rnn_units=rnn_units)

    loss = tf.losses.SparseCategoricalCrossentropy(from_logits=True)
    model.compile(optimizer='adam', loss=loss)
    # Directory where the checkpoints will be saved
    checkpoint_dir = './training_checkpoints'
    # Name of the checkpoint files
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")

    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_prefix, save_weights_only=True)
    EPOCHS = 6
    for input_example_batch, target_example_batch in dataset.take(1):
        example_batch_predictions = model(input_example_batch)
        print(example_batch_predictions.shape, "# (batch_size, sequence_length, vocab_size)")
        sampled_indices = tf.random.categorical(example_batch_predictions[0], num_samples=1)
        sampled_indices = tf.squeeze(sampled_indices, axis=-1).numpy()
        print("Input:\n", text_from_ids(input_example_batch[0]).numpy())
        print()
        print("Next Char Predictions:\n", text_from_ids(sampled_indices).numpy())
        example_batch_mean_loss = loss(target_example_batch, example_batch_predictions)
        print("Prediction shape: ", example_batch_predictions.shape, " # (batch_size, sequence_length, vocab_size)")
        print("Mean loss:        ", example_batch_mean_loss)
        print("exp:              ", tf.exp(example_batch_mean_loss).numpy())
        print("vocab size:       ", tf.exp(example_batch_mean_loss).numpy())




    print(model.summary())

    print('training model')
    history = model.fit(dataset,
                        epochs=EPOCHS,
                        callbacks=[checkpoint_callback],
                        metrics=['accuracy'])
    one_step_model = OneStep(model, chars_from_ids, ids_from_chars)

    print('predicting gen...')
    start = time.time()
    states = None
    next_char = tf.constant(['Ge1:'])
    result = [next_char]

    for n in range(1000):
        next_char, states = one_step_model.generate_one_step(next_char,
                                                             states=states)
        result.append(next_char)

    result = tf.strings.join(result)
    end = time.time()
    print(result[0].numpy().decode('utf-8'), '\n\n' + '_' * 80)
    print('\nRun time:', end - start)

    print('predicting rev...')
    start = time.time()
    states = None
    next_char = tf.constant(['Rev1:'])
    result = [next_char]

    for n in range(1000):
        next_char, states = one_step_model.generate_one_step(next_char,
                                                             states=states)
        result.append(next_char)

    result = tf.strings.join(result)
    end = time.time()
    print(result[0].numpy().decode('utf-8'), '\n\n' + '_' * 80)
    print('\nRun time:', end - start)


class MyModel(tf.keras.Model):

    def __init__(self, vocab_size: int, embedding_dim: int,
                 rnn_units: int) -> None:
        super().__init__(self)
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.gru = tf.keras.layers.GRU(rnn_units,
                                       return_sequences=True,
                                       return_state=True)
        self.dense = tf.keras.layers.Dense(vocab_size)

    def call(self, inputs, states=None, return_state=False, training=False):
        x = inputs
        x = self.embedding(x, training=training)
        if states is None:
            states = self.gru.get_initial_state(x)
        x, states = self.gru(x, initial_state=states, training=training)
        x = self.dense(x, training=training)

        if return_state:
            return x, states
        else:
            return x


class OneStep(tf.keras.Model):

    def __init__(self, model, chars_from_ids, ids_from_chars, temperature=1.0):
        super().__init__()
        self.temperature = temperature
        self.model = model
        self.chars_from_ids = chars_from_ids
        self.ids_from_chars = ids_from_chars

        # Create a mask to prevent "[UNK]" from being generated.
        skip_ids = self.ids_from_chars(['[UNK]'])[:, None]
        sparse_mask = tf.SparseTensor(
            # Put a -inf at each bad index.
            values=[-float('inf')] * len(skip_ids),
            indices=skip_ids,
            # Match the shape to the vocabulary
            dense_shape=[len(ids_from_chars.get_vocabulary())])
        self.prediction_mask = tf.sparse.to_dense(sparse_mask)

    @tf.function
    def generate_one_step(self, inputs, states=None):
        # Convert strings to token IDs.
        input_chars = tf.strings.unicode_split(inputs, 'UTF-8')
        input_ids = self.ids_from_chars(input_chars).to_tensor()

        # Run the model.
        # predicted_logits.shape is [batch, char, next_char_logits]
        predicted_logits, states = self.model(inputs=input_ids,
                                              states=states,
                                              return_state=True)
        # Only use the last prediction.
        predicted_logits = predicted_logits[:, -1, :]
        predicted_logits = predicted_logits / self.temperature
        # Apply the prediction mask: prevent "[UNK]" from being generated.
        predicted_logits = predicted_logits + self.prediction_mask

        # Sample the output logits to generate token IDs.
        predicted_ids = tf.random.categorical(predicted_logits, num_samples=1)
        predicted_ids = tf.squeeze(predicted_ids, axis=-1)

        # Convert from token ids to characters
        predicted_chars = self.chars_from_ids(predicted_ids)

        # Return the characters and model state.
        return predicted_chars, states


def split_input_target(sequence):
    input_text = sequence[:-1]
    target_text = sequence[1:]
    return input_text, target_text


if __name__ == "__main__":
    print('starting')
    main()
