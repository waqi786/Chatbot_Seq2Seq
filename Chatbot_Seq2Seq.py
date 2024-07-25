import tensorflow as tf
from tensorflow.keras.layers import Input, LSTM, Dense
from tensorflow.keras.models import Model
import numpy as np

def load_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.read().split('\n')
    input_texts = []
    target_texts = []
    for line in lines:
        if line:
            input_text, target_text = line.split('|')
            input_texts.append(input_text)
            target_texts.append(target_text)
    return input_texts, target_texts

def tokenize_data(input_texts, target_texts):
    input_tokenizer = tf.keras.preprocessing.text.Tokenizer(filters='')
    input_tokenizer.fit_on_texts(input_texts)
    input_sequences = input_tokenizer.texts_to_sequences(input_texts)
    max_input_length = max([len(seq) for seq in input_sequences])
    padded_input_sequences = tf.keras.preprocessing.sequence.pad_sequences(input_sequences, maxlen=max_input_length, padding='post')

    target_tokenizer = tf.keras.preprocessing.text.Tokenizer(filters='')
    target_tokenizer.fit_on_texts(target_texts)
    target_sequences = target_tokenizer.texts_to_sequences(target_texts)
    max_target_length = max([len(seq) for seq in target_sequences])
    padded_target_sequences = tf.keras.preprocessing.sequence.pad_sequences(target_sequences, maxlen=max_target_length, padding='post')

    return padded_input_sequences, padded_target_sequences, input_tokenizer, target_tokenizer, max_input_length, max_target_length

def build_model(input_vocab_size, output_vocab_size, max_input_length, max_target_length, latent_dim=256):
    encoder_inputs = Input(shape=(max_input_length,))
    encoder_embedding = tf.keras.layers.Embedding(input_vocab_size, latent_dim, mask_zero=True)(encoder_inputs)
    encoder_lstm = LSTM(latent_dim, return_state=True)
    encoder_outputs, state_h, state_c = encoder_lstm(encoder_embedding)
    encoder_states = [state_h, state_c]

    decoder_inputs = Input(shape=(max_target_length,))
    decoder_embedding = tf.keras.layers.Embedding(output_vocab_size, latent_dim, mask_zero=True)(decoder_inputs)
    decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True)
    decoder_outputs, _, _ = decoder_lstm(decoder_embedding, initial_state=encoder_states)
    decoder_dense = Dense(output_vocab_size, activation='softmax')
    output = decoder_dense(decoder_outputs)

    model = Model([encoder_inputs, decoder_inputs], output)
    return model

def train_model(model, encoder_input_data, decoder_input_data, decoder_target_data, batch_size=64, epochs=50, validation_split=0.2):
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.fit([encoder_input_data, decoder_input_data], decoder_target_data,
              batch_size=batch_size,
              epochs=epochs,
              validation_split=validation_split)

def generate_response(input_text, model, input_tokenizer, target_tokenizer, max_target_length):
    input_seq = input_tokenizer.texts_to_sequences([input_text])
    padded_input_seq = tf.keras.preprocessing.sequence.pad_sequences(input_seq, maxlen=len(input_seq[0]), padding='post')
    decoder_input = np.zeros((1, max_target_length))
    decoder_input[0, 0] = target_tokenizer.word_index['<start>']
    for i in range(1, max_target_length):
        predictions = model.predict([padded_input_seq, decoder_input]).argmax(axis=-1)
        decoder_input[0, i] = predictions[0, i - 1]
        if predictions[0, i] == target_tokenizer.word_index['<end>']:
            break
    return ' '.join([target_tokenizer.index_word[word] for word in decoder_input[0] if word not in [0, target_tokenizer.word_index['<start>'], target_tokenizer.word_index['<end>']]])

if __name__ == "__main__":
    input_texts, target_texts = load_data('conversations.txt')
    encoder_input_data, decoder_input_data, input_tokenizer, target_tokenizer, max_input_length, max_target_length = tokenize_data(input_texts, target_texts)
    model = build_model(len(input_tokenizer.word_index)+1, len(target_tokenizer.word_index)+1, max_input_length, max_target_length)
    train_model(model, encoder_input_data, decoder_input_data, decoder_input_data, epochs=50)
    input_text = "How are you?"
    response = generate_response(input_text, model, input_tokenizer, target_tokenizer, max_target_length)
    print("Chatbot:", response)



