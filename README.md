# Chatbot_Seq2Seq

This repository features a sequence-to-sequence (Seq2Seq) chatbot built using TensorFlow and Keras. The chatbot is trained on conversational data to generate responses based on input queries. The model utilizes LSTM layers for both the encoder and decoder to handle the sequential nature of the data.


Key Features:

Data Loading and Tokenization: Loads conversation data, splits it into input and target texts, and tokenizes the texts for training.
Seq2Seq Model: Constructs a Seq2Seq model with LSTM layers for encoding and decoding.
Training: Compiles and trains the model using the provided conversational data.
Response Generation: Generates responses to input queries using the trained model.
Installation
To run this project, you'll need to have Python installed. You can install the required packages using pip:

    pip install tensorflow numpy

Usage
Load Data: Ensure you have a text file conversations.txt with conversational data in the format input_text|target_text.

Train the Model: Run the training script to build and train the Seq2Seq model.

Generate Responses: Use the trained model to generate responses to input queries.

How to Run
Prepare Data: Create a conversations.txt file in the same directory as the script. Each line should be in the format input_text|target_text.

Run the Training Script:

    import tensorflow as tf
    from tensorflow.keras.layers import Input, LSTM, Dense
    from tensorflow.keras.models import Model
    import numpy as np

    # Load Data
    input_texts, target_texts = load_data('conversations.txt')

    # Tokenize Data
    encoder_input_data, decoder_input_data, input_tokenizer, target_tokenizer, max_input_length, max_target_length = tokenize_data(input_texts, target_texts)

    # Build Model
    model = build_model(len(input_tokenizer.word_index) + 1, len(target_tokenizer.word_index) + 1, max_input_length, max_target_length)

    # Train Model
    train_model(model, encoder_input_data, decoder_input_data, decoder_input_data, epochs=50)


Functions Explained:

load_data(file_path): Reads conversation data from a file and splits it into input and target texts.

tokenize_data(input_texts, target_texts): Tokenizes the input and target texts and pads them to the same length.

build_model(input_vocab_size, output_vocab_size, max_input_length, max_target_length, latent_dim=256): Builds the Seq2Seq model with LSTM layers for the encoder and decoder.

train_model(model, encoder_input_data, decoder_input_data, decoder_target_data, batch_size=64, epochs=50, validation_split=0.2): Compiles and trains the Seq2Seq model.

generate_response(input_text, model, input_tokenizer, target_tokenizer, max_target_length): Generates a response to an input query using the trained model.


Requirements:

TensorFlow
NumPy


Example:

    input_text = "How are you?"
    response = generate_response(input_text, model, input_tokenizer, target_tokenizer, max_target_length)
    print("Chatbot:", response)


Author
Waqar Ali


Uploaded Date
July 24, 2024
