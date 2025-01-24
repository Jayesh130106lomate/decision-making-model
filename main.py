import os
# Disable oneDNN optimizations
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, LSTM, Input, Dropout, Embedding, Bidirectional, BatchNormalization
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight

funcs = [
    "exit", "general", "realtime", "open", "close", "play", "system",
    "generate image", "content", "google search", "youtube search", "reminder"
]

# Define the tokenizer
tokenizer = keras.preprocessing.text.Tokenizer(char_level=True)

# Check if the model file exists
model_file = 'trained_model.keras'
tokenizer_file = 'tokenizer_config.json'
if os.path.exists(model_file) and os.path.exists(tokenizer_file):
    print("Loading existing model and tokenizer...")
    model = keras.models.load_model(model_file)
    
    # Load the tokenizer configuration from a file
    with open(tokenizer_file, 'r') as file:
        tokenizer_config = file.read()
    tokenizer = keras.preprocessing.text.tokenizer_from_json(tokenizer_config)

else:
    print("Model or tokenizer file not found. Importing training data and training new model...")
    
    from training_data import training_data  # Importing training data
    from training_data1 import training_data1  # Importing training data 1
    from training_data2 import training_data2  # Importing training data 2
    from training_data3 import training_data3  # Importing training data 3
    from training_data4 import training_data4  # Importing training data 4
    from training_data5 import training_data5  # Importing training data 5
    from training_data6 import training_data6  # Importing training data 6
    
    # Update the training data with more examples for better accuracy
    extended_training_data = training_data6 + training_data5 + training_data4 + training_data3 + training_data2 + training_data1 + training_data

    label_to_index = {label: index for index, label in enumerate(funcs)}
    index_to_label = {index: label for label, index in label_to_index.items()}

    # Tokenize and pad the input sentences
    tokenizer.fit_on_texts([item["query"] for item in extended_training_data])
    X_train = tokenizer.texts_to_sequences([item["query"] for item in extended_training_data])
    X_train = keras.preprocessing.sequence.pad_sequences(X_train, maxlen=50, padding='post')
    y_train = np.array([label_to_index[item["label"]] for item in extended_training_data])

    # Ensure the shapes are correct
    X_train = np.array(X_train)
    y_train = keras.utils.to_categorical(y_train, num_classes=len(funcs))

    # Calculate class weights to handle class imbalance
    class_weights = class_weight.compute_class_weight(class_weight='balanced', classes=np.unique(y_train.argmax(axis=1)), y=y_train.argmax(axis=1))
    class_weights = {i: class_weights[i] for i in range(len(class_weights))}

    # Split the data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

    def build_model(input_shape):
        model = Sequential()
        model.add(Embedding(input_dim=len(tokenizer.word_index)+1, output_dim=128, input_length=input_shape[0]))
        model.add(Bidirectional(LSTM(256, return_sequences=True)))
        model.add(Bidirectional(LSTM(128)))
        model.add(Dense(64, activation='relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.5))
        model.add(Dense(len(funcs), activation='softmax'))

        model.compile(loss='categorical_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])
        return model

    input_shape = (X_train.shape[1],)
    model = build_model(input_shape)

    # Callbacks for early stopping and learning rate reduction
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, min_lr=1e-6)

    # Train the model
    model.fit(X_train, y_train, epochs=100, batch_size=64, validation_data=(X_val, y_val), callbacks=[early_stopping, reduce_lr], class_weight=class_weights)

    # Save the model using the native Keras format
    model.save(model_file)
    
    # Save the tokenizer configuration to a file
    tokenizer_config = tokenizer.to_json()
    with open(tokenizer_file, 'w') as file:
        file.write(tokenizer_config)

def preprocess_data(prompt):
    data = tokenizer.texts_to_sequences([prompt])
    data = keras.preprocessing.sequence.pad_sequences(data, maxlen=50, padding='post')
    data = np.array(data)
    return data

def FirstLayerDMM(prompt: str):
    data = preprocess_data(prompt)
    prediction = model.predict(data)
    response_index = np.argmax(prediction)
    response = funcs[response_index]
    return response

if __name__ == "__main__":
    while True:
        user_input = input(">>> ")
        response = FirstLayerDMM(user_input)
        print(response)
