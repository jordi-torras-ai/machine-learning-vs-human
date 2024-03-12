import sys
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import TensorBoard

# Function to create the neural network model
def create_model(input_shape):
    model = Sequential([
        Dense(10, activation='relu', input_shape=input_shape),
        Dense(5, activation='relu'),
        Dense(3, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Function to preprocess the data
def preprocess_data(data):
    # Map values of COL1 to COL4 from [0, 10] to [-1.0, 1.0]
    data['COL1'] = (data['COL1'] - 5.0) / 5.0
    data['COL2'] = (data['COL2'] - 5.0) / 5.0
    data['COL3'] = (data['COL3'] - 5.0) / 5.0
    data['COL4'] = (data['COL4'] - 5.0) / 5.0
    data = data.drop(columns=['id'])
    return data

# Function to display the model
def print_model_parameters(model):
    """Print the parameters of each layer in the model."""
    print("Model Summary:")
    print("--------------------")
    for layer in model.layers:
        print(f"Layer {layer.name} {type(layer).__name__}")
        weights, biases = layer.get_weights()
        print("\tWeights:\t\t")
        print_weights(weights)
        print("\tBiases:", *(f"{x:.2f}" for x in biases), sep=", ")

def print_weights(weights):
    """Print the weights of a layer."""
    for i, w in enumerate(weights):
        print(f"\t\tWeight {i + 1}:", *(f"{x:.2f}" for x in w), sep=", ")

# Check if the CSV file name and model file name are provided as command-line arguments
if len(sys.argv) != 3:
    print("Usage: python train_and_save_model.py <csv_file> <model_file>")
    sys.exit(1)

# Extract the CSV file name and model file name from command-line arguments
csv_file = sys.argv[1]
model_file = sys.argv[2]

# Load the CSV file into a DataFrame
try:
    data = pd.read_csv(csv_file)
except FileNotFoundError:
    print(f"Error: File '{csv_file}' not found.")
    sys.exit(1)

# Preprocess the data
data = preprocess_data(data)

# Split the data into features (X) and target (y)

X = data.drop(columns=['LABEL'])
y = data['LABEL'].map({'NO':0, 'YES': 1})

# Input shape is the number of features (in this case, 4)
print (X.shape[1])
input_shape = (X.shape[1],)

# Create the model
model = create_model(input_shape)

# Create a TensorBoard callback
tensorboard_callback = TensorBoard(log_dir="../logs")

# Train the model
history = model.fit(X, y, epochs=500, batch_size=64, validation_split=0.2, callbacks=[tensorboard_callback])

loss, accuracy = model.evaluate(X, y)

model.summary()
print_model_parameters(model)

print(f"Final Loss: {loss:.4f}")
print(f"Final Accuracy: {accuracy:.4f}")

# Save the trained model to a file in Keras format
model.save(model_file)

print(f"Model saved to '{model_file}'.")
