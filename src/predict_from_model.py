import sys
import numpy as np
from tensorflow.keras.models import load_model

# Check if the model file name is provided as a command-line argument
if len(sys.argv) != 2:
    print("Usage: python predict_from_model.py <model_file>")
    sys.exit(1)

# Extract the model file name from command-line arguments
model_file = sys.argv[1]

# Load the saved model
try:
    model = load_model(model_file)
except FileNotFoundError:
    print(f"Error: File '{model_file}' not found.")
    sys.exit(1)

# Function to preprocess user input string
def preprocess_input(user_input):
    try:
        values = [float(x) for x in user_input.split(',')]
        if len(values) != 4:
            raise ValueError
        # Adding a placeholder value (0) for the label
        values = [(x - 5.0) / 5.0 for x in values]
        # values.append(0)  # Placeholder for label
        return np.array(values).reshape(1, -1)
    except ValueError:
        print("Error: Invalid input format. Please enter comma-separated values for COL1 to COL4.")
        return None

# Loop to continuously prompt the user for input
while True:
    try:
        # Prompt the user to input values for COL1 to COL4
        user_input = input("Enter values for COL1 to COL4 (comma-separated): ")

        # Preprocess the user input
        X_input = preprocess_input(user_input)

        # If input is valid, make predictions
        if X_input is not None:
            # Predict the label and probability
            prediction = model.predict(X_input)
            label = "YES" if prediction[0][0] >= 0.5 else "NO"
            probability = prediction[0][0]

            # Display the predicted label, probability, and predicted value
            predicted_value = prediction[0][0]
            print(f"Predicted Label: {label}")
            print(f"Probability: {probability:.4f}")
            print(f"Predicted Value: {predicted_value:.4f}")

    except EOFError:
        print("\nExiting...")
        break
