import sys
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model

def preprocess_input(data):
    # remove "id"
    data = data.drop(columns=['id'])
    # Convert 'LABEL' column to numerical values
    data['LABEL'] = data['LABEL'].map({'NO': 0, 'YES': 1})

    # Initialize an empty DataFrame to store the scaled values
    data_scaled = pd.DataFrame()

    # Iterate over each column in the input DataFrame (excluding 'LABEL')
    for col in data.columns[:-1]:
        # Convert the column to numeric, coercing errors to NaN
        col_numeric = pd.to_numeric(data[col], errors='coerce')
        
        # If the column contains non-numeric values, print a warning
        if col_numeric.isna().any():
            print(f"Warning: Non-numeric values found in column '{col}'.")

        # Scale the numeric values to the range [-1, 1]
        col_scaled = (col_numeric - 5.0) / 5.0

        # Add the scaled column to the output DataFrame
        data_scaled[col] = col_scaled
    
    # Add the transformed 'LABEL' column to the output DataFrame
    data_scaled['LABEL'] = data['LABEL']

    return data_scaled

# Check if the CSV file and model file names are provided as command-line arguments
if len(sys.argv) != 3:
    print("Usage: python validate_model.py <csv_file> <model_file>")
    sys.exit(1)

# Extract the CSV file and model file names from command-line arguments
csv_file = sys.argv[1]
model_file = sys.argv[2]

# Load the saved model
try:
    model = load_model(model_file)
except FileNotFoundError:
    print(f"Error: File '{model_file}' not found.")
    sys.exit(1)

# Load the testing data from the CSV file
try:
    test_data = pd.read_csv(csv_file)
except FileNotFoundError:
    print(f"Error: File '{csv_file}' not found.")
    sys.exit(1)

# Preprocess the testing data
test_data_scaled = preprocess_input(test_data)

# Split the testing data into features (X_test) and labels (y_test)
X_test = test_data_scaled.drop(columns=['LABEL'])
y_test = test_data_scaled['LABEL']

# Make predictions using the model
predictions = model.predict(X_test)

# Convert predictions to binary values (0 or 1)
predicted_labels = np.where(predictions >= 0.5, 1, 0)

# Compare predicted labels with actual labels and compute accuracy
correct_predictions = np.sum(predicted_labels.flatten() == y_test.to_numpy())
total_predictions = len(y_test)
accuracy = correct_predictions / total_predictions * 100


# Show details of incorrect predictions
incorrect_indices = np.where(predicted_labels.flatten() != y_test.to_numpy())[0]
for idx in incorrect_indices:
    expected_label = "YES" if y_test.iloc[idx] == 1 else "NO"
    predicted_label = "YES" if predicted_labels.flatten()[idx] == 1 else "NO"
    probability = predictions[idx][0]
    cols_values = ", ".join([f"{col}: {test_data.iloc[idx][col]}" for col in test_data.columns[:-1]])
    print(f"Expected: {expected_label}, Predicted: {predicted_label}, Probability: {probability:.4f}, Values: {cols_values}")

# Display results
print(f"Total predictions: {total_predictions}")
print(f"Correct predictions: {correct_predictions}")
print(f"Incorrect predictions: {total_predictions - correct_predictions}")
print(f"Accuracy: {accuracy:.2f}%")
