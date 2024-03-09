# Machine Learning vs. Human

## Overview
Machine Learning vs. Human is an initiative delving into different machine learning algorithms and techniques to address classification challenges. The primary objective is to assess how machine learning models fare against human intuition in a given scenario.

## Challenge

To begin, you'll need to examine the Comma-Separated-Value (CSV) dataset available at:

https://github.com/jordi-torras-ai/machine-learning-vs-human/blob/main/csv/eada-human-vs-machine-dataset%20-%20static-training-data.csv

This CSV file comprises six columns:

- **id**: A numeric identifier ranging from 1 to 1000, serving solely for row identification.
- **COL1, COL2, COL3, COL4**: Each column contains an integer value ranging from 0 to 10.
- **LABEL**: This column contains either "YES" or "NO" labels.

Your task as a human participant is to deduce the underlying logic behind the assignment of "YES" or "NO" labels based on the values in the four columns. You can utilize spreadsheet software to explore various hypotheses, formulas, or calculations to discern this logic.

Subsequently, the project implements a basic neural network trained on the same dataset to create a predictive model for determining when a "YES" or "NO" label should be assigned.

To validate the effectiveness of the model, the dataset `eada-human-vs-machine-dataset - static-testing-data.csv` is employed to verify whether the neural network accurately captures the underlying logic.

In the code, the neural network is designed with a total of 19 neurons distributed across 4 layers. 

1. **Input Layer (Dense, 10 neurons, ReLU activation):** This is the first layer of the neural network. It consists of 10 neurons, each receiving input from the dataset. The ReLU (Rectified Linear Unit) activation function is applied to each neuron's output, ensuring that negative values are set to zero while positive values remain unchanged.

2. **Hidden Layer 1 (Dense, 5 neurons, ReLU activation):** The second layer is a hidden layer, meaning it is not directly connected to the input or output. It contains 5 neurons, each receiving input from the previous layer. Again, the ReLU activation function is applied to each neuron's output.

3. **Hidden Layer 2 (Dense, 3 neurons, ReLU activation):** Similar to the first hidden layer, this layer contains 3 neurons, each receiving input from the previous layer. ReLU activation is applied to each neuron's output.

4. **Output Layer (Dense, 1 neuron, Sigmoid activation):** This is the final layer of the neural network. It contains a single neuron, representing the output of the network. The Sigmoid activation function is applied to the neuron's output, squashing the value between 0 and 1. This is commonly used for binary classification tasks, where the output can be interpreted as the probability of a certain class (e.g., "YES" or "NO").

You can experiment with this neural network by adjusting the number of neurons, layers, and other parameters.

You can examine the creation of the neural network in the file `create_model_from_csv.py`.

```
    model = Sequential([
        Dense(10, activation='relu', input_shape=input_shape),
        Dense(5, activation='relu'),
        Dense(3, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
```
It's worth noting that in the majority of cases, the trained neural network achieves an accuracy rate of over 99% when predicting labels in the test dataset.

_Most of the code in this project was generated interactively with ChatGPT_.

## Requirements
To ensure the proper functioning of the application, you need to have the following installed on your platform:

- **Unix-like Command Line Interface**: If you are using a Unix-like operating system, such as Linux or macOS, you already have access to a command line interface. If not, you may need to install one. You can find more information on how to install and use Unix-like command line interfaces [here](https://www.tutorialspoint.com/unix/unix-getting-started.htm).

- **Python 3**: Make sure you have Python 3 installed on your system. You can download and install Python 3 from the official Python website [here](https://www.python.org/downloads/).

- **TensorFlow**: TensorFlow is a popular open-source machine learning framework developed by Google. You can install TensorFlow using Python's package manager, pip. Simply run the following command in your command line interface:

```bash
pip install tensorflow
```

For more information on TensorFlow and installation instructions, you can visit the TensorFlow website [here](https://www.tensorflow.org/install).

## Features

- **Utilizes TensorFlow**: The application harnesses the power of TensorFlow to construct a neural network model, which is then trained using the provided dataset.

- **Interactive Model Exploration**: Users have the option to interactively engage with the model via the command line interface. This feature allows users to observe the model's predictions for various inputs, providing valuable insight into its behavior.

- **Comprehensive Testing**: The application rigorously tests the trained model against a dedicated testing dataset. This extensive testing process enables users to evaluate the accuracy and reliability of the model across diverse scenarios.


## Folder Structure
```
.
├── README.md
├── csv
│   ├── eada-human-vs-machine-dataset - alternate-test-data.csv
│   ├── eada-human-vs-machine-dataset - alternate-training-data.csv
│   ├── eada-human-vs-machine-dataset - static-testing-data.csv
│   └── eada-human-vs-machine-dataset - static-training-data.csv
├── logs
├── models
│   ├── eada-human-vs-machine-alternate.keras
│   └── eada-human-vs-machine.keras
├── shell
│   ├── predict.sh
│   ├── remove_logs.sh
│   ├── show_model.sh
│   ├── test_predictions.sh
│   ├── test_predictions_alternate.sh
│   ├── train.sh
│   └── train_alternate.sh
└── src
    ├── create_model_from_csv.py
    ├── predict_from_model.py
    └── test_predictions.py

```

## Usage
1. Install [Python 3](python.org) and [TensorFlow](https://www.tensorflow.org)
2. Train a machine learning model 
```
cd shell
sh train.sh
```
After running this, the process should create the model and display some data:
```
Model Summary:
--------------------
Layer dense Dense
	Weights:		
		Weight 1:, -0.394, -0.631, 0.627, 0.230, -0.446, 0.698, 0.705, -0.523, -0.513, -0.607
		Weight 2:, -0.113, -0.736, 0.751, -0.802, -0.497, 0.559, 0.485, -0.491, 0.131, -0.607
		Weight 3:, -0.629, -0.558, 0.546, 0.671, -0.413, 0.755, 0.446, -0.758, -0.146, -0.488
		Weight 4:, 0.125, -0.587, 1.087, 0.212, -0.661, -0.140, 0.483, -0.620, 0.491, -0.358
	Biases:, 0.385, 0.169, 0.423, 0.103, -0.053, -0.197, -0.166, 0.253, -0.232, 0.135
Layer dense_1 Dense
	Weights:		
		Weight 1:, 0.493, 0.969, 0.462, -0.204, -0.298
		Weight 2:, 1.113, 1.433, -0.510, -0.450, -0.115
		Weight 3:, -0.846, -1.493, 0.280, 1.747, -0.119
		Weight 4:, -0.091, 0.021, 0.479, 0.471, 0.160
		Weight 5:, 0.080, 0.971, -0.129, -0.643, -0.186
		Weight 6:, -0.139, -0.672, -0.857, 0.384, 0.353
		Weight 7:, 0.009, 0.246, -0.472, 0.826, -0.579
		Weight 8:, 0.668, 0.547, -0.097, -0.096, -0.383
		Weight 9:, 0.088, 0.337, -0.745, -0.020, 0.659
		Weight 10:, 1.080, 0.704, 0.199, -0.985, -0.131
	Biases:, -0.095, -0.023, 0.307, -0.031, -0.120
Layer dense_2 Dense
	Weights:		
		Weight 1:, 0.643, -0.852, 1.418
		Weight 2:, 1.742, -0.023, 1.085
		Weight 3:, 0.163, -2.348, 0.136
		Weight 4:, -1.259, 0.448, -1.215
		Weight 5:, -0.595, 0.559, 0.334
	Biases:, -0.024, -1.138, -0.076
Layer dense_3 Dense
	Weights:		
		Weight 1:, 2.866
		Weight 2:, 2.864
		Weight 3:, 2.178
	Biases:, -2.781
Final Loss: 0.0542
Final Accuracy: 0.9950
Model saved to '../models/eada-human-vs-machine.keras'.

real	0m7.996s
user	0m10.645s
sys	0m1.057s
```

The Model Summary displays the weights of every neuron in each layer, along with the biases in every layer (note that your environment will show different numbers). It also provides information on the loss and accuracy metrics. Additionally, the location where the model `eada-human-vs-machine.keras` has been saved is indicated.

3. Explore the model
To assess the model and evaluate its learning process, execute:
```
cd shell
sh show_model.sh
```
To monitor the training process and explore the model's features using TensorBoard, simply open your web browser and navigate to http://localhost:6006. To stop the web server, press `Ctrl+C` on your command line interface.
```
Open http://localhost:6006 in browser
Serving TensorBoard on localhost; to expose to the network, use a proxy or pass --bind_all
TensorBoard 2.15.2 at http://localhost:6006/ (Press CTRL+C to quit)
^C 
```

4. Massively check the quality of the model. 
Check how good is the model using the testing data set `eada-human-vs-machine-dataset - static-testing-data.csv`
```
cd shell
sh test_predictions.sh
````
You should see an outcome similar to this:
```
sh test_predictions.sh 
32/32 [==============================] - 0s 333us/step
Total predictions: 1000
Correct predictions: 991
Incorrect predictions: 9
Accuracy: 99.10%
Expected: YES, Predicted: NO, Probability: 0.1737, Values: id: 71, COL1: 8, COL2: 4, COL3: 10, COL4: 10
Expected: YES, Predicted: NO, Probability: 0.3539, Values: id: 154, COL1: 9, COL2: 5, COL3: 8, COL4: 10
Expected: YES, Predicted: NO, Probability: 0.4925, Values: id: 295, COL1: 8, COL2: 9, COL3: 5, COL4: 9
Expected: YES, Predicted: NO, Probability: 0.3586, Values: id: 352, COL1: 9, COL2: 10, COL3: 10, COL4: 2
Expected: YES, Predicted: NO, Probability: 0.2324, Values: id: 459, COL1: 9, COL2: 6, COL3: 6, COL4: 10
Expected: YES, Predicted: NO, Probability: 0.4327, Values: id: 598, COL1: 9, COL2: 9, COL3: 9, COL4: 4
Expected: YES, Predicted: NO, Probability: 0.2706, Values: id: 749, COL1: 10, COL2: 4, COL3: 8, COL4: 9
Expected: YES, Predicted: NO, Probability: 0.3936, Values: id: 772, COL1: 7, COL2: 10, COL3: 10, COL4: 4
Expected: YES, Predicted: NO, Probability: 0.4582, Values: id: 904
```
It will display the cases in the dataset (which contains 1000 rows) where the predictions were incorrect.

5. Check individual cases
You can check specific examples by entering the data for columns C1 through C4, separated by commas. The program will continue to prompt for examples until you exit with `Ctrl-d`.
```
cd shell
sh predict.sh
```
An example of execution would be: 
```
Enter values for COL1 to COL4 (comma-separated): 1,2,3,4
1/1 [==============================] - 0s 48ms/step
Predicted Label: YES
Probability: 1.0000
Predicted Value: 1.0000
Enter values for COL1 to COL4 (comma-separated): 
```
## Alternative Challenge

An alternative dataset is provided [here](https://github.com/jordi-torras-ai/machine-learning-vs-human/blob/main/csv/eada-human-vs-machine-dataset%20-%alternate-training-data.csv).

If you've successfully tackled the initial challenge, you may find this alternative challenge considerably easier.

However, you'll notice that machine learning algorithms struggle to create an effective model for predicting the YES/NO label based on this data.

To observe how the same neural network performs with this alternative data:

```bash
cd shell
sh train_alternate.sh
sh test_predictions.sh
```

Upon running these commands, you'll observe that the model correctly approximately 50% of the cases, resembling random guessing for YES/NO labels.

## Contributing
Contributions are welcome! If you'd like to contribute to this project, please fork the repository, make your changes, and submit a pull request.

## License
This project is licensed under the terms of the [GNU General Public License v3.0](https://www.gnu.org/licenses/gpl-3.0.html).

You are free to:

- Share — copy and redistribute the material in any medium or format.
- Adapt — remix, transform, and build upon the material for any purpose, even commercially.

Under the following terms:

- Attribution — You must give appropriate credit, provide a link to the license, and indicate if changes were made. You may do so in any reasonable manner, but not in any way that suggests the licensor endorses you or your use.
- ShareAlike — If you remix, transform, or build upon the material, you must distribute your contributions under the same license as the original.
- No additional restrictions — You may not apply legal terms or technological measures that legally restrict others from doing anything the license permits.

For more details, see the [GNU General Public License v3.0](https://www.gnu.org/licenses/gpl-3.0.html) documentation.
## Contact

For any questions or suggestions, feel free to contact [Jordi Torras](https://www.torras.ai).

