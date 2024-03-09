# Machine Learning vs. Human

## Overview
Machine Learning vs. Human is a project that explores various machine learning algorithms and techniques to solve classification problems. 
It aims to compare the performance of machine learning models against human model guessing in a specific example.

## Challenge

First you need to explore the Comma-Separated-Value dataset:

https://github.com/jordi-torras-ai/machine-learning-vs-human/blob/main/csv/eada-human-vs-machine-dataset%20-%20static-training-data.csv

The CSV file has 4 columns: 
. id : is a numeric id, simply goes from 1 to 1000, and does not have any purpose, other than identify a row
. COL1, COL2, COL3, COL4: each column contains a integer number that ranges from 0 to 10, 
. LABEL: the label has either "YES" or "NO"

Now, as a human, your mission is to guess what is the logic of the YES/NO label so you are able to predict it. 
You can upload the file into a spreadsheet program, and try different hypothesis, formulas or calculations in order to guess what is the logic that the 4 columns produce a YES or a NO (there is a logic).

The rest of the project implements a simple neural network that learns from the same file, and creates a model that tries to predict when a YES or NO should be given. 

To check if the model works, the dataset `eada-human-vs-machine-dataset - static-testing-data.csv` is used to check if the neural network got the logic correctly.  
In most of the cases, the trained neural network predicts correctly more than 99% of the cases in the test dataset. 

## Requirements
For the application to work you need installed in your platform: 
- Unix-like command line interface
- Python 3
- TensorFlow

## Features
- uses TensorFlow to create a neural network that is trained based on the dataset 
- you can interactively play with the model on the command line to see how predicts labels
- massively test the model against a testing data set to see the accuracy

## Folder Structure
```
├── csv
│   ├── eada-human-vs-machine-dataset - static-testing-data.csv
│   └── eada-human-vs-machine-dataset - static-training-data.csv
├── logs
│   ├── train
│   │   └── events.out.tfevents.1709915622.Jordis-Air.attlocal.net.9543.0.v2
│   └── validation
│       └── events.out.tfevents.1709915622.Jordis-Air.attlocal.net.9543.1.v2
├── models
│   └── eada-human-vs-machine.keras
├── shell
│   ├── predict.sh
│   ├── show_model.sh
│   ├── test_predictions.sh
│   └── train.sh
└── src
    ├── create_model_from_csv.py
    ├── predict_from_model.py
    └── test_predictions.py
```

## Usage
1. Install Python 3 and TensorFlow
2. Train a machine learning model 
```
cd shell
sh train.sh
```
the result of that should create the model and diplay some data:
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

The Model Summary shows the the weights of every neuron of every layer as well as the biases in every layer (your environment will show different numbers, of course).
It tells the loss and the accuracy. 
Where the model `eada-human-vs-machine.keras` has been saved.

3. Explore the model
To check the model and how it has learned.
```
cd shell
sh show_model.sh
```
On a browser, you can open the URL http://localhost:6006 and check the training process and the features of the model with TensorBoard. 
To finish the webserver press `Ctrl+C`
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
It will show the cases in the dataset (that contains 1000 rows) that did not predict the data correctly. 

5. Check individual cases
You can check specific examples, entergin the data for C1..C4 columns, separated by comma. 
It keeps asking for examples, you can leave with `Ctrl-d`.
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

