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
You can upload the file into a spreadsheet program, and try different formulas or calculations in order to guess what is the logic that the 4 columns produce a YES or a NO (there is a logic).

The rest of the project implements a simple neural network that learns from the same file, and creates a model that tries to predict when a YES or NO should be given. 

To check if the model works, the dataset `eada-human-vs-machine-dataset - static-testing-data.csv` is used to check if the neural network got the logic correctly.  
In most of the cases, the trained neural network predicts correctly more than 99.5% of the cases in the test dataset. 


## Features
- Training and evaluation of machine learning models using TensorFlow/Keras.
- Comparison of model predictions with human decisions.
- Visualization of model performance using TensorBoard.
- Data preprocessing and analysis.

## Folder Structure
```
.
.
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
1. Install the required dependencies listed in `requirements.txt`.
2. Train a machine learning model using `train_model.py`.
3. Evaluate model predictions using `test_predictions.py`.
4. View TensorBoard logs for model performance visualization.

## Contributing
Contributions are welcome! If you'd like to contribute to this project, please fork the repository, make your changes, and submit a pull request.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Contact
For any questions or suggestions, feel free to contact [Your Name](mailto:your.email@example.com).

---

You can customize the sections and content of the README file based on your project's specifics. Make sure to replace placeholders like "Your Name" and "your.email@example.com" with your actual information.
