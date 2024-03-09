DATA="../csv/eada-human-vs-machine-dataset - alternate-training-data.csv"
MODEL=../models/eada-human-vs-machine-alternate.keras
LOGS=../logs/
rm -rf $LOGS
mkdir $LOGS
time python3 ../src/create_model_from_csv.py "$DATA" "$MODEL"
