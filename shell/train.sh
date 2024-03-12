DATA="../csv/eada-human-vs-machine-dataset - static-training-data.csv"
MODEL=../models/eada-human-vs-machine.keras
LOGS=../logs/
rm -rf $LOGS
mkdir $LOGS
time python3 ../src/create_model_from_csv.py "$DATA" "$MODEL"
