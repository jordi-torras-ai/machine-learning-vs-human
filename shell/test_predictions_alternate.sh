DATA="../csv/eada-human-vs-machine-dataset - alternate-test-data.csv"
MODEL=../models/eada-human-vs-machine-alternate.keras
python3 ../src/test_predictions.py "$DATA" "$MODEL"
