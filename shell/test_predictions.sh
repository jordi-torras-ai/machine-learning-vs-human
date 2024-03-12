DATA="../csv/eada-human-vs-machine-dataset - static-testing-data.csv"
MODEL=../models/eada-human-vs-machine.keras
python3 ../src/test_predictions.py "$DATA" "$MODEL"
