# Electricity Consumption Prediction

This project predicts electricity consumption using a GRU neural network and weather forecast data. The project is implemented in Python using the Keras library.


## Files

- `model_electricity_consumption.py`: Main script that includes data preprocessing, model training, and prediction.
- [sahko.csv]: Electricity consumption data.
- [Oulunsalo_ennuste_25.11.2024_cleaned.csv] Weather forecast data.
- [Oulunsalo_ennuste_25.11.2024.csv] This file contains temperature data based on the observation station and date. This file was obtained from the Finnish Meteorological Institute's website.
- [clean.py] Script used to clean and preprocess the weather forecast data.

## Model Architecture

The model uses a GRU neural network consisting of the following layers:
- Two GRU layers with 50 units and `tanh` activation function.
- Dropout layers to prevent overfitting.
- Dense layers with `relu` activation function.
- A final Dense layer with `linear` activation function to produce the prediction.

## Results

The model outputs the predicted electricity consumption in kWh based on the weather forecast file. However, the model is not yet reliable as the results still vary significantly. Further tuning and optimization of the model are needed to achieve more consistent and accurate predictions.
