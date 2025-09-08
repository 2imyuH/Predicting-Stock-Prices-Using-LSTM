import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.models import load_model

from preprocess import load_data, preprocess_data

def evaluate():
    df = load_data("data/fpt.csv")
    X, y, scaler = preprocess_data(df)

    model = load_model("model/model_fpt.keras")
    predictions = model.predict(X)
    predictions = scaler.inverse_transform(predictions)
    y_true = scaler.inverse_transform(y.reshape(-1, 1))

    # plot results
    plt.figure(figsize=(10,6))
    plt.plot(y_true, label="Actual")
    plt.plot(predictions, label="Predicted")
    plt.legend()
    plt.title("Stock Price Prediction")
    plt.savefig("results/prediction_vs_actual.png")
    plt.close()

if __name__ == "__main__":
    evaluate()
