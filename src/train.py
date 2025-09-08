import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

from preprocess import load_data, preprocess_data
from model import build_lstm

def train():
    df = load_data("data/fpt.csv")
    X, y, scaler = preprocess_data(df)

    # train-test split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )

    model = build_lstm((X_train.shape[1], 1))

    # callbacks
    checkpoint = ModelCheckpoint("models/best_model.h5", save_best_only=True)
    early_stop = EarlyStopping(monitor="val_loss", patience=5)

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=50,
        batch_size=32,
        callbacks=[checkpoint, early_stop]
    )

    # plot loss
    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='val')
    plt.legend()
    plt.title("Loss curve")
    plt.savefig("results/loss_curve.png")
    plt.close()

if __name__ == "__main__":
    train()
