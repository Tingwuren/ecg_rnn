import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense, Dropout

def build_rnn_model(timesteps=200, num_features=1, num_classes=2):
    """
    构建 Vanilla RNN 模型。
    """
    model = Sequential([
        SimpleRNN(64, activation='tanh', input_shape=(timesteps, num_features)),
        Dropout(0.2),
        Dense(num_classes, activation='softmax')
    ])

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

if __name__ == "__main__":
    model = build_rnn_model()
    model.summary()