"""
SmartCity Traffic Optimization - Traffic Flow Predictor Prototype
Framework: TensorFlow / Keras
"""
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def generate_synthetic_traffic_data(samples=1000):
    """
    Simula telemetría de tráfico:
    - Entrada (X): [hora_del_día (0-23), día_semana (0-6), condición_clima (0: soleado, 1: lluvia)]
    - Salida (y): volumen vehicular estimado (vehículos/hora)
    """
    np.random.seed(42)
    tf.random.set_seed(42)

    hour = np.random.randint(0, 24, size=(samples, 1))
    day = np.random.randint(0, 7, size=(samples, 1))
    rain = np.random.binomial(1, 0.2, size=(samples, 1))

    # Relación de prueba: horas pico (8h y 18h) + lluvia aumentan el flujo
    traffic = (
        150 +
        300 * np.exp(-((hour - 8) ** 2) / 8) +
        350 * np.exp(-((hour - 18) ** 2) / 8) +
        100 * rain +
        np.random.normal(0, 25, size=(samples, 1))
    )

    X = np.hstack([hour / 23.0, day / 6.0, rain])  # Normalización básica
    y = traffic
    return X, y

def build_traffic_model():
    """Construye una red neuronal densa adecuada para inferencia rápida."""
    model = keras.Sequential([
        layers.Input(shape=(3,)),
        layers.Dense(32, activation="relu"),
        layers.Dense(16, activation="relu"),
        layers.Dense(1)  # Salida: volumen vehicular continuo
    ])

    model.compile(
        optimizer="adam",
        loss="mse",
        metrics=["mae"]
    )
    return model

if __name__ == "__main__":
    print(f"Versión de TensorFlow: {tf.__version__}")
    print("Generando datos de telemetría vehicular...")
    X, y = generate_synthetic_traffic_data()

    print("Entrenando modelo de optimización de tráfico...")
    model = build_traffic_model()
    model.fit(X, y, epochs=15, batch_size=32, verbose=1)

    # Inferencia de prueba (Hora pico 18:00, día laborable, con lluvia)
    test_input = np.array([[18.0 / 23.0, 2.0 / 6.0, 1.0]])
    predicted_volume = model.predict(test_input)[0][0]
    print(f"\nPredicción de flujo vehicular (18:00h con lluvia): {predicted_volume:.1f} vehículos/hora")