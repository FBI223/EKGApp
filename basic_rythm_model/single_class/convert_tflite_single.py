
import tensorflow as tf


SAVE_NAME = "model_mobile_single.pt"

# Załaduj zapisany model
converter = tf.lite.TFLiteConverter.from_saved_model( SAVE_NAME + "_tf")

# Konwertuj do TFLite (float32)
tflite_model = converter.convert()

# Zapisz do pliku
with open( SAVE_NAME + ".tflite", "wb") as f:
    f.write(tflite_model)

print("✅ Konwersja do model.tflite zakończona")