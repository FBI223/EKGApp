import tensorflow as tf

# Załaduj zapisany model
converter = tf.lite.TFLiteConverter.from_saved_model("model_tf")

# Konwertuj do TFLite (float32)
tflite_model = converter.convert()

# Zapisz do pliku
with open("model 3 no st  no georgia/model.tflite", "wb") as f:
    f.write(tflite_model)

print("✅ Konwersja do model.tflite zakończona")