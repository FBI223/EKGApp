import tensorflow as tf

SAVE_NAME = "model_mobile_single"

# Ścieżka do zapisanej reprezentacji TF
saved_model_path = SAVE_NAME + "_tf"

# === Konwerter
converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_path)

# ✅ Wymuś tylko operacje dostępne w TFLite (bez TF Select/Unsupported)
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]

# ✅ Wymuś typy wejścia/wyjścia na float32
converter.inference_input_type = tf.float32
converter.inference_output_type = tf.float32

# === Konwertuj
tflite_model = converter.convert()

# === Zapisz model TFLite
with open(SAVE_NAME + ".tflite", "wb") as f:
    f.write(tflite_model)

print("✅ Konwersja do model.tflite zakończona")


















# TESTING

interpreter = tf.lite.Interpreter(model_path=SAVE_NAME + ".tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print("Input types:")
for i in input_details:
    print(i["name"], i["dtype"])

print("Output types:")
for o in output_details:
    print(o["name"], o["dtype"])
