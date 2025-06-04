import tensorflow as tf
import numpy as np

interpreter = tf.lite.Interpreter(model_path="model_mobile_single.tflite")
interpreter.allocate_tensors()

# Zbierz szczegóły
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
tensor_details = interpreter.get_tensor_details()

print("📥 INPUTS:")
for i in input_details:
    print(f"{i['name']}: {i['dtype']} {i['shape']}")

print("\n📤 OUTPUTS:")
for o in output_details:
    print(f"{o['name']}: {o['dtype']} {o['shape']}")

print("\n🔎 ALL TENSORS (podgląd typu danych):")
for t in tensor_details:
    print(f"{t['name']:<30} | dtype: {t['dtype']} | shape: {t['shape']}")












# Wczytaj i zainicjalizuj interpreter
interpreter = tf.lite.Interpreter(model_path="model_mobile_single.tflite")
interpreter.allocate_tensors()

# Pobierz wszystkie tensory
tensor_details = interpreter.get_tensor_details()

# Filtruj tylko te typu int64
int64_tensors = [t for t in tensor_details if t['dtype'] == tf.int64]

# Wypisz je
for t in int64_tensors:
    print(f"🔴 Tensor: {t['name']} | shape: {t['shape']} | dtype: int64")

