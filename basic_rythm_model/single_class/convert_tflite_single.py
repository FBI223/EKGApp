import tensorflow as tf

SAVE_NAME = "model_mobile_single"

converter = tf.lite.TFLiteConverter.from_saved_model("model_no_lstm_final/" + SAVE_NAME + "_tf")

# Make sure only supported ops and types are used (mostly float ops)
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]

# ensure inference input/output types are float32
converter.inference_input_type = tf.float32
converter.inference_output_type = tf.float32

tflite_model = converter.convert()

with open(SAVE_NAME + ".tflite", "wb") as f:
    f.write(tflite_model)

print("✅ Konwersja do model.tflite zakończona")
