import onnx
import tensorflow as tf
from onnx_tf.backend import prepare

onnx_model = onnx.load("training/onnx/model_fold3.onnx")
tf_rep = prepare(onnx_model)
tf_rep.export_graph("model_tf_fold3")
print("✅ SavedModel zapisany → model_tf_fold3/")

converter = tf.lite.TFLiteConverter.from_saved_model("training/tflite/model_tf_fold3")
tflite_model = converter.convert()
with open("training/tflite/model_fold3.tflite", "wb") as f:
    f.write(tflite_model)
print("✅ Gotowe! model_fold1.tflite zapisany")
