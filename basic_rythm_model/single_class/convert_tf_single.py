from onnx_tf.backend import prepare
import onnx

SAVE_NAME = "model_mobile_single"

# Załaduj model ONNX
onnx_model = onnx.load(SAVE_NAME + ".onnx")

# Przygotuj reprezentację TensorFlow (upewnij się, że używa float32 tylko)
tf_rep = prepare(onnx_model)

# Eksportuj jako SavedModel
tf_rep.export_graph(SAVE_NAME + "_tf")
