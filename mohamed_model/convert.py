import tensorflow as tf
import coremltools as ct

model = tf.keras.models.load_model("cnn_ecg_model.keras")

mlmodel = ct.convert(
    model,
    source="tensorflow",
    convert_to="mlprogram",
    minimum_deployment_target=ct.target.iOS15,
    compute_units=ct.ComputeUnit.ALL
)

mlmodel.save("cnn_ecg_model.mlmodel")



import tf2onnx
import tensorflow as tf

model = tf.keras.models.load_model("cnn_ecg_model.keras")

spec = (tf.TensorSpec((None, 188, 1), tf.float32, name="input"),)
onnx_model, _ = tf2onnx.convert.from_keras(model, input_signature=spec, opset=13)

with open("cnn_ecg_model.onnx", "wb") as f:
    f.write(onnx_model.SerializeToString())



