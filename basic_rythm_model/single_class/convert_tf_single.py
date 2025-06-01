
from onnx_tf.backend import prepare
import onnx


onnx_model = onnx.load("model 3 no st  no georgia/model.onnx")
tf_rep = prepare(onnx_model)
tf_rep.export_graph("model_tf")
