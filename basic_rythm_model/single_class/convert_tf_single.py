
from onnx_tf.backend import prepare
import onnx
import numpy as np


SAVE_NAME = "model_mobile_single"

onnx_model = onnx.load( SAVE_NAME + ".onnx")
tf_rep = prepare(onnx_model)
tf_rep.export_graph(SAVE_NAME + "_tf")
