import onnx

model = onnx.load("model_mobile_single.onnx")
for tensor in model.graph.initializer:
    if tensor.data_type == onnx.TensorProto.INT64:
        tensor.data_type = onnx.TensorProto.INT32

for node in model.graph.node:
    for attr in node.attribute:
        if attr.type == onnx.AttributeProto.INT:
            attr.i = int(attr.i)

onnx.save(model, "model.onnx")