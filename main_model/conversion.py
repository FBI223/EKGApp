from onnx_coreml import convert

coreml_model = convert(model="model.onnx", minimum_ios_deployment_target="13")
coreml_model.save("model.mlmodel")
print("✅ Gotowe! model.mlmodel zapisany.")
