from onnx_coreml import convert

coreml_model = convert(model="training/onnx/model_fold1.onnx", minimum_ios_deployment_target="13")
coreml_model.save("model_fold1.mlmodel")
print("✅ Gotowe! model.mlmodel zapisany.")
