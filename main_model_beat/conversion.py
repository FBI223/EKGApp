from onnx_coreml import convert

coreml_model = convert(model="best_model.onnx", minimum_ios_deployment_target="13")
coreml_model.save("best_model.mlmodel")
print("✅ Gotowe! model.mlmodel zapisany.")
