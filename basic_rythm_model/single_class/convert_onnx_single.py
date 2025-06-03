
import torch
from main_single import N_CLASSES , SEG_LEN , AGE_MEAN , SAVE_NAME
from main_single import SE_MobileNet1D_noLSTM

OPSET_VERSION = 11


# Załaduj model
model = SE_MobileNet1D_noLSTM(num_classes=N_CLASSES)
model.load_state_dict(torch.load(SAVE_NAME + ".pt", map_location="cpu"))
model.eval()
# Przykładowe wejście
ecg_input = torch.randn(1, 1, SEG_LEN)
demo_input = torch.tensor([[AGE_MEAN, 1.0]])

# Eksport do ONNX
torch.onnx.export(
    model,
    (ecg_input, demo_input),
    SAVE_NAME + ".onnx",
    input_names=["ecg", "demo"],
    output_names=["output"],
    dynamic_axes={"ecg": {0: "batch"}, "demo": {0: "batch"}, "output": {0: "batch"}},
    opset_version=OPSET_VERSION
)

