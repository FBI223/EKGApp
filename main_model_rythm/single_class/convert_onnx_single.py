import torch
from main_single import N_CLASSES, SEG_LEN, AGE_MEAN, SAVE_NAME
from main_single import SE_MobileNet1D_noLSTM

OPSET_VERSION = 11

# === [1] Wymuszamy float32 również dla ecg_input ===
ecg_input = torch.randn(1, 1, SEG_LEN, dtype=torch.float32)  # ✅ zmiana typów
demo_input = torch.tensor([[AGE_MEAN, 1.0]], dtype=torch.float32)  # ✅ zawsze float32

# === [2] Załaduj model ===
model = SE_MobileNet1D_noLSTM(num_classes=N_CLASSES)
model.load_state_dict(torch.load(SAVE_NAME + ".pt", map_location="cpu"))
model.eval()

# === [3] (Opcjonalnie) Dodaj wymuszenie float32 w forward(), jeśli nie masz ===
# Dodaj w metodzie `forward()` modelu:
# demo = demo.float()

# === [4] Eksport do ONNX ===
torch.onnx.export(
    model,
    (ecg_input, demo_input),
    SAVE_NAME + ".onnx",
    input_names=["ecg", "demo"],
    output_names=["output"],
    opset_version=11  # <= OK
)
