import torch  # lub importuj klasę tak samo jak dla multi
import numpy as np
import coremltools as ct
from main_single import N_CLASSES , SEG_LEN , AGE_MEAN , SAVE_NAME
from main_single import SE_MobileNet1D_noLSTM

# === Wczytanie modelu ===
model = SE_MobileNet1D_noLSTM(num_classes=N_CLASSES)
model.load_state_dict(torch.load(SAVE_NAME + ".pt" , map_location="cpu"))
model.eval()

# === Przykładowe dane wejściowe ===
example_input = torch.randn(1, 1, SEG_LEN)
example_demo = torch.tensor([[AGE_MEAN, 1.0]])  # np. wiek i płeć

# === Trace modelu ===
traced = torch.jit.trace(model, (example_input, example_demo))

# === Konwersja do Core ML ===
mlmodel = ct.convert(
    traced,
    inputs=[
        ct.TensorType(name="ecg", shape=(1, 1, SEG_LEN), dtype=np.float32),
        ct.TensorType(name="demo", shape=(1, 2), dtype=np.float32),
    ],
    convert_to="mlprogram"
)

# === Zapis ===
mlmodel.save(SAVE_NAME + ".mlpackage")
