import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt
from PIL import Image
from torch.utils.data import Dataset
import struct
import numpy as np



# Ustawienia
BATCH_SIZE = 64
EPOCHS = 5
LEARNING_RATE = 0.001
MODEL_PATH = "mnist_model.pt"
ONNX_PATH = "mnist_model.onnx"






def load_mnist_images(filename):
    with open(filename, 'rb') as f:
        magic, num_images, rows, cols = struct.unpack(">IIII", f.read(16))
        assert magic == 2051
        images = np.frombuffer(f.read(), dtype=np.uint8).reshape(num_images, 28, 28)
        return images

def load_mnist_labels(filename):
    with open(filename, 'rb') as f:
        magic, num_labels = struct.unpack(">II", f.read(8))
        assert magic == 2049
        labels = np.frombuffer(f.read(), dtype=np.uint8)
        return labels




train_images_path = "databases/mnist/train-images.idx3-ubyte"
train_labels_path = "databases/mnist/train-labels.idx1-ubyte"
test_images_path = "databases/mnist/t10k-images.idx3-ubyte"
test_labels_path = "databases/mnist/t10k-labels.idx1-ubyte"




class CustomMNISTDataset(Dataset):
    def __init__(self, images_path, labels_path, transform=None):
        self.images = load_mnist_images(images_path)
        self.labels = load_mnist_labels(labels_path)
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img = Image.fromarray(self.images[idx].astype(np.uint8), mode='L')  # 👈 poprawiona linia
        label = self.labels[idx]
        if self.transform:
            img = self.transform(img)
        return img, label


# 1. Transformacje i dane


transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])



train_dataset = CustomMNISTDataset(train_images_path, train_labels_path, transform=transform)
test_dataset = CustomMNISTDataset(test_images_path, test_labels_path, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)



sample_img, label = train_dataset[0]
print(f"Label: {label}")
plt.imshow(sample_img.squeeze(), cmap='gray')
plt.show()

# 2. Model (MLP)
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28*28, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        return self.model(x)

model = MLP()

# 3. Trenowanie
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

print("Trenowanie...")
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        output = model(data)
        loss = loss_fn(output, target)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    print(f"Epoka {epoch+1}/{EPOCHS}, Strata: {avg_loss:.4f}")

# 4. Zapis modelu w PyTorch
torch.save(model.state_dict(), MODEL_PATH)
print(f"Model zapisany jako: {MODEL_PATH}")

# 5. Eksport do ONNX
model.eval()
dummy_input = torch.randn(1, 1, 28, 28, device=device)
torch.onnx.export(
    model,
    dummy_input,
    ONNX_PATH,
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
    opset_version=11
)
print(f"Model zapisany w ONNX jako: {ONNX_PATH}")
