from google.colab import drive
import shutil
import os

# Zamontuj Google Drive
drive.mount('/content/drive', force_remount=True)

# Ścieżki do plików w Google Drive
drive_folder = "/content/drive/MyDrive/ECG"
local_folder = "/content/"

# Upewnij się, że folder istnieje
os.makedirs(local_folder, exist_ok=True)

# Skopiuj pliki z Google Drive do lokalnego środowiska Colab
shutil.copy(os.path.join(drive_folder, "X.npy"), os.path.join(local_folder, "X.npy"))
shutil.copy(os.path.join(drive_folder, "y.npy"), os.path.join(local_folder, "y.npy"))

print("✅ Pliki zostały skopiowane do /content/")