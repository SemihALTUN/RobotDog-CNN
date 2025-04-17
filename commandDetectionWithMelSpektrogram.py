import os
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.utils import to_categorical
from tqdm import tqdm
from tensorflow.keras.layers import Dropout
from PIL import Image


# 1. Veri Seti Hazırlığı
data_dir = r"datasetcommand"  # Veri seti klasörü
spectrogram_dir = "spectrograms"  # Kaydedilecek mel-spektrogramlar


def add_noise(data, noise_factor=0.1):
    """Ses verisine gürültü ekler."""
    noise = noise_factor * np.random.randn(len(data))
    augmented_data = data + noise
    return np.clip(augmented_data, -1.0, 1.0)

def adjust_pitch(data, sr, n_steps=2):
    """Ses verisinin pitch'ini değiştirir."""
    return librosa.effects.pitch_shift(data, sr=sr, n_steps=n_steps)

def adjust_speed(data, speed_factor=1.2):
    """Ses hızını değiştirir."""
    return librosa.effects.time_stretch(data, rate=speed_factor)

def create_spectrogram(file_path, save_path, augmentations=None):
    """Ses dosyasından mel-spektrogram çıkar ve kaydet."""
    if not os.path.exists(save_path):  # Eğer spektrogram daha önce oluşturulmamışsa
        y, sr = librosa.load(file_path, duration=10)

        # Uygulanacak augmentations listesi
        if augmentations:
            for augment in augmentations:
                if callable(augment):
                    if 'sr' in augment.__code__.co_varnames:
                        y = augment(y, sr)
                    else:
                        y = augment(y)

        spectrogram = librosa.feature.melspectrogram(y=y, sr=sr)
        plt.figure(figsize=(2.56, 2.56))  # 128x128 pixel görüntü
        librosa.display.specshow(librosa.power_to_db(spectrogram, ref=np.max),
                                 y_axis='mel', fmax=8000, x_axis='time')
        plt.axis('off')
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
        plt.close()

os.makedirs(spectrogram_dir, exist_ok=True)

augmentations = {
    "normal": [],
    "noise": [add_noise], #arka plan gürültü
    "pitch": [lambda y, sr: adjust_pitch(y, sr, n_steps=2)], #ses tonu
    "speed": [lambda y: adjust_speed(y, speed_factor=1.2)] #ses hızı
}

data = [] #mel spektrogram dosya yolu
command_labels_list = []
command_labels = {command: idx for idx, command in enumerate(os.listdir(data_dir))} #komut etiketi

#mel-spektrogram oluştur
for komut in tqdm(os.listdir(data_dir), desc="Processing Command"): #her bir komut dosyasını gezme
    command_label = command_labels[komut] #oluşturulan etiketler ile komut isimlerini eşleme
    command_dir = os.path.join(data_dir, komut)
    for file in tqdm(os.listdir(command_dir), desc=f"Processing {komut}", leave=False): #komut klasöründeki komutları listeler
        file_path = os.path.join(command_dir, file) #komuta ait dosya yolu oluşturur
        for aug_type, aug_list in augmentations.items():
            spectrogram_path = os.path.join(spectrogram_dir, f"{komut}_{file}_{aug_type}.png")
            create_spectrogram(file_path, spectrogram_path, augmentations=aug_list)
            data.append(spectrogram_path)
            command_labels_list.append(command_label)

# 2. Veriyi Bölme
X_train, X_test, y_train, y_test = train_test_split(
    data, command_labels_list, test_size=0.2, random_state=42
)

# Görüntüleri yükleme ve normalizasyon
def load_image(image_path):
    """Mel-spektrogram görüntüsünü yükler ve normalize eder."""

    img = Image.open(image_path).convert('L')  # Gri tonlamaya çevir
    img = img.resize((128, 128))  # 128x128 boyutuna çevir
    return np.array(img) / 255.0

X_train = np.array([load_image(path) for path in X_train]).reshape(-1, 128, 128, 1)
X_test = np.array([load_image(path) for path in X_test]).reshape(-1, 128, 128, 1)
y_train = to_categorical(y_train, num_classes=len(command_labels))
y_test = to_categorical(y_test, num_classes=len(command_labels))

# 3. CNN Modeli
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 1)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(len(command_labels), activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.summary()

# 4. Modeli Eğitme
history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=50, batch_size=32
)

# 5. Modeli Değerlendirme
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {test_accuracy:.2f}")


# Modeli Kaydet
model.save("cnnmodelwithaugmentation.h5")
