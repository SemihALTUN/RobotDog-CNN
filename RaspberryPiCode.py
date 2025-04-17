import sounddevice as sd
from scipy.io.wavfile import write
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from PIL import Image
import os
from adafruit_servokit import ServoKit
import time

# Komut etiketleri (eğitim sırasında kullanılan sıralamaya göre)
command_labels = {
    'dur': 0,
    'geri_git': 1,
    'ileri_git': 2,
    'saga_dogru_git': 3,
    'sola_dogru_git': 4
}

# Komut çıktıları (4 bit)
command_outputs = {
    'dur': [0, 0, 0, 0],
    'geri_git': [0, 1, 1, 0],
    'ileri_git': [1, 0, 0, 1],
    'saga_dogru_git': [0, 0, 1, 1],
    'sola_dogru_git': [1, 1, 0, 0]
}

# CNN modelini yükle
model = load_model(r"/home/rasperrypi/Desktop/cnnmodelwithaugmentation.h5")

# ServoKit nesnesi oluştur
kit = ServoKit(channels=16)


def record_audio(duration=5, fs=22050, filename="mic_input.wav"):
    """Mikrofondan ses kaydı alır ve WAV dosyasına kaydeder."""
    print(f"🎤 {duration} saniye boyunca konuşun...")
    audio = sd.rec(int(duration * fs), samplerate=fs, channels=1)
    sd.wait()
    audio = np.squeeze(audio)
    write(filename, fs, (audio * 32767).astype(np.int16))  # WAV formatında kaydet
    print("✅ Ses kaydedildi.")
    return filename


def extract_spectrogram(audio_path):
    """Ses dosyasından mel-spektrogram çıkarır."""
    y, sr = librosa.load(audio_path, duration=5)
    spectrogram = librosa.feature.melspectrogram(y=y, sr=sr)
    plt.figure(figsize=(2.56, 2.56))
    librosa.display.specshow(librosa.power_to_db(spectrogram, ref=np.max),
                             y_axis='mel', fmax=8000, x_axis='time')
    plt.axis('off')
    temp_path = "temp_spectrogram.png"
    plt.savefig(temp_path, bbox_inches='tight', pad_inches=0)
    plt.close()

    img = Image.open(temp_path).convert('L')
    img = img.resize((128, 128))
    img_array = np.array(img) / 255.0
    img_array = img_array.reshape(1, 128, 128, 1)

    os.remove(temp_path)
    return img_array


def predict_command(audio_path):
    """Tahmin yapar ve 4 bitlik komutu döndürür."""
    spectrogram = extract_spectrogram(audio_path)
    prediction = model.predict(spectrogram)
    predicted_index = np.argmax(prediction)
    predicted_label = list(command_labels.keys())[list(command_labels.values()).index(predicted_index)]
    output_bits = command_outputs[predicted_label]
    print(f"\n🎙️ Tahmin edilen komut: {predicted_label}")
    print(f"🔢 Komut çıktısı (4 bit): {output_bits}")
    return predicted_label, output_bits


# Komut fonksiyonları

def dur():
    def idle():
        """Dur komutunu uygular."""
        print("🚦 Dur komutu uygulandı")
        kit.servo[11].angle = 100  # sol_arka_ust
        kit.servo[15].angle = 90  # sol_arka_alt
        kit.servo[7].angle = 70  # sag_arka_ust
        kit.servo[8].angle = 70  # sag_arka_alt
        kit.servo[12].angle = 120  # sag_on_ust
        kit.servo[0].angle = 90  # sag_on_alt
        kit.servo[3].angle = 60  # sol_on_ust
        kit.servo[4].angle = 90  # sol_on_alt
        time.sleep(3)  # 5 saniye bekle

    idle()

    """Sınav pozu aldırır."""
    print("📸 Sınav pozu alınıyor...")
    for _ in range(10):
        # Başlangıç durumu
        kit.servo[11].angle = 100  # sol_arka_ust
        kit.servo[15].angle = 90  # sol_arka_alt
        kit.servo[7].angle = 70  # sag_arka_ust
        kit.servo[8].angle = 70  # sag_arka_alt
        kit.servo[12].angle = 120  # sag_on_ust
        kit.servo[0].angle = 90  # sag_on_alt
        kit.servo[3].angle = 60  # sol_on_ust
        kit.servo[4].angle = 90  # sol_on_alt

        time.sleep(0.5)

        # Poz
        kit.servo[15].angle = 40  # sol_arka_alt
        kit.servo[8].angle = 110  # sag_arka_alt
        kit.servo[0].angle = 140  # sag_on_alt
        kit.servo[4].angle = 40  # sol_on_alt

        time.sleep(0.5)

    idle()


def ileri():
    def idle():
        """Dur komutunu uygular."""
        print("🚦 Dur komutu uygulandı")
        kit.servo[11].angle = 100  # sol_arka_ust
        kit.servo[15].angle = 90  # sol_arka_alt
        kit.servo[7].angle = 70  # sag_arka_ust
        kit.servo[8].angle = 70  # sag_arka_alt
        kit.servo[12].angle = 120  # sag_on_ust
        kit.servo[0].angle = 90  # sag_on_alt
        kit.servo[3].angle = 60  # sol_on_ust
        kit.servo[4].angle = 90  # sol_on_alt
        time.sleep(1)  # 5 saniye bekle

    idle()

    """İleri yürüme hareketi uygular."""
    print("🚗 İleri git komutu uygulanıyor (10 tekrar)")
    for _ in range(10):
        kit.servo[11].angle = 100  # sol_arka_ust
        kit.servo[7].angle = 40  # sag_arka_ust
        kit.servo[12].angle = 120  # sag_on_ust
        kit.servo[3].angle = 110  # sol_on_ust
        time.sleep(0.1)

        kit.servo[15].angle = 90  # sol_arka_alt
        kit.servo[8].angle = 120  # sag_arka_alt
        kit.servo[0].angle = 90  # sag_on_alt
        kit.servo[4].angle = 40  # sol_on_alt
        time.sleep(0.1)

        kit.servo[11].angle = 140  # sol_arka_ust
        kit.servo[7].angle = 70  # sag_arka_ust
        kit.servo[12].angle = 70  # sag_on_ust
        kit.servo[3].angle = 60  # sol_on_ust
        time.sleep(0.1)

        kit.servo[15].angle = 40  # sol_arka_alt
        kit.servo[8].angle = 70  # sag_arka_alt
        kit.servo[0].angle = 140  # sag_on_alt
        kit.servo[4].angle = 90  # sol_on_alt
        time.sleep(0.1)

    idle()


def geri():
    def idle():
        """Dur pozisyonuna dön."""
        print("🚦 Dur pozisyonu")
        kit.servo[11].angle = 100  # sol_arka_ust
        kit.servo[15].angle = 90  # sol_arka_alt
        kit.servo[7].angle = 70  # sag_arka_ust
        kit.servo[8].angle = 70  # sag_arka_alt
        kit.servo[12].angle = 120  # sag_on_ust
        kit.servo[0].angle = 90  # sag_on_alt
        kit.servo[3].angle = 60  # sol_on_ust
        kit.servo[4].angle = 90  # sol_on_alt
        time.sleep(1)

    idle()

    """Geri adımı YLine_Demo mantığıyla yapar."""
    print("⬅️ Geri Git komutu uygulanıyor (10 tekrar)")
    idle()
    for _ in range(10):
        """sol ön ve sağ arka üst bacaklar"""
        kit.servo[7].angle = 30
        kit.servo[3].angle = 100
        time.sleep(0.1)
        """aynı ayakların alt kısmı açılır"""
        kit.servo[8].angle = 30
        kit.servo[4].angle = 130
        time.sleep(0.1)
        """üst kısım tekrar ileri alınır"""
        kit.servo[7].angle = 70
        kit.servo[3].angle = 60
        time.sleep(0.1)
        """alt kısımlar geri denge noktasına alınır"""
        kit.servo[8].angle = 70
        kit.servo[4].angle = 90
        time.sleep(0.1)
        """sağ ön ve sol arka üst bacaklar"""
        kit.servo[12].angle = 80
        kit.servo[11].angle = 140
        time.sleep(0.1)
        """sağ ön ve sol arka alt bacaklar """
        kit.servo[0].angle = 50
        kit.servo[15].angle = 140
        time.sleep(0.1)
        """sağ ön ve sol arka üst bacaklar """
        kit.servo[12].angle = 120
        kit.servo[11].angle = 100
        time.sleep(0.1)
        """sağ ön ve sol arka alt bacaklar"""
        kit.servo[0].angle = 90
        kit.servo[15].angle = 90
        time.sleep(0.1)

    idle()


def saga_dogru_git():
    """Sağa doğru git komutunu uygular."""

    # Sağa git komutunun motor hareketlerini burada tanımlayın
    def idle():
        """Dur komutunu uygular."""
        print("🚦 Dur komutu uygulandı")
        kit.servo[11].angle = 100  # sol_arka_ust
        kit.servo[15].angle = 90  # sol_arka_alt
        kit.servo[7].angle = 70  # sag_arka_ust
        kit.servo[8].angle = 70  # sag_arka_alt
        kit.servo[12].angle = 120  # sag_on_ust
        kit.servo[0].angle = 90  # sag_on_alt
        kit.servo[3].angle = 60  # sol_on_ust
        kit.servo[4].angle = 90  # sol_on_alt
        time.sleep(1)  # 5 saniye bekle

    idle()

    print("⬅️ Sağa Doğru Git komutu uygulanıyor (10 tekrar)")
    for _ in range(10):
        kit.servo[3].angle = 20
        time.sleep(0.1)
        kit.servo[4].angle = 130
        time.sleep(0.1)
        kit.servo[3].angle = 60
        time.sleep(0.1)
        kit.servo[4].angle = 90
        time.sleep(0.1)

        kit.servo[11].angle = 60
        time.sleep(0.1)
        kit.servo[15].angle = 130
        time.sleep(0.1)
        kit.servo[11].angle = 100
        time.sleep(0.1)
        kit.servo[15].angle = 90
        time.sleep(0.1)

        kit.servo[3].angle = 20
        time.sleep(0.1)
        kit.servo[4].angle = 130
        time.sleep(0.1)
        kit.servo[3].angle = 60
        time.sleep(0.1)
        kit.servo[4].angle = 90
        time.sleep(0.1)

        kit.servo[11].angle = 60
        time.sleep(0.1)
        kit.servo[15].angle = 130
        time.sleep(0.1)
        kit.servo[11].angle = 100
        time.sleep(0.1)
        kit.servo[15].angle = 90
        time.sleep(0.1)

    idle()


def sola_dogru_git():
    def idle():
        """Dur komutunu uygular."""
        print("🚦 Dur komutu uygulandı")
        kit.servo[11].angle = 100  # sol_arka_ust
        kit.servo[15].angle = 90  # sol_arka_alt
        kit.servo[7].angle = 70  # sag_arka_ust
        kit.servo[8].angle = 70  # sag_arka_alt
        kit.servo[12].angle = 120  # sag_on_ust
        kit.servo[0].angle = 90  # sag_on_alt
        kit.servo[3].angle = 60  # sol_on_ust
        kit.servo[4].angle = 90  # sol_on_alt
        time.sleep(3)  # 5 saniye bekle

    idle()

    print("⬅️ Sola Doğru Git komutu uygulanıyor (10 tekrar)")
    for _ in range(10):
        # 1. Adım: Tüm bacaklar geri konuma getirilir (üst bacakları geri bükeriz)
        kit.servo[11].angle = 130  # sol_arka_ust (geri)
        kit.servo[7].angle = 40  # sag_arka_ust (geri)
        kit.servo[12].angle = 90  # sag_on_ust (geri)
        kit.servo[3].angle = 90  # sol_on_ust (geri)
        time.sleep(0.1)

        # 2. Alt bacaklar yere bastırılır
        kit.servo[15].angle = 100  # sol_arka_alt (aşağı)
        kit.servo[8].angle = 80  # sag_arka_alt (aşağı)
        kit.servo[0].angle = 110  # sag_on_alt (aşağı)
        kit.servo[4].angle = 100  # sol_on_alt (aşağı)
        time.sleep(0.1)

        # 3. Üst bacaklar öne alınır (ileri çekilir, bir sonraki adım için hazır konuma geçer)
        kit.servo[11].angle = 100  # sol_arka_ust (nötr)
        kit.servo[7].angle = 70  # sag_arka_ust (nötr)
        kit.servo[12].angle = 120  # sag_on_ust (nötr)
        kit.servo[3].angle = 60  # sol_on_ust (nötr)
        time.sleep(0.1)

        # 4. Alt bacaklar havaya kaldırılır (adım atar gibi)
        kit.servo[15].angle = 80  # sol_arka_alt (yukarı)
        kit.servo[8].angle = 60  # sag_arka_alt (yukarı)
        kit.servo[0].angle = 70  # sag_on_alt (yukarı)
        kit.servo[4].angle = 90  # sol_on_alt (yukarı)
        time.sleep(0.1)

    idle()


# Ana akış
if __name__ == "__main__":
    while True:
        audio_file = record_audio()
        command, _ = predict_command(audio_file)

        # Komutlara göre ilgili fonksiyonu çağırıyoruz
        if command == 'dur':
            dur()
        elif command == 'ileri_git':
            ileri()
        elif command == 'geri_git':
            geri()
        elif command == 'saga_dogru_git':
            saga_dogru_git()
        elif command == 'sola_dogru_git':
            sola_dogru_git()
        time.sleep(5)  # 1 saniye bekle