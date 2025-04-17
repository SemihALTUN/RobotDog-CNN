import os
import numpy as np
import librosa
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import class_weight
from sklearn.utils import shuffle
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv1D, MaxPooling1D, LSTM
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt
import pickle


# Özellik çıkarım fonksiyonu
def extract_features(file_path, sr=22050, n_mfcc=20, n_fft=512, hop_length=256):
    try:
        y, sr = librosa.load(file_path, sr=sr)

        if len(y) < n_fft:
            print(f"File too short: {file_path}")
            return None

        y, _ = librosa.effects.trim(y, top_db=20)

        noise_sample = y[:sr // 2] if len(y) >= sr // 2 else y
        noise_reduced = y - np.mean(noise_sample)

        y_normalized = (noise_reduced - np.mean(noise_reduced)) / (np.std(noise_reduced) + 1e-9)

        mfcc = librosa.feature.mfcc(y=y_normalized, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length)
        mfcc_mean = np.mean(mfcc, axis=1)
        mfcc_std = np.std(mfcc, axis=1)

        spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=y_normalized, sr=sr, n_fft=n_fft, hop_length=hop_length))
        spectral_bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=y_normalized, sr=sr, n_fft=n_fft, hop_length=hop_length))
        zcr = np.mean(librosa.feature.zero_crossing_rate(y_normalized, frame_length=n_fft, hop_length=hop_length))
        rms = np.mean(librosa.feature.rms(y=y_normalized, frame_length=n_fft, hop_length=hop_length))

        features = np.concatenate([
            mfcc_mean,
            mfcc_std,
            [spectral_centroid],
            [spectral_bandwidth],
            [zcr],
            [rms]
        ])

        return features

    except Exception as e:
        print(f"Error processing {file_path}: {str(e)}")
        return None


# Veri yükleme
def load_dataset(dataset_path):
    X = []
    y = []

    huseyin_path = os.path.join(dataset_path, "Huseyin")
    other_path = os.path.join(dataset_path, "other_voice")

    for folder, label in [(huseyin_path, 1), (other_path, 0)]:
        if os.path.exists(folder):
            for filename in os.listdir(folder):
                file_path = os.path.join(folder, filename)
                features = extract_features(file_path)
                if features is not None:
                    X.append(features)
                    y.append(label)

    if len(X) == 0:
        raise ValueError("Hiç geçerli ses dosyası bulunamadı")

    return np.array(X), np.array(y)


# Model
def create_model(input_shape):
    model = Sequential([
        Conv1D(64, 3, activation='relu', input_shape=input_shape),
        MaxPooling1D(2),
        Dropout(0.3),

        Conv1D(128, 3, activation='relu'),
        MaxPooling1D(2),
        Dropout(0.3),

        LSTM(64, return_sequences=False),
        Dropout(0.3),

        Dense(64, activation='relu'),
        Dropout(0.3),

        Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    return model


# Ana fonksiyon
def main():
    try:
        dataset_path = "datasetSpeaker"
        X, y = load_dataset(dataset_path)

        # Sınıf ağırlıkları hesapla
        class_weights = class_weight.compute_class_weight(
            class_weight='balanced',
            classes=np.unique(y),
            y=y
        )
        class_weights_dict = dict(enumerate(class_weights))

        # Train-test böl
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y)

        # Ölçekle
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        X_train_reshaped = X_train_scaled[..., np.newaxis]
        X_test_reshaped = X_test_scaled[..., np.newaxis]

        model = create_model((X_train_reshaped.shape[1], 1))
        model.summary()

        callbacks = [
            EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
            ModelCheckpoint('best_model.h5', save_best_only=True)
        ]

        # Eğitim
        history = model.fit(
            X_train_reshaped, y_train,
            validation_data=(X_test_reshaped, y_test),
            epochs=50,
            batch_size=32,
            callbacks=callbacks,
            class_weight=class_weights_dict,
            verbose=1
        )

        # Değerlendirme
        loss, accuracy = model.evaluate(X_test_reshaped, y_test, verbose=0)
        print(f"\n🔍 Test Doğruluk: {accuracy * 100:.2f}%")

        # Kaydet
        model.save("cnn_lstm_speaker_model.h5")
        with open('scaler.pkl', 'wb') as f:
            pickle.dump(scaler, f)

        # Grafikler
        plt.figure(figsize=(12, 5))

        plt.subplot(1, 2, 1)
        plt.plot(history.history['accuracy'], label='Eğitim')
        plt.plot(history.history['val_accuracy'], label='Doğrulama')
        plt.title('Doğruluk Grafiği')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(True)

        plt.subplot(1, 2, 2)
        plt.plot(history.history['loss'], label='Eğitim')
        plt.plot(history.history['val_loss'], label='Doğrulama')
        plt.title('Kayıp Grafiği')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        plt.show()

    except Exception as e:
        print(f"Bir hata oluştu bebeğim: {str(e)}")


if __name__ == "__main__":
    main()
