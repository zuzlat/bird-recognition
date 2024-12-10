import scipy.signal as signal
import librosa
import numpy as np
import matplotlib.pyplot as plt


# Funkcja do zastosowania filtru pasmowego
def bandpass_filter(audio, sr, low_cutoff, high_cutoff, order=5):
    nyquist = 0.5 * sr  # Częstotliwość Nyquista
    low = low_cutoff / nyquist
    high = high_cutoff / nyquist
    b, a = signal.butter(order, [low, high], btype='band')
    filtered_audio = signal.filtfilt(b, a, audio)
    return filtered_audio


def preprocess_audio(input_path, output_spectrogram_path, full_file=False):
    try:
        # Load the audio file
        audio, sr = librosa.load(input_path, sr=None)

        if not full_file:
            # Calculate the duration of the audio
            duration = librosa.get_duration(y=audio, sr=sr)

            # Pad or truncate to 10 seconds
            target_duration = 10
            if duration < target_duration:
                missing_duration = target_duration - duration
                missing_samples = int(missing_duration * sr)
                # Repeat the start of the audio to fill the gap
                audio = np.concatenate((audio, audio[:missing_samples]))
            elif duration > target_duration:
                audio = audio[:int(target_duration * sr)]

            # Resample to 22.05 kHz
            audio = librosa.resample(audio, orig_sr=sr, target_sr=22050)
            sr = 22050

            # Apply pre-emphasis and bandpass filter
            audio = librosa.effects.preemphasis(audio)
            audio = bandpass_filter(audio, sr, 400, 10000)

        # Convert to mel spectrogram
        mel_spectrogram = librosa.feature.melspectrogram(
            y=audio, sr=sr, n_mels=512, fmax=sr // 2
        )
        mel_spectrogram_db = librosa.power_to_db(mel_spectrogram, ref=np.max)

        # Plot the mel spectrogram
        plt.figure(figsize=(4.38, 2.56))
        plt.axis("off")
        plt.tight_layout()
        plt.imshow(mel_spectrogram_db, aspect='auto', origin='lower', cmap='viridis')
        plt.savefig(output_spectrogram_path, bbox_inches="tight", pad_inches=0)
        plt.close()

        return True
    except Exception as e:
        print(f"Błąd podczas przetwarzania pliku audio: {e}")
        return False
