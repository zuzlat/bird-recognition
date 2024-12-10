import librosa
import soundfile as sf

def trim_audio(input_path, output_path, start_time, end_time):
    try:
        # Załaduj plik audio
        audio, sr = librosa.load(input_path, sr=None)
        start_sample = int(start_time * sr)
        end_sample = int(end_time * sr)

        # Przycięcie do wybranego przedziału
        trimmed_audio = audio[start_sample:end_sample]

        # Zapisanie przyciętego pliku
        sf.write(output_path, trimmed_audio, sr)
        return True
    except Exception as e:
        print(f"Błąd podczas przycinania pliku audio: {e}")
        return False

def load_audio(file_path):
    try:
        audio, sr = librosa.load(file_path, sr=None)
        return audio, sr
    except Exception as e:
        print(f"Błąd podczas ładowania pliku audio: {e}")
        return None, None