import torch
import torchvision.transforms as transforms
from PIL import Image

from sound_preprocessing import preprocess_audio

# Lista nazw gatunków w języku polskim
species_list = [
    "Skowronek", "Jerzyk zwyczajny", "Jemiołuszka", "Bocian biały", "Wrona siwa",
    "Sikorka modra", "Zięba", "Jaskółka dymówka", "Słowik rdzawy", "Wróbel zwyczajny",
    "Kopciuszek", "Pierwiosnek zwyczajny", "Dzięcioł zielony", "Synogarlica turecka",
    "Inne", "Kos"
]

def classify_spectrogram(model, spectrogram_path, audio):
    """
    Klasyfikuje spektrogram przy użyciu wczytanego modelu.

    Args:
        model (torch.nn.Module): Wcześniej wczytany model PyTorch.
        spectrogram_path (str): Ścieżka do obrazu spektrogramu.

    Returns:
        str: Nazwa zidentyfikowanego gatunku lub komunikat błędu.
    """
    try:
        # Transformacja obrazu do formatu wejściowego modelu
        transform = transforms.Compose([
            transforms.Resize((256, 438)),  # Rozmiar zgodny z wymaganiami modelu
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # Normalizacja
        ])
        succes = preprocess_audio(audio, spectrogram_path, full_file=False)
        img = Image.open(spectrogram_path).convert("RGB")
        img = transform(img).unsqueeze(0)

        # Predykcja
        with torch.no_grad():
            outputs = model(img)
            _, predicted = torch.max(outputs, 1)

        # Zwrócenie nazwy gatunku
        return species_list[predicted.item()]
    except Exception as e:
        print(f"Błąd podczas klasyfikacji spektrogramu: {e}")
        return "Błąd klasyfikacji"
