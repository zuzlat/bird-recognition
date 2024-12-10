import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import json

from torchvision.models import resnet18, ResNet18_Weights

from CNN_model import BirdSoundCNN

# Ustawienie urządzenia: jeśli dostępne CUDA, użyj GPU, w przeciwnym razie CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

if torch.cuda.is_available():
    print(f"CUDA jest dostępne. Urządzenie GPU: {torch.cuda.get_device_name(0)}")
else:
    print("CUDA nie jest dostępne.")


tensor = torch.randn(3, 3)
print(f"Przed przeniesieniem na GPU: {tensor.device}")

# Przenosimy tensor na GPU
tensor = tensor.to('cuda')  # Lub tensor.cuda()
print(f"Po przeniesieniu na GPU: {tensor.device}")



# Ustawienia hiperparametrów
batch_size = 32
learning_rate = 0.0001
input_size = (438, 256)  # Rozdzielczość obrazów
num_classes = 16  # Liczba klas
epochs = 150
mel_bands = 512
n = 1 #wersja moedlu


if learning_rate == 0.001:
    lr = 'med'
elif learning_rate == 0.0001:
    lr = 'small'
elif learning_rate == 0.01:
    lr = 'big'

#model_path = f"D:\\studia\\bird_sound_recognition\\model training\\models\\{mel_bands}\\best_model{mel_bands}_LR-{lr}.pth"
model_path = 'D:\studia/bird_sound_recognition\model_training\models/512/best-model_512_LR-small_2.pth'
def train_model(model, train_loader, test_loader, criterion, optimizer, path = model_path, num_epochs=epochs, stop_threshold_train=1e-4, stop_treshold_test=0.01, min_improvement_epochs=15):
    train_losses, test_losses = [], []
    best_acc = 0.0
    best_model_state = None
    model.to(device)

    history = {'loss': [], 'accuracy': [], 'val_loss': [], 'val_accuracy': []}
    no_improvement_count = 0  # Licznik epok bez poprawy

    # Monitorowanie małych zmian w loss
    prev_train_loss = None
    prev_val_loss = None
    stagnation_count_train = 0  # Licznik epok z minimalnymi zmianami w loss
    stagnation_count_test = 0
    overfitting = 0

    for epoch in range(num_epochs):
        print(f'Epoch {epoch + 1}/{num_epochs}')
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            # Dokładność
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        epoch_loss = running_loss / len(train_loader)
        epoch_acc = correct / total
        history['loss'].append(epoch_loss)
        history['accuracy'].append(epoch_acc)

        # Testowanie na danych testowych
        model.eval()
        test_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                test_loss += loss.item()

                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        test_loss /= len(test_loader)
        test_acc = correct / total
        history['val_loss'].append(test_loss)
        history['val_accuracy'].append(test_acc)

        print(f"Train Loss: {epoch_loss:.4f} - Train Accuracy: {epoch_acc:.4f}")
        print(f"Test Loss: {test_loss:.4f} - Test Accuracy: {test_acc:.4f}")

        # Sprawdzanie i zapisywanie najlepszego modelu
        if test_acc > best_acc:
            best_acc = test_acc
            best_epoch = epoch + 1
            best_model_state = model.state_dict()
            print(f"Zapisano najlepszy model w epoce {best_epoch} z dokładnością: {best_acc:.4f}")

        # Sprawdzanie stagnacji w train_loss i val_loss
        if prev_train_loss is not None and prev_val_loss is not None:
            train_loss_change = abs(prev_train_loss - epoch_loss)
            val_loss_change = abs(prev_val_loss - test_loss)

            if train_loss_change < stop_threshold_train:
                stagnation_count_train += 1
                print(f"Minimalne zmiany w Train Loss ({train_loss_change:.6f})przez {stagnation_count_train} epok.")
            else:
                stagnation_count_train = 0  # Resetowanie licznika, jeśli zmiany są większe

            if val_loss_change < stop_treshold_test:
                stagnation_count_test += 1
                print(
                    f"Minimalne zmiany w Val Loss ({val_loss_change:.6f})przez {stagnation_count_test} epok.")
            else:
                stagnation_count_test = 0  # Resetowanie licznika, jeśli zmiany są większe

            if prev_val_loss - test_loss < 0:
                overfitting += 1
                print(
                    f"Rosnący Val Loss ({test_loss:.6f}, {prev_val_loss}) przez {overfitting} epok.")
            else:
                overfitting = 0  # Resetowanie licznika, jeśli zmiany są większe

            if overfitting >= 5 or stagnation_count_test >= min_improvement_epochs or stagnation_count_train >= min_improvement_epochs:
                print(f"Uczenie zatrzymane po {epoch + 1} epokach z powodu minimalnych zmian w loss.")
                break

        # Zapamiętanie poprzednich wartości
        prev_train_loss = epoch_loss
        prev_val_loss = test_loss

    if best_model_state is not None:
        torch.save(best_model_state, path)
        print(f"Najlepszy model został zapisany jako {mel_bands}_LR-{lr}_b-{batch_size}_{n}.pth")

    print("Training complete.")
    return history


def validate_model(model, val_loader, criterion, best_model_path=model_path,
                   output_file=f"D:\\studia\\bird_sound_recognition\\model training\\models\\{mel_bands}\\best_model{mel_bands}_LR-{lr}-2137.txt"): #_b-{batch_size}_{n}.txt"):
    # Wczytanie polskich nazw gatunków
    with open('D:\\studia\\bird_sound_recognition\\spec_name_pl.json', 'r', encoding='utf-8') as f:
        species_pl_names = json.load(f)

    if best_model_path:
        model.load_state_dict(torch.load(best_model_path))

    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    all_labels = []
    all_preds = []

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

            # Dokładność
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # Do raportu klasyfikacji
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())

    val_acc = correct / total
    avg_val_loss = val_loss / len(val_loader)

    # Generowanie raportu klasyfikacji
    report = classification_report(all_labels, all_preds, target_names=val_loader.dataset.classes)
    print(f"Validation Loss: {avg_val_loss:.4f}, Validation Accuracy: {val_acc:.4f}")
    print("\nClassification Report:\n", report)

    # Zapis do pliku
    with open(output_file, "w") as f:
        f.write(f"Validation Loss: {avg_val_loss:.4f}\n")
        f.write(f"Validation Accuracy: {val_acc:.4f}\n\n")
        f.write("Classification Report:\n")
        f.write(report)

    # Macierz pomyłek w procentach
    cm = confusion_matrix(all_labels, all_preds)

    # Konwersja na procenty
    cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100

    # Przygotowanie nazw klas w języku polskim
    pl_classes = [
        species_pl_names[i].title() for i in range(len(val_loader.dataset.classes))
        if i < len(species_pl_names)
    ]
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm_percent, annot=True, fmt=".0f", cmap="BuPu",
                xticklabels=pl_classes,
                yticklabels=pl_classes)
    # plt.xticks(rotation=45)  # Ustawienie kątów na 45 stopni dla etykiet osi X
    # plt.yticks(rotation=0)
    plt.xlabel("Przewidywane etykiety")
    plt.ylabel("Prawdziwe etykiety")
    plt.title("Macierz Pomyłek (Procent)")
    plt.tight_layout()
    # Tworzenie folderu, jeśli nie istnieje
    os.makedirs('conf_matrices', exist_ok=True)

    # Zapis do pliku PNG
    plt.savefig(f'results/conf_matrices/conf_{mel_bands}_LR-{lr}_b-{batch_size}_{n}.png', dpi=300)
    plt.show()
    plt.close()


def generate_plots(history, train_loader, test_loader, device):
    # Generowanie wykresu funkcji straty (train_loss vs test_loss)
    plt.figure(figsize=(10, 5))
    plt.plot(history['loss'], label='Strata w treningu')
    plt.plot(history['val_loss'], label='Strata w teście')
    plt.title("Funkcja straty")
    plt.xlabel("Epoki")
    plt.ylabel("Wartość funkcji straty")
    plt.legend()
    plt.grid(True)
    plt.savefig(f'results/loss/loss_{mel_bands}_LR-{lr}_b-{batch_size}_{n}.png', dpi=300)
    plt.show()


    # Generowanie wykresu dokładności (train_accuracy vs test_accuracy)
    plt.figure(figsize=(10, 5))
    plt.plot(history['accuracy'], label='Dokładność treningowa')
    plt.plot(history['val_accuracy'], label='Dokładność testowa')
    plt.title("Wartość dokładności w każdej epoce")
    plt.xlabel("Eppoki")
    plt.ylabel("Dokładność")
    plt.legend()
    plt.grid(True)
    plt.savefig(f'results/accuracy/acc_{mel_bands}_LR-{lr}_b-{batch_size}_{n}.png', dpi=300)
    plt.show()



# Transformacje danych
transform = transforms.Compose([
    transforms.Resize((438, 256)),  # Dopasowanie do rozdzielczości obrazów
    transforms.ToTensor(),         # Konwersja na tensory
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalizacja
])

# Ścieżki do danych
data_dir = f"D:/studia/bird_sound_recognition/data/training/split_{mel_bands}-mel"
train_dir = os.path.join(data_dir, 'train')
test_dir = os.path.join(data_dir, 'test')

# Ładowanie danych
train_dataset = datasets.ImageFolder(train_dir, transform=transform)
test_dataset = datasets.ImageFolder(test_dir, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Model ResNet
model = resnet18()  # Wczytanie modelu z wagami domyślnymi
model.fc = nn.Linear(model.fc.in_features, num_classes)  # Dopasowanie do liczby klas# Wczytanie modelu z wagami domyślnymi
model = model.to('cuda')

# Strata i optymalizator
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)


# Trenowanie modelu
history = train_model(model, train_loader, test_loader, criterion, optimizer, path=model_path, num_epochs=epochs)

# Generowanie wykresów
generate_plots(history, train_loader, test_loader, device)

# Walidacja modelu
validate_model(model, test_loader, criterion)


