import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter.ttk import Button
import os
import audioread
from pygame import mixer
from PIL import Image, ImageTk
from torchvision.models import resnet18
import torch.nn as nn

from audio_utils import trim_audio
from classifier import classify_spectrogram
from sound_preprocessing import preprocess_audio
import librosa
import time
import torch


class CustomScale(tk.Scale):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class BirdSoundApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Rozpoznawanie dźwięków ptaków")
        self.root.geometry("800x600")
        self.root.resizable(False, False)
        self.time_slider_position = tk.DoubleVar()

        # Inicjalizacja pygame dla dźwięków
        mixer.init()

        # Aktualny plik dźwiękowy i spektrogram
        self.current_file = None
        self.spectrogram_path = os.path.join("database", "gui_files", "current_spectrogram.png")
        self.trimmed_file_path = os.path.join("database", "gui_files", "trimmed_audio.wav")


        self.spectrogram_path_model = os.path.join("database", "model_files", "current_spectrogram.png")
        self.trimmed_file_path_model = os.path.join("database", "model_files", "trimmed_audio.wav")

        # Nowe zmienne do kontroli odtwarzania
        self.song_duration = 0
        self.start_position = 0
        self.playing = False
        self.started = False
        self.min_duration = 5  # Minimalny czas w sekundach

        # Zmienne do obsługi spektrogramu
        self.start_x = 50
        self.end_x = 550

        self.model = self.load_model("best_model512_LR-small.pth")
        self.classes = [
            "Skowronek", "Jerzyk zwyczajny", "Jemiołuszka", "Bocian biały", "Wrona siwa", "Sikorka modra",
            "Zięba", "Jaskółka dymówka", "Słowik rdzawy", "Wróbel zwyczajny", "Kopciuszek",
            "Pierwiosnek zwyczajny", "Dzięcioł zielony", "Synogarlica turecka", "Inne", "Kos"
        ]

        # Przyciski główne
        self.load_button = Button(root, text="Wczytaj plik", command=self.load_file)
        self.load_button.pack(pady=5)

        # Spektrogram
        self.spectrogram_canvas = tk.Canvas(root, width=600, height=320, bg=self.root.cget("bg"))
        self.spectrogram_canvas.pack(pady=10)

        #flaga czy przyciete jest audio
        self.trimmed = False
        # Pasek audio
        self.audio_controls = tk.Frame(root)
        self.audio_controls.pack(pady=10)

        # Czas audio
        self.current_time_label = tk.Label(self.audio_controls, text="00:00")
        self.current_time_label.grid(row=0, column=0, padx=5)

        # Slider audio
        self.audio_slider = tk.Scale(
            self.audio_controls, from_=0, to=1000, orient=tk.HORIZONTAL, length=600,
            sliderlength=20, variable=self.time_slider_position, command=self.time_slider_changed, showvalue=False
        )
        self.audio_slider.grid(row=1, column=1, columnspan=1, sticky='ew')

        self.song_duration_label = tk.Label(self.audio_controls, text="00:00")
        self.song_duration_label.grid(row=1, column=2, padx=5)

        # Panel sterowania
        self.controls_frame = tk.Frame(root)
        self.controls_frame.pack(pady=5)

        # Przycisk play/pause
        self.play_pause_button = tk.Button(self.audio_controls, text="▶", command=self.play_pause_click)
        self.play_pause_button.grid(row=1, column=0, padx=5)

        self.root.after(ms=100, func=self.update_timeslider)

        # Czas całkowity
        self.song_duration_label = tk.Label(self.audio_controls, text="00:00")
        self.song_duration_label.grid(row=1, column=2, padx=5)

        # Panel sterowania
        self.controls_frame = tk.Frame(root)
        self.controls_frame.pack(pady=5)

        # # Przycisk play/pause
        # self.play_pause_button = tk.Button(self.controls_frame, text="▶", command=self.play_pause_click)
        # self.play_pause_button.pack(side=tk.LEFT, padx=5)

        # Suwak głośności
        self.volume_label = tk.Label(self.controls_frame, text="Głośność:")
        self.volume_label.pack(side=tk.LEFT, padx=5)
        self.volume_slider = tk.Scale(self.controls_frame, from_=0, to=100, orient=tk.HORIZONTAL, length=100,
                                      showvalue=False)
        self.volume_slider.set(50)  # Domyślnie 50%
        self.volume_slider.pack(side=tk.LEFT, padx=5)
        self.volume_slider.bind("<Motion>", self.set_volume)

        # Przyciski dodatkowe
        self.cut_button = Button(root, text="Przytnij plik", command=self.cut_file, state=tk.DISABLED)
        self.cut_button.pack(pady=5)

        self.recognize_button = Button(root, text="Rozpoznaj gatunek", command=self.recognize_species, state=tk.DISABLED)
        self.recognize_button.pack(pady=5)

        # Obsługa interaktywnych kresek
        self.spectrogram_canvas.bind("<B1-Motion>", self.move_lines)
    def set_volume(self, event=None):
        volume = self.volume_slider.get() / 100
        mixer.music.set_volume(volume)

    def load_file(self):
        try:
            file_path = filedialog.askopenfilename(filetypes=[("Audio Files", "*.mp3 *.wav")])
            if not file_path:
                return  # Użytkownik anulował wybór pliku

            duration = self.get_audio_duration(file_path)
            if duration < self.min_duration:
                messagebox.showerror("Błąd", "Plik audio musi trwać co najmniej 5 sekund!")
                return

            self.current_file = file_path
            self.trimmed = False  # Resetuj flagę
            self.audio_duration = self.song_duration = duration
            self.start_position = 0
            self.playing = False
            self.started = False

            success = preprocess_audio(self.current_file, self.spectrogram_path, full_file=True)
            if success:
                self.display_spectrogram()
                self.cut_button["state"] = tk.NORMAL
                self.recognize_button["state"] = tk.NORMAL

                total_time_str = time.strftime("%M:%S", time.gmtime(int(self.audio_duration)))
                self.song_duration_label.config(text=total_time_str)

                self.play_pause_button.config(text="▶")
                self.time_slider_position.set(0)
                self.current_time_label.config(text="00:00")
            else:
                messagebox.showerror("Błąd", "Nie udało się przetworzyć pliku audio.")
        except audioread.exceptions.NoBackendError:
            messagebox.showerror("Błąd", "Plik audio może być uszkodzony lub jego format nie jest obsługiwany.")
        except Exception as e:
            messagebox.showerror("Błąd", f"Nie udało się wczytać pliku audio: {e}")

    def get_audio_duration(self, file_path):
        audio, sr = librosa.load(file_path, sr=None)
        return librosa.get_duration(y=audio, sr=sr)

    def play_pause_click(self):
        if self.current_file is None:
            return

        if self.playing:
            # Zatrzymanie odtwarzania
            self.play_pause_button.config(text="▶")
            mixer.music.pause()
            self.playing = False
        else:
            # Wznowienie odtwarzania
            self.play_pause_button.config(text="⏸")
            if not self.started:  # Odtwarzanie od początku
                mixer.music.load(self.current_file)
                mixer.music.play(start=self.start_position)
                self.started = True
            else:
                mixer.music.unpause()  # Wznowienie od miejsca zatrzymania
            self.playing = True

    def time_slider_changed(self, _dummy=None):
        if self.current_file is not None:
            self.start_position = float(self.time_slider_position.get() / 1000) * self.song_duration
            mixer.music.stop()
            mixer.music.load(self.current_file)
            mixer.music.play(start=self.start_position)

            if not self.playing:
                mixer.music.pause()

    def update_timeslider(self):
        try:
            if self.playing and self.song_duration > 0:
                current_pos = mixer.music.get_pos() / 1000 + self.start_position
                if current_pos >= self.song_duration:
                    self.playing = False
                    self.play_pause_button.config(text="▶")
                    self.time_slider_position.set(1000)
                    mixer.music.stop()
                    self.started = False
                    current_pos = self.song_duration
                else:
                    slider_pos = (current_pos / self.song_duration) * 1000
                    self.time_slider_position.set(slider_pos)

                current_time_str = time.strftime("%M:%S", time.gmtime(int(current_pos)))
                self.current_time_label.config(text=current_time_str)
        except Exception as e:
            print(f"Błąd aktualizacji suwaka: {e}")

        self.root.after(ms=100, func=self.update_timeslider)

    def display_spectrogram(self):
        if os.path.exists(self.spectrogram_path):
            # Zwiększamy szerokość i wysokość Canvas
            canvas_width = 640  # Marginesy po 20px z każdej strony
            canvas_height = 320  # Dodatkowy margines na tekst nad spektrogramem
            spectrogram_width = 600
            spectrogram_height = 300

            # Przeskalowanie spektrogramu do wymaganych rozmiarów
            img = Image.open(self.spectrogram_path)
            img = img.resize((spectrogram_width, spectrogram_height), Image.Resampling.LANCZOS)
            self.spectrogram_image = ImageTk.PhotoImage(img)

            # Wyczyszczenie Canvas i ustawienie nowej konfiguracji
            self.spectrogram_canvas.config(width=canvas_width, height=canvas_height)
            self.spectrogram_canvas.delete("all")

            # Wyśrodkowanie spektrogramu na Canvas z marginesami
            spectrogram_x_offset = 20
            spectrogram_y_offset = 20
            self.spectrogram_canvas.create_image(
                spectrogram_x_offset, spectrogram_y_offset, anchor="nw", image=self.spectrogram_image
            )

            # Ustawienie początkowej pozycji kresek z marginesem
            self.start_x = spectrogram_x_offset
            self.end_x = spectrogram_width + spectrogram_x_offset
            self.start_line = self.spectrogram_canvas.create_line(
                self.start_x, spectrogram_y_offset, self.start_x, spectrogram_height + spectrogram_y_offset, fill="red",
                width=2
            )
            self.end_line = self.spectrogram_canvas.create_line(
                self.end_x, spectrogram_y_offset, self.end_x, spectrogram_height + spectrogram_y_offset, fill="red",
                width=2
            )

            # Dodanie tekstów czasu nad spektrogramem
            start_time_str = time.strftime("%M:%S", time.gmtime(0))
            end_time_str = time.strftime("%M:%S", time.gmtime(int(self.audio_duration)))

            self.start_time_text = self.spectrogram_canvas.create_text(
                self.start_x, spectrogram_y_offset - 3, text=start_time_str, fill="black", anchor="s",
                font=("Arial", 10, "bold")
            )
            self.end_time_text = self.spectrogram_canvas.create_text(
                self.end_x, spectrogram_y_offset - 3, text=end_time_str, fill="black", anchor="s",
                font=("Arial", 10, "bold")
            )
        else:
            messagebox.showerror("Błąd", "Nie znaleziono spektrogramu do wyświetlenia.")

    def move_lines(self, event):
        if self.start_line and self.end_line:
            min_distance = 600 * self.min_duration / self.audio_duration  # Minimalna odległość między kreskami w pikselach
            if abs(event.x - self.start_x) < abs(event.x - self.end_x):
                # Przesuwanie linii początkowej
                if 20 <= event.x < self.end_x - min_distance:  # Uwzględnij margines i minimalny dystans
                    self.start_x = event.x
                    self.spectrogram_canvas.coords(
                        self.start_line, self.start_x, 20, self.start_x, 300 + 20
                    )
                    # Aktualizacja tekstu czasu dla linii początkowej
                    start_time_str = time.strftime("%M:%S",
                                                   time.gmtime(int((self.start_x - 20) / 600 * self.audio_duration)))
                    self.spectrogram_canvas.coords(self.start_time_text, self.start_x, 17)
                    self.spectrogram_canvas.itemconfig(self.start_time_text, text=start_time_str)
            else:
                # Przesuwanie linii końcowej
                if self.start_x + min_distance < event.x <= 620:  # Uwzględnij margines i minimalny dystans
                    self.end_x = event.x
                    self.spectrogram_canvas.coords(
                        self.end_line, self.end_x, 20, self.end_x, 300 + 20
                    )
                    # Aktualizacja tekstu czasu dla linii końcowej
                    end_time_str = time.strftime("%M:%S",
                                                 time.gmtime(int((self.end_x - 20) / 600 * self.audio_duration)))
                    self.spectrogram_canvas.coords(self.end_time_text, self.end_x, 17)
                    self.spectrogram_canvas.itemconfig(self.end_time_text, text=end_time_str)

    def cut_file(self):
        start_time = (self.start_x - 20) / 600 * self.audio_duration
        end_time = (self.end_x - 20) / 600 * self.audio_duration
        if end_time - start_time < self.min_duration:
            messagebox.showerror("Błąd", "Fragment musi trwać co najmniej 5 sekund!")
            return

        success = trim_audio(self.current_file, self.trimmed_file_path, start_time, end_time)
        if success:
            self.trimmed = True
            messagebox.showinfo("Sukces", "Plik został przycięty.")
            self.current_file = self.trimmed_file_path
            self.audio_duration = self.song_duration = self.get_audio_duration(self.current_file)

            # Aktualizuj całkowity czas na etykiecie
            total_time_str = time.strftime("%M:%S", time.gmtime(int(self.audio_duration)))
            self.song_duration_label.config(text=total_time_str)

            preprocess_audio(self.current_file, self.spectrogram_path, full_file=True)
            self.display_spectrogram()
        else:
            messagebox.showerror("Błąd", "Nie udało się przyciąć pliku audio.")

    def load_model(self, model_path):
        try:
            model = resnet18()  # Wczytanie modelu z wagami domyślnymi
            model.fc = nn.Linear(model.fc.in_features, 16)  # Dopasowanie do liczby klas
            model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
            model.eval()  # Ustaw model w trybie ewaluacji
            return model
        except Exception as e:
            messagebox.showerror("Błąd", f"Nie udało się wczytać modelu: {e}")
            self.root.destroy()

    def recognize_species(self):
        try:
            # Użyj przyciętego pliku tylko wtedy, gdy jest aktualny
            audio_file_to_classify = self.trimmed_file_path if self.trimmed else self.current_file

            if not os.path.exists(audio_file_to_classify):
                messagebox.showerror("Błąd", "Nie znaleziono pliku audio do klasyfikacji.")
                return

            # Wygeneruj spektrogram na podstawie wybranego pliku audio
            success = preprocess_audio(audio_file_to_classify, self.spectrogram_path_model, full_file=False)
            if not success:
                messagebox.showerror("Błąd", "Nie udało się przetworzyć pliku audio do klasyfikacji. Spróbuj ponownie.")
                return

            # Klasyfikuj spektrogram
            result = classify_spectrogram(self.model, self.spectrogram_path_model, audio_file_to_classify)
            messagebox.showinfo("Wynik klasyfikacji", f"Zidentyfikowano gatunek: {result}")
        except Exception as e:
            messagebox.showerror("Błąd", f"Wystąpił problem podczas klasyfikacji: {e}")


# Reszta kodu pozostaje bez zmian
if __name__ == "__main__":
    root = tk.Tk()
    app = BirdSoundApp(root)
    root.mainloop()