from gui import BirdSoundApp
import tkinter as tk

if __name__ == "__main__":
    # Inicjalizacja głównego okna Tkinter
    root = tk.Tk()
    app = BirdSoundApp(root)
    root.mainloop()