import tkinter as tk
from tkinter import filedialog, messagebox
import subprocess
import os
import threading
import glob
import sys
import shutil

# ==== Nastavení cesty ke skriptům ====
MODEL_OPTIONS = {
    "Rough → Clean Sketch": "test_rough_sketch_simplification.py",
    "Photo → Line Drawing": "test_photograph_to_line.py",
    "Clean Sketch → Vector": "test_vectorization.py",
}

SVG_CONVERTER = os.path.join("tools", "svg_conversion.py")

class VirtualSketchApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Virtual Sketching GUI")
        self.input_file = None
        self.model_script = tk.StringVar(value=list(MODEL_OPTIONS.values())[0])

        self.build_ui()

    def build_ui(self):
        tk.Label(self.root, text="1. Vyber vstupní obrázek:").pack(anchor="w")
        tk.Button(self.root, text="Vybrat obrázek", command=self.choose_file).pack(fill="x")

        self.file_label = tk.Label(self.root, text="Žádný soubor nevybrán", fg="gray")
        self.file_label.pack(anchor="w")

        tk.Label(self.root, text="2. Zvol model:").pack(anchor="w")
        for name, script in MODEL_OPTIONS.items():
            tk.Radiobutton(self.root, text=name, variable=self.model_script, value=script).pack(anchor="w")

        tk.Button(self.root, text="3. Spustit zpracování", command=self.run_processing).pack(pady=10, fill="x")

        self.status = tk.Label(self.root, text="Připraven", fg="green")
        self.status.pack(anchor="w")

    def choose_file(self):
        path = filedialog.askopenfilename(filetypes=[
            ("Obrázkové soubory", "*.png *.jpg *.jpeg *.bmp *.gif *.tif *.tiff"),
            ("PNG", "*.png"),
            ("JPEG", "*.jpg;*.jpeg"),
            ("BMP", "*.bmp"),
            ("GIF", "*.gif"),
            ("TIFF", "*.tif;*.tiff")
        ])
        if path:
            self.input_file = path
            self.file_label.config(text=os.path.basename(path), fg="black")

    def run_processing(self):
        if not self.input_file:
            messagebox.showerror("Chyba", "Nejprve vyber obrázek.")
            return

        script = self.model_script.get()
        cmd = [sys.executable, script, "--input", self.input_file]

        def task():
            self.status.config(text="Zpracovávám...", fg="blue")
            try:
                subprocess.run(cmd, check=True)
                self.status.config(text="✅ Hotovo", fg="green")
                self.move_outputs_to_sketches()
                self.run_svg_conversion()
            except subprocess.CalledProcessError:
                self.status.config(text="❌ Chyba při běhu skriptu", fg="red")

        threading.Thread(target=task).start()

    def move_outputs_to_sketches(self):
        if not self.input_file:
            return

        input_dir = os.path.dirname(self.input_file)
        input_base = os.path.splitext(os.path.basename(self.input_file))[0]
        sketches_dir = os.path.join(input_dir, "sketches")
        os.makedirs(sketches_dir, exist_ok=True)

        for ext in ["_0.npz", "_0_pred.png", "_input.png", "_0.svg"]:
            candidate = os.path.join(input_dir, f"{input_base}{ext}")
            if os.path.isfile(candidate):
                shutil.move(candidate, os.path.join(sketches_dir, os.path.basename(candidate)))

    def run_svg_conversion(self):
        if not self.input_file:
            return

        input_dir = os.path.dirname(self.input_file)
        input_base = os.path.splitext(os.path.basename(self.input_file))[0]
        npz_file = os.path.join(input_dir, f"{input_base}_0.npz")
        sketches_dir = os.path.join(input_dir, "sketches")
        npz_file_in_sketches = os.path.join(sketches_dir, f"{input_base}_0.npz")

        if not os.path.isfile(npz_file_in_sketches):
            print("⚠️ .npz soubor nebyl nalezen pro SVG konverzi.")
            return

        cmd = [sys.executable, SVG_CONVERTER, "--file", npz_file_in_sketches, "--svg_type", "single"]
        try:
            subprocess.run(cmd, check=True)
            svg_path = os.path.join(sketches_dir, f"{input_base}_0.svg")
            if os.path.isfile(svg_path):
                print(f"✅ SVG vytvořeno: {svg_path}")
        except subprocess.CalledProcessError:
            print("⚠️ Chyba při SVG konverzi")

if __name__ == "__main__":
    root = tk.Tk()
    app = VirtualSketchApp(root)
    root.mainloop()
