# 🪟 Windows Installation Guide for Virtual Sketching

This guide provides step-by-step instructions to set up and run the [Virtual Sketching](https://github.com/MarkMoHR/virtual_sketching) project on Windows using Anaconda and Python 3.6.

## ✅ Requirements

- Windows 10 or newer
- Anaconda installed
- Git installed (optional, but recommended)

---

## 📦 Step 1: Create and Activate Conda Environment

```bash
conda create -n virtual_sketching python=3.6 -y
conda activate virtual_sketching
```

## 📂 Step 2: Clone the Repository

```bash
cd D:\
git clone https://github.com/MarkMoHR/virtual_sketching.git
cd virtual_sketching
```

(If you plan to contribute, consider forking the repo and using your own URL.)

---

## 🔧 Step 3: Install Required Packages

### From `conda`:
```bash
conda install opencv=3.4.2 pillow=6.2.0 scipy=1.5.2 -y
conda install -c conda-forge pycairo gtk3 cffi -y
```

### Then remove default TensorFlow (if installed via conda):
```bash
conda remove tensorflow
```

### Install required packages via `pip`:
```bash
pip install tensorflow==1.15.0
pip install numpy gizeh cairocffi matplotlib svgwrite
```

> ⚠️ Do not upgrade `pillow`, `scipy`, or `tensorflow` — newer versions are incompatible.

---

## 🛠️ Step 4: Fix Backend Compatibility

In `utils.py`, near the top, ensure the following:

```python
from PIL import Image
import matplotlib
matplotlib.use('TkAgg')  # force compatible backend
import matplotlib.pyplot as plt
```

This ensures proper rendering with tkinter on Windows.

---

## 🚀 Step 5: Run a Demo

From the project directory:

```bash
python test_vectorization.py --input sample_inputs\muten.png --sample 5
```

> ⚠️ This script only generates `.npz` and `.png` files. To convert to `.svg`, see next section.

---

## 🖼️ Optional: Use GUI Tool (Windows Only)

### Step 1: Launch GUI with provided batch file

Use the `runme.bat` file to activate the conda environment and launch the Python GUI:

```bat
rem runme.bat
set CONDAPATH=C:\ProgramData\anaconda3
set ENVNAME=virtual_sketching
call %CONDAPATH%\Scripts\activate.bat %ENVNAME%
python virtual_sketch_gui.py
pause
```

### Step 2: Select an input image and model

The GUI allows you to:
- Choose input image (PNG, JPEG, BMP, etc.)
- Select one of the three models
- Automatically runs processing and converts results to SVG
- Saves all outputs into a `sketches/` subfolder of the input image directory

---

## 🧠 Known Compatibility Notes

- Python 3.6 and TensorFlow 1.15 are required (due to use of `tensorflow.contrib`)
- Windows support requires manual setup of Gizeh and Cairo backends
- GPU usage optional — TensorFlow 1.15 requires CUDA 10.0 and cuDNN 7

---

## 🧩 Troubleshooting Tips

- ❌ `No module named '_cffi_backend'` → Run: `conda install -c conda-forge cffi`
- ❌ `ImportError: cannot import name 'draw_svg_from_npz'` → Use `svg_conversion.py` instead
- ❌ `.svg` looks wrong → Make sure you're using the official `svg_conversion.py` from `tools/`
- ❌ Missing `.svg`? → Use `virtual_sketch_gui.py` or run `tools/svg_conversion.py` manually on `.npz`

---

## 🤝 Contributing

Feel free to open issues or pull requests if you encounter bugs or want to improve the Windows support!

---

## 📁 Folder Structure Suggestion

```
virtual_sketching/
├── sample_inputs/
├── tools/
├── outputs/
├── virtual_sketch_gui.py
├── runme.bat
├── README.md
└── WINDOWS_INSTALL_GUIDE.md   ← You are here
```

---

Made with ❤️ by the community to help Windows users get started!

