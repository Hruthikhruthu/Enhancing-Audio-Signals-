# Audio Enhancement using PyTorch (DeepFilterNet)

This project performs **audio denoising and enhancement** using **PyTorch** and **DeepFilterNet**.  
It supports audio preprocessing, deep learning–based enhancement, and optional GUI interaction.

---

## Features

- Audio denoising and enhancement
- PyTorch-based inference
- CPU-only support (no GPU required)
- Supports WAV and common audio formats
- Optional GUI using `customtkinter`

---

## Requirements

- Python 3.9 or later
- pip
- Windows / Linux / macOS

---

## Setup Instructions

### 1. Create Virtual Environment

```bash
python -m venv venv


2. Virtual Environment
      venv\Scripts\activate


3. Install Dependencies

   pip install torch==2.3.0+cpu torchaudio==2.3.0+cpu torchvision==0.18.0+cpu \
   -f https://download.pytorch.org/whl/torch_stable.html

 pip install matplotlib soundfile sounddevice customtkinter deepfilternet numpy pydub
 


4.Run the project
     python enhance.py


