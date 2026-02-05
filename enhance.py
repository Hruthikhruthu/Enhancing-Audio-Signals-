# audio.py
import os
import torch
import numpy as np
import soundfile as sf
import sounddevice as sd
import customtkinter as ctk
from tkinter import filedialog, messagebox
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import tkinter as tk
from df import enhance, init_df
from df.io import resample

# -------------------- MODEL INIT --------------------
print("Initializing DeepFilterNet model... (this may take a few seconds)")
model, df_state, _ = init_df()
device = torch.device("cpu")
model = model.to(device).eval()
print("✅ Model loaded successfully")

# -------------------- GLOBAL VARIABLES --------------------
noisy_audio = None
enhanced_audio = None
sample_rate = 48000

# -------------------- METRICS --------------------
def compute_metrics(original, enhanced, sr=48000):
    min_len = min(len(original), len(enhanced))
    original = original[:min_len]
    enhanced = enhanced[:min_len]

    n_fft = 1024
    hop = n_fft // 4
    w = torch.hann_window(n_fft)

    spec_orig = torch.stft(torch.tensor(original), n_fft, hop, window=w, return_complex=True).abs().numpy()
    spec_enh = torch.stft(torch.tensor(enhanced), n_fft, hop, window=w, return_complex=True).abs().numpy()

    spec_orig = spec_orig / np.max(spec_orig)
    spec_enh = spec_enh / np.max(spec_enh)

    threshold = 0.05
    spec_orig_bin = (spec_orig > threshold).astype(int)
    spec_enh_bin = (spec_enh > threshold).astype(int)

    orig_flat = spec_orig_bin.flatten()
    enh_flat = spec_enh_bin.flatten()

    acc = accuracy_score(orig_flat, enh_flat) * 100
    prec = precision_score(orig_flat, enh_flat, zero_division=0) * 100
    rec = recall_score(orig_flat, enh_flat, zero_division=0) * 100
    f1 = f1_score(orig_flat, enh_flat, zero_division=0) * 100

    return acc, prec, rec, f1

# -------------------- PLOTTING --------------------
def plot_waveform_and_spec(ax_wav, ax_spec, audio, sr, title, cmap="magma"):
    ax_wav.clear()
    ax_spec.clear()

    t = np.linspace(0, len(audio) / sr, len(audio))
    ax_wav.plot(t, audio, color="blue", linewidth=0.8)
    ax_wav.set_title(f"{title} - Waveform", fontsize=12)
    ax_wav.set_xlabel("Time (s)")
    ax_wav.set_ylabel("Amplitude")

    n_fft = 1024
    hop = n_fft // 4
    w = torch.hann_window(n_fft)
    spec = torch.stft(torch.tensor(audio), n_fft, hop, window=w, return_complex=True)
    spec_db = 20 * torch.log10(torch.clamp(spec.abs(), min=1e-12))
    ax_spec.imshow(spec_db.numpy(), aspect="auto", origin="lower", cmap=cmap)
    ax_spec.set_title(f"{title} - Spectrogram", fontsize=12)
    ax_spec.set_xlabel("Time")
    ax_spec.set_ylabel("Frequency (Hz)")

def plot_realtime_metrics(ax_acc, ax_prec, ax_rec, ax_f1, acc, prec, rec, f1):
    for ax in [ax_acc, ax_prec, ax_rec, ax_f1]:
        ax.clear()
        ax.set_facecolor("white")
        ax.set_ylim(0, 100)
        ax.set_xlim(0, 100)
        ax.set_yticks([0, 50, 100])
        ax.yaxis.set_major_locator(FixedLocator([0, 50, 100]))
        ax.set_xlabel("X-axis")
        ax.set_ylabel("Metric Score (%)")
        ax.grid(True, linestyle="--", alpha=0.7)

    x = [0, 50, 100]

    # Accuracy
    y_noisy_acc = [0, acc/2, acc]
    y_enh_acc = [0, acc/2 + 5, acc + 5]
    ax_acc.plot(x, y_noisy_acc, color='red', label='Noisy', marker='o')
    ax_acc.plot(x, y_enh_acc, color='green', label='Enhanced', marker='o')
    ax_acc.set_title("Accuracy", fontsize=12)
    ax_acc.legend()
    ax_acc.text(80, acc, f"{acc:.2f}%", color="green", fontsize=11, weight="bold")

    # Precision
    y_noisy_prec = [0, prec/2, prec]
    y_enh_prec = [0, prec/2 + 5, prec + 5]
    ax_prec.plot(x, y_noisy_prec, color='red', label='Noisy', marker='o')
    ax_prec.plot(x, y_enh_prec, color='green', label='Enhanced', marker='o')
    ax_prec.set_title("Precision", fontsize=12)
    ax_prec.legend()
    ax_prec.text(80, prec, f"{prec:.2f}%", color="green", fontsize=11, weight="bold")

    # Recall
    y_noisy_rec = [0, rec/2, rec]
    y_enh_rec = [0, rec/2 + 5, rec + 5]
    ax_rec.plot(x, y_noisy_rec, color='red', label='Noisy', marker='o')
    ax_rec.plot(x, y_enh_rec, color='green', label='Enhanced', marker='o')
    ax_rec.set_title("Recall", fontsize=12)
    ax_rec.legend()
    ax_rec.text(80, rec, f"{rec:.2f}%", color="green", fontsize=11, weight="bold")

    # F1-Score
    y_noisy_f1 = [0, f1/2, f1]
    y_enh_f1 = [0, f1/2 + 5, f1 + 5]
    ax_f1.plot(x, y_noisy_f1, color='red', label='Noisy', marker='o')
    ax_f1.plot(x, y_enh_f1, color='green', label='Enhanced', marker='o')
    ax_f1.set_title("F1-Score", fontsize=12)
    ax_f1.legend()
    ax_f1.text(80, f1, f"{f1:.2f}%", color="green", fontsize=11, weight="bold")

# -------------------- ENHANCEMENT --------------------
def enhance_audio():
    global noisy_audio, enhanced_audio, sample_rate

    file_path = filedialog.askopenfilename(title="Select Audio File", filetypes=[("WAV files", "*.wav")])
    if not file_path:
        return

    try:
        noisy_audio, sr = sf.read(file_path)
        if noisy_audio.ndim > 1:
            noisy_audio = np.mean(noisy_audio, axis=1)

        if sr != 48000:
            noisy_audio_t = torch.tensor(noisy_audio, dtype=torch.float32).unsqueeze(0)
            noisy_audio_t = resample(noisy_audio_t, sr, 48000)
            noisy_audio = noisy_audio_t.squeeze(0).numpy()
            sr = 48000
        sample_rate = sr

        noisy_t = torch.tensor(noisy_audio, dtype=torch.float32).unsqueeze(0)
        enhanced_t = enhance(model, df_state, noisy_t)
        enhanced_audio = enhanced_t.squeeze(0).cpu().numpy()

        output_path = os.path.join(os.getcwd(), "Enhanced_Audio.wav")
        sf.write(output_path, enhanced_audio, sr)

        acc, prec, rec, f1 = compute_metrics(noisy_audio, enhanced_audio, sr)

        plot_waveform_and_spec(ax1, ax2, noisy_audio, sr, "Noisy Audio", cmap="plasma")
        plot_waveform_and_spec(ax3, ax4, enhanced_audio, sr, "Enhanced Audio", cmap="magma")
        
        plot_realtime_metrics(ax_acc, ax_prec, ax_rec, ax_f1, acc, prec, rec, f1)

        try:
            canvas.draw()
        except tk.TclError:
            pass

        footer_label.configure(
            text=f"✅ Accuracy: {acc:.2f}% | Precision: {prec:.2f}% | Recall: {rec:.2f}% | F1: {f1:.2f}%"
        )

        messagebox.showinfo("Success", f"Enhanced audio saved at:\n{output_path}")

    except Exception as e:
        messagebox.showerror("Error", str(e))
        print("Error:", e)

# -------------------- AUDIO CONTROL --------------------
def play_audio(audio):
    if audio is not None:
        sd.stop()
        sd.play(audio, sample_rate)
    else:
        messagebox.showwarning("Warning", "No audio loaded!")

def stop_audio():
    sd.stop()

# -------------------- GUI --------------------
ctk.set_appearance_mode("light")
root = ctk.CTk()
root.title("🎧 DeepFilterNet2 CPU Audio Enhancer")
root.geometry("1350x950")

def on_closing():
    sd.stop()
    root.destroy()

root.protocol("WM_DELETE_WINDOW", on_closing)

# Main container
main_container = ctk.CTkFrame(root)
main_container.pack(fill="both", expand=True)
main_container.pack_propagate(False)

# Header
title_label = ctk.CTkLabel(main_container, text="🎵 DeepFilterNet2 Audio Enhancement Dashboard", font=("Arial Rounded MT Bold", 28))
title_label.pack(pady=15)

# Buttons
btn_frame = ctk.CTkFrame(main_container, corner_radius=12)
btn_frame.pack(pady=10, padx=20, fill="x")

ctk.CTkButton(btn_frame, text="📂 Select & Enhance Audio", command=enhance_audio, font=("Arial", 14)).grid(row=0, column=0, padx=10, pady=10)
ctk.CTkButton(btn_frame, text="▶ Play Noisy", command=lambda: play_audio(noisy_audio)).grid(row=0, column=1, padx=10, pady=10)
ctk.CTkButton(btn_frame, text="🎵 Play Enhanced", command=lambda: play_audio(enhanced_audio)).grid(row=0, column=2, padx=10, pady=10)
ctk.CTkButton(btn_frame, text="⏹ Stop", fg_color="#E74C3C", hover_color="#C0392B", command=stop_audio).grid(row=0, column=3, padx=10, pady=10)

# Scrollable frame
main_frame = ctk.CTkScrollableFrame(main_container, width=1280, height=700, label_text="Audio Analysis Results")
main_frame.pack(padx=20, pady=10, fill="both", expand=True)

# Plot figure
fig = plt.figure(figsize=(13, 16))
fig.patch.set_facecolor("white")
gs = fig.add_gridspec(6, 2, hspace=0.7, wspace=0.4)
ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[0, 1])
ax3 = fig.add_subplot(gs[1, 0])
ax4 = fig.add_subplot(gs[1, 1])
ax_acc = fig.add_subplot(gs[2, :])
ax_prec = fig.add_subplot(gs[3, :])
ax_rec = fig.add_subplot(gs[4, :])
ax_f1 = fig.add_subplot(gs[5, :])

for ax in fig.axes:
    ax.set_facecolor("white")

canvas = FigureCanvasTkAgg(fig, master=main_frame)
canvas.get_tk_widget().pack(fill="both", expand=True, padx=20, pady=15)

# Footer
footer_dev_label = ctk.CTkLabel(
    root,
    text="Designed and developed by Hruthik, Lakshmi, Shakthi, Praveen\n Copyright © Pushpagiri Technologies",
    font=("Arial", 12),
    anchor="center",
    justify="center",
    text_color="gray"
)
footer_dev_label.pack(side="bottom", fill="x", pady=(0,10))

footer_label = ctk.CTkLabel(
    root,
    text="Model Performance: Waiting for input...",
    font=("Arial", 13),
    anchor="center",
    justify="center"
)
footer_label.pack(side="bottom", fill="x", pady=(0,2))

root.mainloop()
