import os
import torch
import numpy as np
import soundfile as sf
import sounddevice as sd
import customtkinter as ctk
from tkinter import filedialog, messagebox
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# DeepFilterNet imports
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

# -------------------- PLOTTING HELPERS --------------------
def plot_waveform_and_spec(ax_wav, ax_spec, audio, sr, title, cmap):
    ax_wav.clear()
    ax_spec.clear()

    # Waveform
    t = np.linspace(0, len(audio) / sr, len(audio))
    ax_wav.plot(t, audio, color="cyan", linewidth=0.8)
    ax_wav.set_title(f"{title} - Waveform", color="white", fontsize=11)
    ax_wav.set_xlabel("Time (s)", color="white")
    ax_wav.set_ylabel("Amplitude", color="white")
    ax_wav.tick_params(colors="white")

    # Spectrogram
    n_fft = 1024
    hop = n_fft // 4
    w = torch.hann_window(n_fft)
    spec = torch.stft(torch.tensor(audio), n_fft, hop, window=w, return_complex=True)
    spec_db = 20 * torch.log10(torch.clamp(spec.abs(), min=1e-12))
    ax_spec.imshow(spec_db.numpy(), aspect="auto", origin="lower", cmap=cmap)
    ax_spec.set_title(f"{title} - Spectrogram", color="white", fontsize=11)
    ax_spec.set_xlabel("Time", color="white")
    ax_spec.set_ylabel("Frequency (Hz)", color="white")
    ax_spec.tick_params(colors="white")

def compute_metrics(original, enhanced):
    min_len = min(len(original), len(enhanced))
    original = original[:min_len]
    enhanced = enhanced[:min_len]
    orig_bin = (np.abs(original) > 0.02).astype(int)
    enh_bin = (np.abs(enhanced) > 0.02).astype(int)
    acc = accuracy_score(orig_bin, enh_bin) * 100
    prec = precision_score(orig_bin, enh_bin, zero_division=0) * 100
    rec = recall_score(orig_bin, enh_bin, zero_division=0) * 100
    f1 = f1_score(orig_bin, enh_bin, zero_division=0) * 100
    return acc, prec, rec, f1

def plot_metrics_bar(ax, acc, prec, rec, f1):
    ax.clear()
    metrics = ["Accuracy", "Precision", "Recall", "F1-score"]
    values = [acc, prec, rec, f1]
    bars = ax.bar(metrics, values, color=["#00BFFF", "#2ECC71", "#F1C40F", "#E67E22"])
    ax.set_ylim(0, 100)
    ax.set_title("Model Performance (%)", color="white", fontsize=12)
    ax.tick_params(colors="white")
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, height + 1, f"{height:.1f}%", ha="center", color="white")

# -------------------- AUDIO PLAYBACK --------------------
def play_audio(audio):
    if audio is not None:
        sd.stop()
        sd.play(audio, sample_rate)
    else:
        messagebox.showwarning("Warning", "No audio loaded!")

def stop_audio():
    sd.stop()

# -------------------- MAIN ENHANCE FUNCTION --------------------
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

        # Plot main visualizations
        plot_waveform_and_spec(ax1_wav, ax1_spec, noisy_audio, sr, "🎧 Noisy Audio", cmap="plasma")
        plot_waveform_and_spec(ax2_wav, ax2_spec, enhanced_audio, sr, "✨ Enhanced Audio", cmap="magma")

        # Compute and plot metrics
        acc, prec, rec, f1 = compute_metrics(noisy_audio, enhanced_audio)
        plot_metrics_bar(ax_metrics, acc, prec, rec, f1)
        canvas.draw()

        footer_label.configure(
            text=f"✅ Accuracy: {acc:.2f}% | Precision: {prec:.2f}% | Recall: {rec:.2f}% | F1-score: {f1:.2f}%"
        )

        messagebox.showinfo("Success", f"Enhanced audio saved at:\n{output_path}\n\n"
                                       f"Accuracy: {acc:.2f}%\nPrecision: {prec:.2f}%\nRecall: {rec:.2f}%\nF1: {f1:.2f}%")

    except Exception as e:
        messagebox.showerror("Error", str(e))
        print("Error:", e)

# -------------------- GUI DESIGN --------------------
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

root = ctk.CTk()
root.title("🎵 DeepFilterNet2 CPU Audio Enhancer")
root.geometry("1300x900")
root.minsize(1100, 800)

title_label = ctk.CTkLabel(root, text="🎧 DeepFilterNet2 Offline Audio Enhancer", font=("Arial Rounded MT Bold", 26))
title_label.pack(pady=12)

# Buttons
btn_frame = ctk.CTkFrame(root, corner_radius=15)
btn_frame.pack(pady=8, padx=20, fill="x")

btn_select = ctk.CTkButton(btn_frame, text="📂 Select & Enhance Audio", command=enhance_audio, font=("Arial", 14))
btn_select.grid(row=0, column=0, padx=10, pady=10)
btn_play_noisy = ctk.CTkButton(btn_frame, text="▶ Play Noisy", command=lambda: play_audio(noisy_audio))
btn_play_noisy.grid(row=0, column=1, padx=10, pady=10)
btn_play_enhanced = ctk.CTkButton(btn_frame, text="🎵 Play Enhanced", command=lambda: play_audio(enhanced_audio))
btn_play_enhanced.grid(row=0, column=2, padx=10, pady=10)
btn_stop = ctk.CTkButton(btn_frame, text="⏹ Stop", fg_color="#E74C3C", hover_color="#C0392B", command=stop_audio)
btn_stop.grid(row=0, column=3, padx=10, pady=10)

# Figure Layout (3x2 grid -> 5 plots total)
fig = plt.figure(figsize=(11, 7))
fig.patch.set_facecolor("#1a1a1a")
gs = fig.add_gridspec(3, 2, hspace=0.5, wspace=0.3)

ax1_wav = fig.add_subplot(gs[0, 0])
ax1_spec = fig.add_subplot(gs[0, 1])
ax2_wav = fig.add_subplot(gs[1, 0])
ax2_spec = fig.add_subplot(gs[1, 1])
ax_metrics = fig.add_subplot(gs[2, :])

for ax in (ax1_wav, ax1_spec, ax2_wav, ax2_spec, ax_metrics):
    ax.set_facecolor("#1a1a1a")

canvas = FigureCanvasTkAgg(fig, master=root)
canvas.get_tk_widget().pack(fill="both", expand=True, padx=20, pady=10)

footer_label = ctk.CTkLabel(root, text="Model Performance: Waiting for input...", font=("Arial", 13))
footer_label.pack(pady=8)

root.mainloop()