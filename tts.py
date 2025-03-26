import torch
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
from scipy.io.wavfile import write

# Add paths for Tacotron 2 and HiFi-GAN
sys.path.append(os.path.abspath("./tacotron2"))
sys.path.append(os.path.abspath("./hifi-gan"))


from tacotron2.hparams import create_hparams
from tacotron2.model import Tacotron2
from tacotron2.text import text_to_sequence
from hifi_gan.models import Generator as HiFiGAN
from hifi_gan.env import AttrDict

# Load Tacotron 2 Model
def load_tacotron2(model_path):
    hparams = create_hparams()
    model = Tacotron2(hparams)
    model.load_state_dict(torch.load(model_path, map_location='cpu'), strict=False)
    model.eval()
    return model, hparams

# Load HiFi-GAN Model
def load_hifigan(model_path):
    checkpoint = torch.load(model_path, map_location='cpu')
    config = AttrDict(checkpoint['config'])
    model = HiFiGAN(config)
    model.load_state_dict(checkpoint['generator'], strict=False)
    model.eval()
    model.remove_weight_norm()
    return model

# Convert text to mel spectrogram
def text_to_mel(text, model, hparams):
    sequence = np.array(text_to_sequence(text, ['english_cleaners']))[None, :]
    sequence = torch.from_numpy(sequence).long()
    with torch.no_grad():
        _, mel_outputs, _, _ = model.inference(sequence)
    return mel_outputs

# Convert mel spectrogram to waveform
def mel_to_audio(mel, model):
    with torch.no_grad():
        audio = model(mel).squeeze().cpu().numpy()
    return audio

# Main function
def generate_speech(text, tacotron_model, hifigan_model, hparams, output_wav="output.wav"):
    mel_spectrogram = text_to_mel(text, tacotron_model, hparams)
    audio = mel_to_audio(mel_spectrogram, hifigan_model)
    
    # Normalize and save audio
    audio = (audio / np.max(np.abs(audio))) * 32767
    write(output_wav, 22050, audio.astype(np.int16))
    print(f"✅ Speech saved to {output_wav}")

# Load models
tacotron2, tacotron_hparams = load_tacotron2("pretrained_models/tacotron2.pt")
hifigan = load_hifigan("pretrained_models/hifigan.pt")

# Example text
text = "Hello, this is a test of the Tacotron 2 text-to-speech system."

# Generate speech
generate_speech(text, tacotron2, hifigan, tacotron_hparams)
