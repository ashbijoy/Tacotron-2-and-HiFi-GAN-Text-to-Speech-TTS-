import torch
import numpy as np
import sys
import os
from scipy.io.wavfile import write

# Add paths for Tacotron 2 and HiFi-GAN
sys.path.append(os.path.abspath("./tacotron2"))
sys.path.append(os.path.abspath("./hifi-gan"))

from tacotron2.model import Tacotron2
from tacotron2.hparams import create_hparams
from tacotron2.text import text_to_sequence
from hifi_gan.models import Generator as HiFiGAN
from hifi_gan.env import AttrDict

# Load Tacotron 2 Model (Updated for PyTorch version)
def load_tacotron2(model_path):
    hparams = create_hparams()
    hparams.sampling_rate = 22050  # Ensure this matches HiFi-GAN's expected rate
    model = Tacotron2(hparams)
    checkpoint = torch.load(model_path, map_location='cpu')
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    return model, hparams

# Load HiFi-GAN Model (Unchanged)
def load_hifigan(model_path):
    checkpoint = torch.load(model_path, map_location='cpu')
    config = AttrDict(checkpoint['config'])
    model = HiFiGAN(config)
    model.load_state_dict(checkpoint['generator'], strict=False)
    model.eval()
    model.remove_weight_norm()
    return model

# Convert text to mel spectrogram (Updated for PyTorch Tacotron2)
def text_to_mel(text, model, hparams):
    sequence = torch.tensor(text_to_sequence(text, ['english_cleaners'])).unsqueeze(0).long()
    with torch.no_grad():
        mel_outputs_postnet, _, _ = model.inference(sequence)
        # Use mel_outputs_postnet for better quality
        return mel_outputs_postnet

# Convert mel spectrogram to waveform (Unchanged)
def mel_to_audio(mel, model):
    with torch.no_grad():
        audio = model(mel).squeeze().cpu().numpy()
    return audio

# Main function (Updated for proper device handling)
def generate_speech(text, tacotron_model, hifigan_model, hparams, output_wav="output.wav"):
    # Ensure models are on CPU (or GPU if available)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tacotron_model = tacotron_model.to(device)
    hifigan_model = hifigan_model.to(device)
    
    # Generate mel spectrogram
    mel = text_to_mel(text, tacotron_model, hparams)
    
    # HiFi-GAN expects specific input shape
    if len(mel.shape) == 2:
        mel = mel.unsqueeze(0)
    
    # Generate audio
    audio = mel_to_audio(mel.to(device), hifigan_model)
    
    # Normalize and save
    audio = (audio / np.max(np.abs(audio))) * 32767
    write(output_wav, hparams.sampling_rate, audio.astype(np.int16))
    print(f"✅ Speech saved to {output_wav}")

if __name__ == "__main__":
    # Load models
    tacotron2, tacotron_hparams = load_tacotron2("pretrained_models/tacotron2.pt")
    hifigan = load_hifigan("pretrained_models/hifigan.pt")

    # Example text
    text = "Hello, this is a test of the Tacotron 2 text-to-speech system."

    # Generate speech
    generate_speech(text, tacotron2, hifigan, tacotron_hparams)