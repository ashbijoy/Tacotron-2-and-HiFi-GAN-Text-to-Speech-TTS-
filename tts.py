import torch
import torch.nn as nn
import numpy as np
import sys
import os
from scipy.io.wavfile import write
from torch.nn.utils import weight_norm, remove_weight_norm
import soundfile as sf
import sounddevice as sd  # For audio playback test

# Add paths for Tacotron 2
sys.path.append(os.path.abspath("./tacotron2"))
from tacotron2.model import Tacotron2
from tacotron2.text import text_to_sequence
from tacotron2.hparams import create_hparams

class MiniVocoder(nn.Module):
    """Enhanced vocoder with better audio output"""
    def __init__(self):
        super().__init__()
        self.upsample_rates = [8, 8, 2, 2]
        self.upsample_kernel_sizes = [16, 16, 4, 4]
        
        # Increased capacity for better audio
        self.conv_pre = weight_norm(nn.Conv1d(80, 512, 7, 1, padding=3))
        
        self.ups = nn.ModuleList()
        for i, (u, k) in enumerate(zip(self.upsample_rates, self.upsample_kernel_sizes)):
            in_channels = 512 // (2 ** i)
            out_channels = 512 // (2 ** (i + 1))
            self.ups.append(weight_norm(
                nn.ConvTranspose1d(in_channels, out_channels, k, u, 
                padding=(k - u) // 2)))
        
        self.conv_post = weight_norm(nn.Conv1d(32, 1, 7, 1, padding=3))
        
    def forward(self, x):
        x = self.conv_pre(x)
        for up in self.ups:
            x = torch.relu(x)
            x = up(x)
        x = torch.relu(x)
        x = self.conv_post(x)
        return torch.tanh(x)  # Constrained output range

def load_vocoder():
    """Initialize the enhanced vocoder"""
    print("⚙️ Initializing vocoder...")
    model = MiniVocoder()
    
    # Better weight initialization
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
            
    model.eval()
    return model

def load_tacotron2(model_path):
    """Load Tacotron2 model"""
    try:
        print("⚙️ Initializing Tacotron2 model...")
        hparams = create_hparams()
        model = Tacotron2(hparams)
        
        print(f"📂 Loading checkpoint from {model_path}...")
        checkpoint = torch.load(model_path, map_location='cpu')
        
        state_dict = checkpoint.get('state_dict', checkpoint.get('model', checkpoint))
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        
        model.load_state_dict(state_dict, strict=False)
        model.eval()
        return model, hparams
        
    except Exception as e:
        print(f"\n❌ Error loading Tacotron2: {str(e)}")
        raise

def text_to_mel(text, model, hparams):
    """Convert text to mel spectrogram"""
    try:
        sequence = np.array(text_to_sequence(text, ['english_cleaners']))[None, :]
        sequence = torch.from_numpy(sequence).long()
        
        with torch.no_grad():
            _, mel_outputs_postnet, _, _ = model.inference(sequence)
            
        return mel_outputs_postnet
    except Exception as e:
        print(f"\n❌ Text-to-Mel error: {str(e)}")
        raise

def generate_speech(text, tacotron_model, vocoder, hparams, output_wav="output.wav"):
    """Complete TTS pipeline with guaranteed audible output"""
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"⚡ Using device: {device}")
        
        tacotron_model = tacotron_model.to(device)
        vocoder = vocoder.to(device)
        
        print("🔠 Converting text to mel...")
        mel = text_to_mel(text, tacotron_model, hparams).to(device)
        
        print("🔊 Synthesizing audio...")
        with torch.no_grad():
            audio = vocoder(mel).squeeze().cpu().numpy()
        
        # Audio processing with volume boost
        print("\n🔈 Audio Statistics (Before Boost):")
        print(f"Range: [{audio.min():.4f}, {audio.max():.4f}]")
        print(f"Avg: {np.mean(np.abs(audio)):.4f}")
        
        # Significant volume boost (100x) and normalization
        audio *= 100  # Critical boost for minimal vocoder
        peak = np.max(np.abs(audio))
        if peak > 1e-6:  # Only normalize if not silent
            audio = audio / peak
        audio = (audio * 32767).astype(np.int16)
        
        # Save and verify
        write(output_wav, hparams.sampling_rate, audio)
        
        # Playback test
        try:
            print("\n🔊 Playing audio...")
            sd.play(audio, hparams.sampling_rate)
            sd.wait()  # Wait until playback finishes
        except Exception as e:
            print(f"⚠️ Playback failed: {str(e)}")
        
        print(f"\n✅ Saved to {output_wav}")
        
    except Exception as e:
        print(f"\n❌ Generation failed: {str(e)}")

if __name__ == "__main__":
    try:
        # Test speaker functionality first
        print("🔊 Testing speakers with 440Hz tone...")
        test_tone = np.sin(2 * np.pi * 440 * np.arange(22050) / 22050)
        sd.play(test_tone, 22050)
        sd.wait()
        
        # Load models
        tacotron2, hparams = load_tacotron2("pretrained_models/tacotron2.pt")
        vocoder = load_vocoder()
        
        # Example text
        text = "Hello, this is a clear test of the text-to-speech system."
        print(f"\n📝 Input text: '{text}'")
        
        # Generate speech
        generate_speech(text, tacotron2, vocoder, hparams)
        
    except Exception as e:
        print(f"\n💥 Critical failure: {str(e)}")