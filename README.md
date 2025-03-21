# Tacotron 2 + HiFi-GAN Text-to-Speech (TTS)

This repository contains a text-to-speech (TTS) system using **Tacotron 2** for generating mel-spectrograms and **HiFi-GAN** for vocoding (converting spectrograms to audio). The output is a high-quality speech waveform.

## Installation

### 1. Install Dependencies
Ensure you have Python 3.8+ installed. Then install the required dependencies:
```bash
pip install torch torchaudio numpy matplotlib scipy
```

### 2. Clone Repositories
Clone the required repositories:
```bash
git clone https://github.com/NVIDIA/tacotron2.git
git clone https://github.com/jik876/hifi-gan.git
```

### 3. Download Pretrained Models
Since some previous links may be outdated, download the models manually from Google Drive:

#### **Tacotron 2 Model:**
Download from: [Tacotron 2 Pretrained Model](https://drive.google.com/drive/folders/1X20XhHWa7g3W9mDXIrPCN6Pkiuq42rkF)  
Save as: `pretrained_models/tacotron2.pt`

#### **HiFi-GAN Model:**
Download from: [HiFi-GAN Pretrained Model](https://drive.google.com/drive/folders/1-eEYtb8ocRJPg9aiJrpVhTnd06nDCXg5)  
Save as: `pretrained_models/hifigan.pt`

## Usage

### **1. Run the Python Script**
After downloading the models, execute the script to generate speech:
```bash
python tts.py
```

### **2. Example Output**
The script will generate an audio file named `output.wav`, containing the synthesized speech.

## Repository Structure
```
├── tacotron2/             # Tacotron 2 repository
├── hifi-gan/              # HiFi-GAN repository
├── pretrained_models/     # Directory for storing downloaded models
│   ├── tacotron2.pt       # Pretrained Tacotron 2 model
│   ├── hifigan.pt         # Pretrained HiFi-GAN model
├── tts.py                 # Main script for generating speech
├── README.md              # Documentation
```

## Troubleshooting

- If you encounter a **404 Not Found** error when downloading models, manually download them from the links above.
- Ensure you have installed all dependencies before running the script.
- If audio output is distorted, try using a different vocoder such as **WaveGlow** instead of HiFi-GAN.

## Author
**Ashbijoy**  
GitHub: [@ashbijoy](https://github.com/ashbijoy)

## Credits
- [NVIDIA Tacotron 2](https://github.com/NVIDIA/tacotron2)
- [HiFi-GAN](https://github.com/jik876/hifi-gan)

