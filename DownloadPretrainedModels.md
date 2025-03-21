#Download the pretrained models
mkdir pretrained_models
wget -O pretrained_models/tacotron2.pt https://github.com/NVIDIA/tacotron2/releases/download/v1.0/tacotron2_statedict.pt
wget -O pretrained_models/hifigan.pt https://github.com/jik876/hifi-gan/releases/download/v1.0/g_02500000

#and then run the script