
Pytorch Implementation of the paper:  
**Audio Adversarial Examples: Targeted Attacks on Speech-to-Text**  
*Nicholas Carlini David Wagner*  

Tensorflow implementation available here :   
[GITHUB](https://github.com/carlini/audio_adversarial_examples)


Set up a Audio to Speech (ASR) model first   (if you get stuck with the error of linker Read Troble shooting below (`python setup.py install`) ) 
[GITHUB](https://github.com/SeanNaren/deepspeech.pytorch)  

Note you need : `CTCLoss` as well [follow this](https://github.com/SeanNaren/warp-ctc)


Download a pretrained model : `librispeech_pretrained_v2.pth`   
librispeech short 10 seconds audio clips : [here](https://github.com/ppriyank/Adveserial-Attacks/tree/master/audio-dataset/short-audio)  

## Evaluate
`python transcribe.py --model-path saved_model/... --audio-path 8.wav`   

## Creating noise (stage1)
python main --model-path saved_model/... -x=[..,..] -t "test"

## Installation : 
`pip install -r requirements.txt`  

 

## Troubleshooting 
### CLC loss 
LOL, you are reading this
`python setup.py install`
erorr : 
`build/temp.linux-x86_64-3.7/torch/csrc/stub.o: file not recognized: file format not recognized`  

do the following 
* `mv anaconda/compiler_compat/ld anaconda/compiler_compat/ld-old` 
* change setup.py `extra_link_args=['-L/usr/lib/x86_64-linux-gnu/']`  
* In the last 2 g++ commands : change the following   
`g++ -pthread -shared /usr/bin/ld -L...`  
`gcc -pthread  /usr/bin/ld -Wl...`


### librosa  
if you face trouble importing : `import librosa`  
conda install -c conda-forge librosa
in addition to pip install librosa

### python-levenshtein 
conda install -c conda-forge python-levenshtein  



### NYU : 
CILVR

```
export PATH="/misc/vlgscratch4/LakeGroup/pathak/anaconda3/bin:$PATH"
conda activate pathak 
conda install -n pathak nb_conda_kernels

python 

import torch
from warpctc_pytorch import CTCLoss
ctc_loss = CTCLoss()
```
