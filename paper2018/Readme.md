
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

Prince:

```
module load cmake/intel/3.11.4
module load gcc/9.1.0
```
conda install gcc==4.8.5

add at the top of `CMakeLists.txt`
```
INCLUDE(CMakeForceCompiler)
CMAKE_FORCE_C_COMPILER(gcc GNU)
CMAKE_FORCE_CXX_COMPILER(g++ GNU)
```


```
export PATH="~/anaconda3/bin:$PATH"
cmake -DCMAKE_CXX_COMPILER=/usr/bin/g++ -DCMAKE_C_COMPILER=/usr/bin/gcc ..
conda install gxx_linux-64
```

~/anaconda3/envs/pathak/bin/gcc
cmake -DCMAKE_CXX_COMPILER=/home/pp1953/anaconda3/envs/pathak/bin/g++ -DCMAKE_C_COMPILER=/home/pp1953/anaconda3/envs/pathak/bin/gcc ..



conda install gcc==4.8.5
gcc --version 
g++ --version 

export CC=`which gcc`
export CXX=`which g++`


export CC=/usr/bin/gcc
export CXX=/usr/bin/g++

echo 'export CXX=/usr/bin/g++' >> ~/.bashrc
echo 'export CC=/usr/bin/gcc' >> ~/.bashrc

export CXX=/c/MinGW/bin/g++.exe


module load cudnn/9.0v7.3.0.29 
module load cuda/9.0.176

python 



ln -s /usr/local/bin/gcc-4.8 cc
ln -s /usr/local/bin/gcc-4.8 gcc
ln -s /usr/local/bin/c++-4.8 c++
ln -s /usr/local/bin/g++-4.8 g++


INCLUDE(CMakeForceCompiler)
CMAKE_FORCE_C_COMPILER(gcc GNU)
CMAKE_FORCE_CXX_COMPILER(g++ GNU)






git clone https://github.com/SeanNaren/warp-ctc.git
cd warp-ctc
mkdir build; cd build
cmake ..
make