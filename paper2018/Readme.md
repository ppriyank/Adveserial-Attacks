
Pytorch Implementation of the paper:  
**Audio Adversarial Examples: Targeted Attacks on Speech-to-Text**  
*Nicholas Carlini David Wagner*  

Tensorflow implementation available here :   
[GITHUB](https://github.com/carlini/audio_adversarial_examples)

Installation : 
`pip install -r requirements.txt`  

Set up a Audio to Speech (ASR) model first   (if you get stuck with the error of linker Read Troble shooting below (`python setup.py install`) ) 
[GITHUB](https://github.com/SeanNaren/deepspeech.pytorch)  


Download a pretrained model : `librispeech_pretrained_v2.pth`   
librispeech short 10 seconds audio clips : [here](https://github.com/ppriyank/Adveserial-Attacks/tree/master/audio-dataset/short-audio)  
Evaluate : `python transcribe.py --model-path saved_model/... --audio-path ../../../xyz.wav`   


## Troubleshooting 
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

