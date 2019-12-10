
Pytorch Implementation of the paper:  
**Imperceptible, Robust, and Targeted Adversarial Examples for Automatic Speech Recognition**  
*Yao Qin, Nicholas Carlini, Ian Goodfellow, Garrison Cottrell, Colin Raffel*    
[Paper](http://proceedings.mlr.press/v97/qin19a/qin19a.pdf)  

Tensorflow implementation available here :   
[GITHUB](https://github.com/tensorflow/cleverhans/tree/master/examples/adversarial_asr)


Set up a Audio to Speech (ASR) model first   (if you get stuck with the error of linker Read Troble shooting below (`python setup.py install`) ) 
[GITHUB](https://github.com/SeanNaren/deepspeech.pytorch)  

If you are on a server, and don't have have sudo rights. Just leave CTC loss part. Its not worth the effort. 
else you will need : `CTCLoss` as well [follow this](https://github.com/SeanNaren/warp-ctc)  

Download a pretrained model : `librispeech_pretrained_v2.pth`   
librispeech short 10 seconds audio clips : [here](https://github.com/ppriyank/Adveserial-Attacks/tree/master/audio-dataset/short-audio)  

## Installation : 
`pip install -r requirements.txt`  


module load gcc/9.1.0


## Evaluate
`python transcribe.py --model-path saved_model/... --audio-path 8.wav`   

## Creating noise (stage1)
(module load gcc/9.1.0) (gcc 9.1.0 required)  
`python main --model-path saved_model/... -x=[..,..] -t "test"`

 
## Troubleshooting 
All issues tried to be addressed [here](https://github.com/ppriyank/Adveserial-Attacks/tree/master/paper2018)

