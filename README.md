# Adveserial-Attacks

`pip install -r requirements.txt`

## Demo
To listen to the results, please check the [demo](https://github.com/ppriyank/Adveserial-Attacks/tree/master/demo) directory. Each directory has the corresponding transcripts for Google and IBM in seperate folders for reference.
* original.wav - The original audio clip
* [Existing Attacks](https://github.com/ppriyank/Adveserial-Attacks/tree/master/demo/Existing%20Attacks%20) - Adversarial attacks on original.wav based on [2018](https://people.eecs.berkeley.edu/~daw/papers/audio-dls18.pdf) [github](https://github.com/carlini/audio_adversarial_examples) and [2019](http://proceedings.mlr.press/v97/qin19a/qin19a.pdf) [github](https://github.com/tensorflow/cleverhans/tree/master/examples/adversarial_asr) papers
* [Our Basic Attacks](https://github.com/ppriyank/Adveserial-Attacks/tree/master/demo/Our%20Basic%20Attack) - Basic attacks on original.wav by simple audio perturbations.
* [Our Imporved Attack](https://github.com/ppriyank/Adveserial-Attacks/tree/master/demo/Our%20Improved%20Attack) - Our proposed robust (ourproposed_attack(re-recorded).wav) adversarial attacks on original.wav.
* [Room Simulator Samples](https://github.com/ppriyank/Adveserial-Attacks/tree/master/demo/Room%20Simulators%20Samples) - Room simulation audio clips used to make robust attacks.
* [pdfs](https://github.com/ppriyank/Adveserial-Attacks/tree/master/pdfs) - Final report and presentation slides provided for detailed analysis


## References for Audio Attacks and Theory :  
* [Noises](https://github.com/jfsantos/maracas/blob/master/maracas/maracas.py)  
* [FFT & Audio Manipulation](https://timsainburg.com/noise-reduction-python.html)  
* [pydub](https://github.com/jiaaro/pydub)   
* [librosa Manipulations](https://librosa.github.io/librosa/generated/librosa.effects.preemphasis.html)  
* [2019](http://proceedings.mlr.press/v97/qin19a/qin19a.pdf) [github](https://github.com/tensorflow/cleverhans/tree/master/examples/adversarial_asr)
* [Pytorch Deep Speech](https://github.com/SeanNaren/deepspeech.pytorch)
* [2018](https://people.eecs.berkeley.edu/~daw/papers/audio-dls18.pdf) [github](https://github.com/carlini/audio_adversarial_examples)
* [SimBA](https://arxiv.org/abs/1905.07121) [github](https://github.com/cg563/simple-blackbox-attack)







