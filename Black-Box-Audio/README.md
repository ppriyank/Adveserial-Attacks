# Targeted Adversarial Examples for Black Box Audio Systems

Sample code to let you create your own adversarial examples! [Paper linked here](https://arxiv.org/abs/1805.07820)

## Setup

### Prerequisite

* Anaconda

### Commands

Create a conda environment with python 3.6. Activate this environment, and do the following.
```
pip install -r requirements.txt
git clone https://github.com/mozilla/DeepSpeech.git
cd DeepSpeech
pip install $(python util/taskcluster.py --decoder)
git checkout tags/v0.1.0
cd ..
wget https://github.com/mozilla/DeepSpeech/releases/download/v0.1.0/deepspeech-0.1.0-models.tar.gz
tar -xzf deepspeech-0.1.0-models.tar.gz
rm deepspeech-0.1.0-models.tar.gz
python make_checkpoint.py
```

## Run

On NYU prince, you may to load the following modules.

```
module load anaconda3/5.3.1
module load cudnn/9.0v7.0.5
module load cuda/9.0.176
```

Now create run an attack with the following format.
```
python3 run_audio_attack.py input_file.wav "target phrase"
```
For example, `python3 run_audio_attack.py sample_input.wav "hello world"`. 

You can also listen to pre-created audio samples in the [samples](samples/) directory. Each original/adversarial pair is denoted by a leading number, with model transcriptions as the title.

## Changes Made

In `requirements.txt`, line 3, `tensorflow-gpu` to `tensorflow-gpu==1.8.0`.
In `run_audio_attack.py`, line 246, `return itr > self.max_iterations` to `return itr < self.max_iters`.
