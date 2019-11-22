import argparse
import numpy as np 

import torch
import torch.nn as nn
from warpctc_pytorch import CTCLoss

from scipy.io import wavfile
from model import DeepSpeech, supported_rnns

parser = argparse.ArgumentParser(description='idk :P')
parser.add_argument('-t', '--target-phrase', type=str, default='testing')
parser.add_argument('-x', '--input-audio-paths', type=list, default=[c for c in '[/scratch/pp1953/short-audio/2.wav]'])
args = parser.parse_args()


print(args)
args.input_audio_paths = ("".join(args.input_audio_paths[1:-1])).split(",")
print(args.input_audio_paths)

# Read audios 
audios = []
lengths = []
for path in args.input_audio_paths: 
	rate, data = wavfile.read(path)
	audios.append(list(data))
	lengths.append(len(data))


maxlen = max(map(len,audios))
audios = np.array([x+[0]*(maxlen-len(x)) for x in audios])
target_phrase = args.target_phrase


print(maxlen , target_phrase , lengths)



# audio_conf = dict(sample_rate=args.sample_rate,
#                   window_size=args.window_size,
#                   window_stride=args.window_stride,
#                   window=args.window,
#                   noise_dir=args.noise_dir,
#                   noise_prob=args.noise_prob,
#                   noise_levels=(args.noise_min, args.noise_max))

# rnn_type = args.rnn_type.lower()
# assert rnn_type in supported_rnns, "rnn_type should be either lstm, rnn or gru"
# model = DeepSpeech(rnn_hidden_size=args.hidden_size,
#                    nb_layers=args.hidden_layers,
#                    labels=labels,
#                    rnn_type=supported_rnns[rnn_type],
#                    audio_conf=audio_conf,
#                    bidirectional=args.bidirectional)




def DB(x):
	return 20 * torch.max(x).log().clamp(min=-5.0) / torch.log(torch.tensor(10.0))


    



# 
# audios = [data1,data1]


# maxlen = len(data1)

# print(args)


# python main.py -x=[/scratch/pp1953/short-audio/1.wav,/scratch/pp1953/short-audio/2.wav,/scratch/pp1953/short-audio/3.wav] -t "test"
