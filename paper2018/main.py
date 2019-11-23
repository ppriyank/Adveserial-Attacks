import argparse
import numpy as np 
import os 
from scipy.io import wavfile

import torch
import torch.nn as nn
from torch.autograd import Variable
# from warpctc_pytorch import CTCLoss
import torchaudio
from torch_stft import STFT



from model import DeepSpeech, supported_rnns
from decoder import GreedyDecoder


parser = argparse.ArgumentParser(description='idk :P')
parser.add_argument('-t', '--target-phrase', type=str, default='testing')
parser.add_argument('-x', '--input-audio-paths', type=list, default=[c for c in '[/scratch/pp1953/short-audio/2.wav]'])
parser.add_argument('-mp', '--model-path', type=str, default="saved_model/librispeech_pretrained_v2.pth")
parser.add_argument('-gpu', '--gpu-devices', type=int, default=0)
parser.add_argument('--num-iterations', type=int, default=5000)

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




print("Loading checkpoint model %s" % args.model_path)
package = torch.load(args.model_path, map_location=lambda storage, loc: storage)
model = DeepSpeech.load_model_package(package)
labels = model.labels
audio_conf = model.audio_conf

def DB(x):
	return 20 * torch.max(x).log().clamp(min=-5.0) / torch.log(torch.tensor(10.0))



batch_size = len(audios)
use_gpu = torch.cuda.is_available()
cuda = 0 
noise = nn.Parameter(torch.rand(batch_size, maxlen))
if use_gpu:
	os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_devices
	noise.cuda()
	cuda=1


device = torch.device("cuda" if cuda else "cpu")
torch.manual_seed(1)
rescale = 1
model.to(device)


mask = np.array([[1 if i < l else 0 for i in range(maxlen)] for l in lengths])
mask = torch.tensor(mask).float()


final_deltas = [None] * batch_size
original = torch.tensor(np.array(audios)).float()
criterion = nn.CTCLoss()



# new_input =  self.apply_delta*mask + original

# for i in range(args.num_iterations):
apply_delta = torch.clamp(noise, min=-2000, max=2000)*rescale
new_input =  apply_delta*mask + original


noise = torch.zeros(batch_size, maxlen)
noise.data.normal_(0, std=2)
pass_in = torch.clamp(new_input+noise, min=-2**15, max=2**15-1)




# spect_parser = SpectrogramParser(model.audio_conf, normalize=True)


# out, output_sizes = model(pass_in, (batch_size,maxlen))

window_size = audio_conf['window_size']
sample_rate = audio_conf['sample_rate']
window_stride = audio_conf['window_stride']
n_fft = int(sample_rate * window_size)
win_length = n_fft
hop_length = int(sample_rate * window_stride)
        
stft = STFT(filter_length=n_fft,  hop_length=hop_length, win_length=n_fft,window='hann').to(device)

audio = torch.FloatTensor(data)
audio = audio.unsqueeze(0)
magnitude, phase = stft.transform(audio)
magnitude = magnitude.unsqueeze(0)
magnitude = magnitude.to(device)


input_sizes = torch.IntTensor([magnitude.size(3)]).int()
out, output_sizes = model(magnitude, input_sizes)

decoder = GreedyDecoder(model.labels, blank_index=model.labels.index('_'))
decoded_output, decoded_offsets = decoder.decode(out, output_sizes)




# 
# audios = [data1,data1]


# maxlen = len(data1)

# print(args)


# python main.py -x=[/scratch/pp1953/short-audio/1.wav,/scratch/pp1953/short-audio/2.wav,/scratch/pp1953/short-audio/3.wav] -t "test"
