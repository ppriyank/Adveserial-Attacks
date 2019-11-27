import argparse
import numpy as np 
import os 
import json 
from apex import amp
from scipy.io import wavfile


import torch
import torch.nn as nn
from torch.autograd import Variable
# from warpctc_pytorch import CTCLoss
import torchaudio
from torch_stft import STFT

from model import DeepSpeech, supported_rnns
from decoder import GreedyDecoder

from data.data_loader import load_audio

parser = argparse.ArgumentParser(description='idk :P')
parser.add_argument('-t', '--target-phrase', type=str, default='testing')
parser.add_argument('-x', '--input-audio-paths', type=list, default=[c for c in '[/scratch/pp1953/short-audio/2.wav]'])
parser.add_argument('-mp', '--model-path', type=str, default="saved_model/librispeech_pretrained_v2.pth")
parser.add_argument('-gpu', '--gpu-devices', type=str, default="0")
parser.add_argument('--num-iterations', type=int, default=5000)


args = parser.parse_args()



# print(args)
args.input_audio_paths = ("".join(args.input_audio_paths[1:-1])).split(",")
# print(args.input_audio_paths)

# Read audios 
audios = []
lengths = []
for path in args.input_audio_paths: 
	data = load_audio(path)
	audios.append(data)
	lengths.append(len(data))


target_phrase = args.target_phrase.upper()
print("Loading checkpoint model %s" % args.model_path)
package = torch.load(args.model_path, map_location=lambda storage, loc: storage)
model = DeepSpeech.load_model_package(package)
labels = model.labels
audio_conf = model.audio_conf
# model.eval()

use_gpu = torch.cuda.is_available()
cuda = 0 
if use_gpu:
	print("USING GPUS :D :D :D ")
	os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_devices
	cuda=1


device = torch.device("cuda" if cuda else "cpu")
torch.manual_seed(1)
batch_size = 1
rescale = 1

window_size = audio_conf['window_size']
sample_rate = audio_conf['sample_rate']
window_stride = audio_conf['window_stride']
n_fft = int(sample_rate * window_size)
win_length = n_fft
hop_length = int(sample_rate * window_stride)
stft = STFT(filter_length=n_fft,  hop_length=hop_length, win_length=win_length,window='hamming').to(device)
final_deltas = [None] * len(audios)
model.to(device)



labels_map = dict([(labels[i], i) for i in range(len(labels))])

class Noise_model(nn.Module):
    def __init__(self, batch_size=1, maxlen=None, use_gpu=True):
        super(Noise_model, self).__init__()
        self.use_gpu = use_gpu
        if self.use_gpu:
            self.noise = nn.Parameter(torch.randn(batch_size, maxlen).cuda())
        else:
            self.noise = nn.Parameter(torch.randn(batch_size, maxlen))
        self.rescale = 1
    def forward(self, x, debug= False):
        if debug == True:
        	return x
        apply_delta = torch.clamp(self.noise, min=-2000, max=2000)*self.rescale
        new_input =  apply_delta + original
        noise = torch.zeros(batch_size, maxlen).to(device)
        noise.data.normal_(0, std=2)
        pass_in = torch.clamp(new_input+noise, min=-2**15, max=2**15-1)
        return pass_in



def DB(x):
	return 20 * torch.max(x).log().clamp(min=-5.0) / torch.log(torch.tensor(10.0))


criterion = nn.CTCLoss().to(device)
softmax = torch.nn.LogSoftmax(2)

for i in range(len(audios)):
	print("processing %d-th audio (%s)" %(i+1, args.input_audio_paths[i]))
	avg_loss = 0 
	maxlen =lengths[i]
	noise_model = Noise_model(1,maxlen, cuda)
	transcript_path = args.input_audio_paths[i][:-3] + "txt"
	with open(transcript_path, 'r', encoding='utf8') as transcript_file:
		transcript = transcript_file.read().replace('\n', '')
	print(transcript)
	# import pdb
	# pdb.set_trace()
	audio = audios[i]
	original = torch.FloatTensor(audio).to(device)
	original.requires_grad= False
	model.requires_grad = True	
	original = original.unsqueeze(0)
	magnitude, phase = stft.transform(original)
	magnitude = magnitude.unsqueeze(0)
	magnitude = magnitude.to(device)
	temp = torch.log(magnitude + 1)
	mean = temp.mean()
	std = temp.std()
	temp.add_(-mean)
	temp.div_(std)
	input_sizes = torch.IntTensor([temp.size(3)]).int()
	out, output_sizes = model(temp, input_sizes)
	decoder = GreedyDecoder(model.labels, blank_index=model.labels.index('_'))
	decoded_output, decoded_offsets = decoder.decode(out, output_sizes)
	print(decoded_output)
	int_transcript = list(filter(None, [labels_map.get(x) for x in list(target_phrase)]))
	optimizer_noise = torch.optim.SGD(noise_model.parameters(), lr=0.001, momentum=0.01, weight_decay=1, nesterov=True)
	if cuda:
		noise_model, optimizer_noise = amp.initialize(noise_model, optimizer_noise,
			opt_level='O1',keep_batchnorm_fp32=None,loss_scale=1.0)
	tau = 1000
	for k in range(10) :
		tau = tau / 10 
		for j in range(args.num_iterations):
			model.zero_grad()
			wave = noise_model(original, False).to(device)
			wave = wave.unsqueeze(0)
			magnitude, phase = stft.transform(wave)
			magnitude = magnitude.unsqueeze(0)
			magnitude = magnitude.to(device)
			magnitude = magnitude.clamp(min=1e-12, max=1.0)
			temp = torch.log(magnitude + 1).clamp(min=1e-12, max=1.0)
			mean = temp.mean()
			std = temp.std()
			temp = (temp - mean) / (std + 1e-6)
			input_sizes = torch.IntTensor([temp.size(3)]).int()
			out, output_sizes = model(temp, input_sizes)
			if j % 100 == 0 : 
				decoder = GreedyDecoder(model.labels, blank_index=model.labels.index('_'))
				decoded_output, decoded_offsets = decoder.decode(out, output_sizes)
				temp_audio = wave.view(-1).cpu().detach().numpy()
				wavfile.write('temp.wav',16000,temp_audio)
			out = out.transpose(0, 1)  # TxNxH
			float_out = out.float()  # ensure float32 for loss
			targets = torch.tensor(int_transcript , dtype= torch.int32 )
			target_sizes = torch.tensor([len(target_phrase)] , dtype= torch.int32)

			l2_norm = torch.norm(noise_model.noise, p=2)
			db_x  = DB(noise_model.noise) - DB(original)
			db_x = torch.abs(db_x)
			if db_x.item() < 10**(tau/20) :  
				loss = l2_norm +  0.5 * criterion(softmax(float_out).cpu(), targets, output_sizes, target_sizes).to(device)
			else:
				loss = l2_norm +  10 * (db_x - 10**(tau/20)) + 0.5 * criterion(softmax(float_out).cpu(), targets, output_sizes, target_sizes).to(device)

			loss_value = loss.item()
			optimizer_noise.zero_grad()
			if use_gpu:
				with amp.scale_loss(loss, optimizer_noise) as scaled_loss:
				    scaled_loss.backward()
			else:
				loss.backward()
			torch.nn.utils.clip_grad_value_(noise_model.parameters(), 400)
			optimizer_noise.step()
			print("Epoch {} Loss: {:.6f}".format( j,  loss))
			if j % 100 == 0 :
				print("Epoch {} Loss: {:.6f}".format( j,  loss))
				print("decoded_output: %s"%(decoded_output[0][0]))



# for g in optimizer_noise.param_groups:
#     g['lr'] = g['lr'] / 1.1


# print('Learning rate annealed to: {lr:.6f}'.format(lr=g['lr']))



# python main.py -x=[/scratch/pp1953/short-audio/1.wav,/scratch/pp1953/short-audio/2.wav,/scratch/pp1953/short-audio/3.wav] -t "test"
# python main.py -t "test"




