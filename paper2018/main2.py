import argparse
import numpy as np 
import os 
import json 
from apex import amp
from scipy.io import wavfile
import sys

import torch
import torch.nn as nn
from torch.autograd import Variable
# from warpctc_pytorch import CTCLoss
import torchaudio
from torch_stft import STFT

from model import DeepSpeech, supported_rnns
from decoder import GreedyDecoder
import wave as chikka
from data.data_loader import load_audio

parser = argparse.ArgumentParser(description='idk :P')
parser.add_argument('-t', '--target-phrase', type=str, default='testing')
parser.add_argument('-x', '--input-audio-paths', type=list, default=[c for c in '[/scratch/pp1953/short-audio/2.wav]'])
parser.add_argument('-mp', '--model-path', type=str, default="saved_model/librispeech_pretrained_v2.pth")
parser.add_argument('-gpu', '--gpu-devices', type=str, default="0")
parser.add_argument('-out', '--out-name', type=str, default="0")
parser.add_argument('--num-iterations', type=int, default=2000)
parser.add_argument('--ctc-weight', type=float, default=1)
parser.add_argument('--l2-weight', type=float, default=0.005)
parser.add_argument('--db-weight', type=float, default=0.05)

args = parser.parse_args()


import pdb 
thresold = 5
thresold_2 = 3
# ctc_loss_weight = 0.0005
ctc_loss_weight = args.ctc_weight
l2_weight = args.l2_weight
db_weight = args.db_weight
scaling_factor = 10

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
            # self.noise = nn.Parameter(torch.randn(batch_size, maxlen).cuda())
            self.noise = nn.Parameter(torch.zeros(batch_size, maxlen).cuda())
        else:
            # self.noise = nn.Parameter(torch.randn(batch_size, maxlen))
            self.noise = nn.Parameter(torch.zeros(batch_size, maxlen))
        self.rescale = 1
        self.batch_size = batch_size
        self.factor  = 1
    def forward(self, x, debug= False, no_noise= False, check=False):
        if debug == True:
        	return x
        apply_delta = torch.clamp(self.noise, min=-2000, max=2000)*self.rescale
        if check:
        	new_input = apply_delta / self.factor  + original * scaling_factor
        	return new_input
        
        new_input = apply_delta / self.factor  + original * scaling_factor
        if no_noise:
        	return new_input

        noise = torch.zeros(self.batch_size, maxlen).to(device)
        noise.data.normal_(0, std=2)
        noise_energy = (noise.view(-1) * noise.view(-1)).sum() 
        # noise_levels=(0, 0.5)
        data_energy = (scaling_factor**2) * (original.view(-1) * original.view(-1)).sum() 
        # noise_level = np.random.uniform(noise_levels)[1]
        noise_level = self.rescale
        noise = (noise_level **2) * noise * data_energy / noise_energy

        pass_in = new_input +  noise 
        pass_in = torch.clamp(pass_in , min=-2**15, max=2**15-1)
        return pass_in 



def DB(x):
	return 20 * torch.max(torch.abs(x) + 1e-6).log() / torch.log(torch.tensor(10.0))



def output(wave ):
	magnitude, phase = stft.transform(wave)
	magnitude = magnitude.unsqueeze(0)
	magnitude = magnitude.to(device)
	# magnitude = magnitude.clamp(min=1e-12, max=1.0)
	temp = torch.log(magnitude + 1).clamp(min=1e-12, max=1.0)
	mean = temp.mean()
	std = temp.std()
	temp = (temp - mean) / (std + 1e-6)
	input_sizes = torch.IntTensor([temp.size(3)]).int()
	out, output_sizes = model(temp, input_sizes)
	decoded_output, decoded_offsets = decoder.decode(out, output_sizes)
	return decoded_output, decoded_offsets 

def numpy_to_req_wav(audio_file_name, audio_np):
	audio_bytes = audio_np.tobytes()
	nchannels = 1
	sampwidth = 2
	framerate = 16000
	#audio_file_name is the path with filename of the .wav file
	with chikka.open(audio_file_name, "wb") as wave_file:
		wave_file.setnchannels(nchannels)
		wave_file.setsampwidth(sampwidth)
		wave_file.setframerate(framerate)
		wave_file.writeframes(audio_bytes)


criterion = nn.CTCLoss(zero_infinity=True).to(device)
softmax = torch.nn.LogSoftmax(2)
decoder = GreedyDecoder(model.labels, blank_index=model.labels.index('_'))
model.requires_grad = True	

for i in range(len(audios)):
	print("processing %d-th audio (%s)" %(i+1, args.input_audio_paths[i]))
	maxlen =lengths[i]
	noise_model = Noise_model(1,maxlen, cuda)
	transcript_path = args.input_audio_paths[i][:-3] + "txt"
	with open(transcript_path, 'r', encoding='utf8') as transcript_file:
		transcript = transcript_file.read().replace('\n', '')
	print(transcript)
	audio = audios[i]
	original = torch.FloatTensor(audio).to(device)
	original.requires_grad= False
	original = original.unsqueeze(0)
	original = nn.functional.normalize(original, p=2, dim=1)
	magnitude, phase = stft.transform(original)
	magnitude = magnitude.unsqueeze(0)
	magnitude = magnitude.to(device)

	temp = torch.log(magnitude + 1)
	mean = temp.mean()
	std = temp.std()
	temp = (temp - mean) / (std + 1e-6)
	input_sizes = torch.IntTensor([temp.size(3)]).int()
	out, output_sizes = model(temp, input_sizes)
	decoded_output, decoded_offsets = decoder.decode(out, output_sizes)
	print(decoded_output)
	int_transcript = list(filter(None, [labels_map.get(x) for x in list(target_phrase)]))
	optimizer_noise = torch.optim.Adam(noise_model.parameters(), lr=0.01, weight_decay=0)
	# optimizer_noise = torch.optim.SGD(noise_model.parameters(), lr=0.01, momentum=0, nesterov=False)
	if cuda:
		noise_model, optimizer_noise = amp.initialize(noise_model, optimizer_noise,
			opt_level='O1',keep_batchnorm_fp32=None,loss_scale=1.0)
	tau = 10
	targets = torch.tensor(int_transcript , dtype= torch.int32 )
	target_sizes = torch.tensor([len(target_phrase)] , dtype= torch.int32)
	db_original = DB(original * scaling_factor )
	# l2_norm = torch.norm((original), p=2).pow(2)

	for k in range(20) :
		count  = 0
		prev = 100

		wave2 = noise_model(original, check=True)
		wave2 = wave2.to(device)
		wave2 = wave2.unsqueeze(0)
		decoded_output2, decoded_offsets2 = output(wave2)
		temp_audio2 = wave2.view(-1).cpu().detach().numpy()
		print("============= Epoch Change")
		print("decoded_output2: %s"%(decoded_output2[0][0]))								
		wavfile.write('yey_1%s.wav'%(args.out_name),16000,temp_audio2 )
		scaling_factor = max(scaling_factor - 1 , 1)
		# tau = tau - 5 
		# noise_model.factor +=10
		for j in range(args.num_iterations):
			wave = noise_model(original)
			wave = wave.to(device)
			wave = wave.unsqueeze(0)
			magnitude, phase = stft.transform(wave)
			magnitude = magnitude.unsqueeze(0)
			magnitude = magnitude.to(device)

			temp = torch.log(magnitude + 1).clamp(min=1e-12, max=1.0)
			mean = temp.mean()
			std = temp.std()
			temp = (temp - mean + 1e-6 ) / (std + 1e-6)
			input_sizes = torch.IntTensor([temp.size(3)]).int()
			out, output_sizes = model(temp, input_sizes)
			decoded_output, decoded_offsets = decoder.decode(out, output_sizes)
			
			out = out.transpose(0, 1)  # TxNxH
			float_out = out.float()  # ensure float32 for loss
			
			l2_norm = torch.norm((wave - scaling_factor * original), p=2).pow(2)
			db_x  = DB(noise_model.noise) - db_original
			ctc_loss = criterion(softmax(float_out).cpu(), targets, output_sizes, target_sizes).to(device).clamp(min=-400.0 , max=400.0)

			# if l2_norm.item() < 0.6:
			# 	l2_norm = 1000 / l2_norm
			# 	loss = l2_norm 
				# + 0.5 * ctc_loss
			# elif decoded_output[0][0] == target_phrase or ctc_loss.item() < thresold_2:				
			# 	loss =  db_weight * (db_x - tau)+ l2_norm.clamp(min=1) * l2_weight + 0.0005 * ctc_loss
			if db_x.item() < tau :  
				loss = 10 * ctc_loss_weight * ctc_loss + db_weight* (db_x  - tau) + l2_norm * l2_weight
			else:
				loss = 10 * db_weight * (db_x  - tau)  + ctc_loss_weight * ctc_loss + l2_norm * l2_weight
			model.zero_grad()
			loss_value = loss.item()
			optimizer_noise.zero_grad()
			if use_gpu:
				with amp.scale_loss(loss, optimizer_noise) as scaled_loss:
				    scaled_loss.backward()
			else:
				loss.backward()
			if torch.isnan(noise_model.noise.grad).any():
				noise_model.noise.grad[torch.isnan(noise_model.noise.grad)] = 0
				
			optimizer_noise.step()
			if j % 10 == 0 :
				wave = noise_model(original, no_noise=True)
				wave = wave.to(device)
				wave = wave.unsqueeze(0)
				decoded_output, decoded_offsets = output(wave)
				temp_audio = wave.view(-1).cpu().detach().numpy()
				
				print("count= {} k={} Epoch {} L2 norm {} DB Loss: {:.6f} CTC Loss {:.6f} , tau {} Noise_factor : 1/{}".format(count, k, j, l2_norm.item(),  db_x.item(), ctc_loss.item() ,tau, noise_model.factor))
				print("decoded_output: %s"%(decoded_output[0][0]))

				
				if decoded_output[0][0] == target_phrase or ctc_loss.item() < thresold_2 :
					wavfile.write('yey_0%s.wav'%(args.out_name),16000,temp_audio2)
					
					wave2 = noise_model(original)
					wave2 = wave2.to(device)
					wave2 = wave2.unsqueeze(0)
					decoded_output2, decoded_offsets2 = output(wave2)
					temp_audio2 = wave2.view(-1).cpu().detach().numpy()
					print("decoded_output: %s"%(decoded_output2[0][0]))
					wavfile.write('yey_2%s.wav'%(args.out_name),16000,temp_audio2 )
					

					torch.save(noise_model.state_dict(), "yey_noise%s.pth"%(args.out_name))
					if (ctc_loss.item() < thresold_2 or decoded_output[0][0] == target_phrase) and db_x.item() < tau :
						noise_model.factor += 5
						tau = tau - 5

					if decoded_output[0][0] == target_phrase:
						count +=1 
						if count >= thresold:
							break	 
				if loss_value < prev and loss_value > prev -0.10:
					for g in optimizer_noise.param_groups:
					    g['lr'] = g['lr'] / 1.05
					count += 1
					prev = loss_value
					if count > thresold :
						break	 
				else:
					prev = min(loss_value, prev)
				# 	torch.save(noise_model.state_dict(), "temp_noise%s.pth"%(args.out_name))


# print('Learning rate annealed to: {lr:.6f}'.format(lr=g['lr']))

# python main2.py -t "test is a test" --out-name=0  --db-weight=100 --l2-weight=0 

# python main.py -x=[/scratch/pp1953/short-audio/1.wav,/scratch/pp1953/short-audio/2.wav,/scratch/pp1953/short-audio/3.wav] -t "test"


# python main2.py -t "test is a test" --out-name=0  --ctc-weight=10 --db-weight=0.5 --l2-weight=0.0005
# python main2.py -t "test is a test" --out-name=0  --ctc-weight=1 --db-weight=1 --l2-weight=0.0005