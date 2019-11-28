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
parser.add_argument('-out', '--out-name', type=str, default="0")
parser.add_argument('--num-iterations', type=int, default=10000)


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
            # self.noise = nn.Parameter(torch.randn(batch_size, maxlen).cuda())
            self.noise = nn.Parameter(torch.zeros(batch_size, maxlen).cuda())
        else:
            # self.noise = nn.Parameter(torch.randn(batch_size, maxlen))
            self.noise = nn.Parameter(torch.zeros(batch_size, maxlen))
        self.rescale = 1
        self.batch_size = batch_size

    def forward(self, x, debug= False, no_noise= False):
        if debug == True:
        	return x
        apply_delta = torch.clamp(self.noise, min=-2000, max=2000)*self.rescale
        new_input = apply_delta + original
        
        noise = torch.zeros(self.batch_size, maxlen).to(device)
        noise.data.normal_(0, std=2)
        noise_energy = (noise.view(-1) * noise.view(-1)).sum() 
        # noise_levels=(0, 0.5)
        data_energy = (new_input.view(-1) * new_input.view(-1)).sum() 
        # noise_level = np.random.uniform(noise_levels)[1]
        noise_level = 1
        noise = noise_level * noise * data_energy / noise_energy

        pass_in = new_input +  noise 
        pass_in = torch.clamp(pass_in , min=-2**15, max=2**15-1)
        return pass_in 



def DB(x):
	return 20 * torch.max(x).log().clamp(min=-5.0) / torch.log(torch.tensor(10.0))


criterion = nn.CTCLoss().to(device)
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
	temp.add_(-mean)
	temp.div_(std)
	input_sizes = torch.IntTensor([temp.size(3)]).int()
	out, output_sizes = model(temp, input_sizes)
	decoded_output, decoded_offsets = decoder.decode(out, output_sizes)
	print(decoded_output)
	int_transcript = list(filter(None, [labels_map.get(x) for x in list(target_phrase)]))
	optimizer_noise = torch.optim.Adam(noise_model.parameters(), lr=0.01, weight_decay=0.0005)
	# optimizer_noise = torch.optim.SGD(noise_model.parameters(), lr=0.001, momentum=0.001, weight_decay=0.01, nesterov=True)
	if cuda:
		noise_model, optimizer_noise = amp.initialize(noise_model, optimizer_noise,
			opt_level='O1',keep_batchnorm_fp32=None,loss_scale=1.0)
	tau = 1000
	for k in range(100) :
		count  = 0
		prev = 1000
		tau = tau / 10 
		for j in range(args.num_iterations):
			wave = noise_model(original)
			wave = wave.to(device)
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
			out = out.transpose(0, 1)  # TxNxH
			float_out = out.float()  # ensure float32 for loss
			targets = torch.tensor(int_transcript , dtype= torch.int32 )
			target_sizes = torch.tensor([len(target_phrase)] , dtype= torch.int32)

			l2_norm = torch.norm(wave - original, p=2).pow(2)
			db_x  = DB(wave - original) - DB(original)
			db_x = torch.abs(db_x)
			ctc_loss = criterion(softmax(float_out).cpu(), targets, output_sizes, target_sizes).to(device)
			if db_x.item() < 10**(tau/20) :  
				loss =  10*ctc_loss
			else:
				loss = l2_norm +  100 * (db_x - 10**(tau/20)) +   10*ctc_loss
			model.zero_grad()
			loss_value = loss.item()
			optimizer_noise.zero_grad()
			if use_gpu:
				with amp.scale_loss(loss, optimizer_noise) as scaled_loss:
				    scaled_loss.backward()
			else:
				loss.backward()
			torch.nn.utils.clip_grad_value_(noise_model.parameters(), 400)
			optimizer_noise.step()
			# if noise_model.rescale * 2000 >  torch.max(noise_model.noise).item():
			# 	noise_model.rescale = torch.max(noise_model.noise).item() / 2000 
			if j % 100 == 0 : 
				# noise_model.rescale *= 0.8
				wave = noise_model(original, no_noise=False)
				wave = wave.to(device)
				wave = nn.functional.normalize(wave, p=2, dim=1)
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
				decoded_output, decoded_offsets = decoder.decode(out, output_sizes)
				temp_audio = wave.view(-1).cpu().detach().numpy()
				wavfile.write('temp%s.wav'%(args.out_name),16000,temp_audio)
				print("count= {} k={} Epoch {} Loss: {:.6f} CTC Loss {:.6f} , tau {:.10f}".format(count, k, j,  loss_value, ctc_loss.item() ,tau))
				print("decoded_output: %s"%(decoded_output[0][0]))
				for g in optimizer_noise.param_groups:
				    g['lr'] = g['lr'] / 1.05
				if loss_value < prev and loss_value > prev -0.5:
					count += 1
					prev = loss_value
					if count >=2:
						break	 
				else:
					prev = min(loss_value, prev)
					torch.save(noise_model.state_dict(), "temp_noise%s.pth"%(args.out_name))



# print('Learning rate annealed to: {lr:.6f}'.format(lr=g['lr']))



# python main.py -x=[/scratch/pp1953/short-audio/1.wav,/scratch/pp1953/short-audio/2.wav,/scratch/pp1953/short-audio/3.wav] -t "test"
# python main.py -t "test is a test" --out-name=0


########################## 
# 1 without normalization training 
# 0 with normalization training 

# i= 0
# audio = audios[i]
# original = torch.FloatTensor(audio).to(device)
# original.requires_grad= False
# model.requires_grad = True	
# original = original.unsqueeze(0)
# maxlen =lengths[i]
# noise_model = Noise_model(1,maxlen, cuda)
# # wave = noise_model(original, False, True).to(device)
# apply_delta = torch.clamp(noise_model.noise, min=-2000, max=2000)*noise_model.rescale
# mean , std = apply_delta.mean() , apply_delta.std()
# apply_delta = nn.functional.normalize(apply_delta, p=2, dim=1) # normalize the features
# apply_delta = (apply_delta - mean) / (std + 1e-6)
# new_input =  apply_delta + original
# pass_in = torch.clamp(new_input, min=-2**15, max=2**15-1)
# wave = 	pass_in	
# wave = wave.unsqueeze(0)
# temp_audio = wave.view(-1).cpu().detach().numpy()
# wavfile.write('temp1.wav',16000,temp_audio)


# magnitude, phase = stft.transform(wave)
# magnitude = magnitude.unsqueeze(0)
# magnitude = magnitude.to(device)
# magnitude = magnitude.clamp(min=1e-12, max=1.0)
# temp = torch.log(magnitude + 1).clamp(min=1e-12, max=1.0)
# mean = temp.mean()
# std = temp.std()
# temp = (temp - mean) / (std + 1e-6)
# input_sizes = torch.IntTensor([temp.size(3)]).int()
# out, output_sizes = model(temp, input_sizes)
# decoded_output, decoded_offsets = decoder.decode(out, output_sizes)

# print(decoded_output)



