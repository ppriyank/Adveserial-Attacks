import argparse
import numpy as np 
import os 
import json 

from scipy.io import wavfile


import torch
import torch.nn as nn
from torch.autograd import Variable
# from warpctc_pytorch import CTCLoss
import torchaudio
from torch_stft import STFT



from model import DeepSpeech, supported_rnns
from decoder import GreedyDecoder
from utils import reduce_tensor, check_loss


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

transcript_path = "/scratch/pp1953/short-audio/2.txt"
labels_map = dict([(labels[i], i) for i in range(len(labels))])

labels_path = "labels.json"
with open(labels_path) as label_file:
            labels = str(''.join(json.load(label_file)))


with open(transcript_path, 'r', encoding='utf8') as transcript_file:
	transcript = transcript_file.read().replace('\n', '')


transcript = list(filter(None, [labels_map.get(x) for x in list(transcript)]))
out = out.transpose(0, 1)  # TxNxH
float_out = out.float()  # ensure float32 for loss

targets=[]
targets.extend(transcript)
targets = torch.IntTensor(targets)
target_sizes = torch.IntTensor(batch_size)

target_sizes[0]= len(transcript)


loss = criterion(float_out, targets, output_sizes, target_sizes).to(device)
loss = loss / magnitude.size(0)  # average the loss by minibatch
loss = loss.to(device)
loss_value = reduce_tensor(loss, args.world_size).item()
            
loss_value = loss.item()
valid_loss, error = check_loss(loss, loss_value)
            

optimizer.zero_grad()
# compute gradient

with amp.scale_loss(loss, optimizer) as scaled_loss:
    scaled_loss.backward()
torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_norm)
optimizer.step()


avg_loss += loss_value
losses.update(loss_value, inputs.size(0))

print('Epoch: [{0}][{1}/{2}]\t'
      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
      'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
        (epoch + 1), (i + 1), len(train_sampler), batch_time=batch_time, data_time=data_time, loss=losses))
            

del loss, out, float_out
avg_loss /= len(train_sampler)

with torch.no_grad():
    wer, cer, output_data = evaluate(test_loader=test_loader,
                                     device=device,
                                     model=model,
                                     decoder=decoder,
                                     target_decoder=decoder)
loss_results[epoch] = avg_loss
wer_results[epoch] = wer
cer_results[epoch] = cer
print('Validation Summary Epoch: [{0}]\t'
      'Average WER {wer:.3f}\t'
      'Average CER {cer:.3f}\t'.format(
    epoch + 1, wer=wer, cer=cer))

    values = {
        'loss_results': loss_results,
        'cer_results': cer_results,
        'wer_results': wer_results
    }
        # anneal lr
for g in optimizer.param_groups:
    g['lr'] = g['lr'] / args.learning_anneal
print('Learning rate annealed to: {lr:.6f}'.format(lr=g['lr']))

best_wer = wer
avg_loss = 0



# python main.py -x=[/scratch/pp1953/short-audio/1.wav,/scratch/pp1953/short-audio/2.wav,/scratch/pp1953/short-audio/3.wav] -t "test"
