
import os 
import torch
import torch.nn as nn
from model import DeepSpeech

path = "saved_model/librispeech_pretrained_v2.pth"
package = torch.load(path, map_location=lambda storage, loc: storage)
model = DeepSpeech.load_model_package(package).cuda()


criterion = nn.CTCLoss()
softmax = torch.nn.LogSoftmax(2)

temp = torch.load("temp.pth")
input_sizes = torch.tensor([989], dtype=torch.int32)
out, output_sizes = model(temp, input_sizes)



out = out.transpose(0, 1)  # TxNxH
float_out = out.float()  # ensure float32 for loss
targets = torch.tensor([21,  6, 20, 21, 10, 15,  8], dtype=torch.int32)
target_sizes = torch.tensor([7], dtype=torch.int32)
ctc_loss = criterion(softmax(float_out), targets, output_sizes, target_sizes)
model.zero_grad()
ctc_loss.backward()

for param in model.parameters():
	param.grad
	braek 


