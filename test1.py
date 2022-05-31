import torch

from models import *
model=Darknet("runs/V1130.cfg")
# load_darknet_weights(model,"runs/v5.weights")

#save_weights(model,path='weights/latest.weights',cutoff=-1)
checkpoint = torch.load("runs/best.pt")
model.load_state_dict(checkpoint['model'],strict=False)
# load_darknet_weights(model, "weights/pruneV50.weights", FPGA=False)
save_weights(model,path='./V1133_best.weights')
# torch.save()
