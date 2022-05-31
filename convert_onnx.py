import sys
import torch
from models import Darknet
from models import *

def transform_to_onnx(batch_size=1):
    model = Darknet("runs/v5.cfg")
    # _=load_darknet_weights(model, "runs/v5.weights")
    model.load_state_dict(torch.load("runs/new.pt")['model'],strict=False)

    dynamic = False
    if batch_size <= 0:
        dynamic = True

    input_names = ["input"]

    if dynamic:
        x = torch.randn((1, 3, 320, 320), requires_grad=True)
        onnx_file_name = "yolov4_-1_3_{}_{}_dynamic.onnx".format(model.height, model.width)
        dynamic_axes = {"input": {0: "batch_size"}, "boxes": {0: "batch_size"}, "confs": {0: "batch_size"}}
        # Export the model
        print('Export the onnx model ...')
        torch.onnx.export(model,
                          x,
                          onnx_file_name,
                          export_params=True,
                          opset_version=11,
                          do_constant_folding=True,
                          input_names=input_names,
                          dynamic_axes=dynamic_axes)

        print('Onnx model exporting done')
        return onnx_file_name

    else:
        x = torch.randn((batch_size, 3, 320, 320), requires_grad=True)
        onnx_file_name = "yolov4_{}_3_{}_{}_static.onnx"
        torch.onnx.export(model,
                          x,
                          onnx_file_name,
                          export_params=True,
                          opset_version=11,
                          do_constant_folding=True,
                          input_names=input_names,
                          dynamic_axes=None)

        print('Onnx model exporting done')
        return onnx_file_name
transform_to_onnx()