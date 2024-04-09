# import safetensors.torch
# import torch
# import safetensors

# def save_model(src, path):
#     model = torch.load(src)
#     safetensors.torch.save_file(model, path)

import os
import re
import torch
from safetensors.torch import save_file

def changelora(src, dst):
    # dir = "faceoutput"
    newDict = dict()
    # 新版diffusers需要改为pytorch_model.bin
    checkpoint = torch.load(src)
    for idx, key in enumerate(checkpoint):

        newKey = re.sub('\.processor\.', '_', key)
        newKey = re.sub('mid_block\.', 'mid_block_', newKey)
        newKey = re.sub('_lora.up.', '.lora_up.', newKey)
        newKey = re.sub('_lora.down.', '.lora_down.', newKey)
        newKey = re.sub('\.(\d+)\.', '_\\1_', newKey)
        newKey = re.sub('to_out', 'to_out_0', newKey)
        newKey = 'lora_unet_'+newKey

        newDict[newKey] = checkpoint[key]

    # newLoraName = src.replace('.bin', '.safetensors')
    print("Saving " + dst)
    save_file(newDict, dst)


if __name__ == '__main__':
    changelora("faceoutput/zqd.bin", "faceoutput/zqd.safetensors")