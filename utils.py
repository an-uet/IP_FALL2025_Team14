import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms

from PIL import Image

from MODNet.src.models.modnet import MODNet

import AdaIN.net as net
from AdaIN.function import adaptive_instance_normalization, coral

ref_size = 512

im_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]
    )

def test_transform(size, crop):
    transform_list = []
    if size != 0:
        transform_list.append(transforms.Resize(size))
    if crop:
        transform_list.append(transforms.CenterCrop(size))
    transform_list.append(transforms.ToTensor())
    transform = transforms.Compose(transform_list)
    return transform


def style_transfer(vgg, decoder, content, style, alpha=1.0,
                   interpolation_weights=None):
    assert (0.0 <= alpha <= 1.0)
    content_f = vgg(content)
    style_f = vgg(style)
    if interpolation_weights:
        _, C, H, W = content_f.size()
        feat = torch.FloatTensor(1, C, H, W).zero_().to(device)
        base_feat = adaptive_instance_normalization(content_f, style_f)
        for i, w in enumerate(interpolation_weights):
            feat = feat + w * base_feat[i:i + 1]
        content_f = content_f[0:1]
    else:
        feat = adaptive_instance_normalization(content_f, style_f)
    feat = feat * alpha + content_f * (1 - alpha)
    return decoder(feat)

def transfer_bg(modnet, device, content_img, style_img):
    # unify image channels to 3
    im = np.asarray(content_img)

    if len(im.shape) == 2:
        im = im[:, :, None]
    if im.shape[2] == 1:
        im = np.repeat(im, 3, axis=2)
    elif im.shape[2] == 4:
        im = im[:, :, 0:3]

    # convert image to PyTorch tensor
    im = Image.fromarray(im)
    im = im_transform(im)

    # add mini-batch dim
    im = im[None, :, :, :]

    im_b, im_c, im_h, im_w = im.shape

    im_rh = ((im_h + 31) // 32) * 32
    im_rw = ((im_w + 31) // 32) * 32

    pad_h = im_rh - im_h
    pad_w = im_rw - im_w
    if pad_h != 0 or pad_w != 0:
        # pad: (left, right, top, bottom) -> we pad on the right and bottom
        im = F.pad(im, (0, pad_w, 0, pad_h), mode='reflect')

    # inference
    _, _, matte = modnet(im.to(device) if torch.cuda.is_available() else im, True)

    # resize and save matte
    matte = F.interpolate(matte, size=(im_h, im_w), mode='area')
    matte = matte[0][0].data.cpu().numpy()
    print("Predicted matte!")
    
    # perform style transfer on background
    decoder = net.decoder
    vgg = net.vgg

    decoder.eval()
    vgg.eval()

    decoder.load_state_dict(torch.load("./AdaIN/models/decoder.pth"))
    vgg.load_state_dict(torch.load("./AdaIN/models/vgg_normalised.pth"))
    vgg = nn.Sequential(*list(vgg.children())[:31])

    vgg.to(device)
    decoder.to(device)

    # keep original image sizes for content/style transforms (don't force 512)
    content_tf = test_transform(size=0, crop=False)
    style_tf = test_transform(size=0, crop=False)

    content = content_tf(content_img)
    style = style_tf(style_img)
    style = style.to(device).unsqueeze(0)
    content = content.to(device).unsqueeze(0)
    with torch.no_grad():
        output = style_transfer(vgg, decoder, content, style, alpha=1.0)
    print("Transfered style!")

    output = F.interpolate(output, size=(im_h, im_w), mode='area')
    output = output.permute(0, 2, 3, 1)
    output = output[0].data.cpu().numpy()
    output = np.clip(output, 0, 1)  
    output = (output * 255).astype(np.uint8)
    final_image = np.array(content_img) * np.expand_dims(matte, axis=2) / 255.0 + output * (1 - np.expand_dims(matte, axis=2)) / 255.0

    final_image = np.clip(final_image * 255.0, 0, 255).astype(np.uint8)
    print("Done")
    return final_image

# if __name__ == "__main__":
#     # create MODNet and load the pre-trained ckpt
#     ckpt_path = "MODNet/models/P3M-10k-1-20-1/best_test_finetuned_model.pth"

#     device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
#     # device = torch.device("cpu")
#     modnet = MODNet(backbone_pretrained=False)
#     state_dict = torch.load(ckpt_path, map_location='cpu')

#     if list(state_dict.keys())[0].startswith("module."):
#         state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}

#     modnet.load_state_dict(state_dict)

#     modnet = modnet.to(device)
#     modnet.eval()

#     transfer_bg(
#         modnet,
#         device,
#         content_img=Image.open("/mnt/HDD1/anlt/image_processing/medical/AdaIN/input/content/blonde_girl.jpg"),
#         style_img=Image.open("/mnt/HDD1/anlt/image_processing/medical/AdaIN/input/style/antimonocromatismo.jpg")
#     )