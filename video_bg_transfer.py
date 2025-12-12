import argparse
from pathlib import Path
from tqdm import tqdm

from MODNet.src.models.modnet import MODNet
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import cv2
import imageio
from torchvision import transforms
from torchvision.utils import save_image
import torch.nn.functional as F
import os

import AdaIN.net as net
from AdaIN.function import adaptive_instance_normalization, coral

import warnings
warnings.filterwarnings("ignore")

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

def preprocess_img(content_img):
    im_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]
    )
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

    return im

parser = argparse.ArgumentParser()
# Basic options
parser.add_argument('--content_video', type=str,
                    help='File path to the content video')
parser.add_argument('--style_path', type=str,
                    help='File path to the style video or single image')
parser.add_argument('--vgg', type=str, default='AdaIN/models/vgg_normalised.pth')
parser.add_argument('--decoder', type=str, default='AdaIN/models/decoder.pth')

# Additional options
parser.add_argument('--content_size', type=int, default=512,
                    help='New (minimum) size for the content image, \
                    keeping the original size if set to 0')
parser.add_argument('--style_size', type=int, default=512,
                    help='New (minimum) size for the style image, \
                    keeping the original size if set to 0')
parser.add_argument('--crop', action='store_true',
                    help='do center crop to create squared image')
parser.add_argument('--save_ext', default='.mp4',
                    help='The extension name of the output video')
parser.add_argument('--output', type=str, default='output',
                    help='Directory to save the output image(s)')

# Advanced options
parser.add_argument('--preserve_color', action='store_true',
                    help='If specified, preserve color of the content image')
parser.add_argument('--alpha', type=float, default=1.0,
                    help='The weight that controls the degree of \
                             stylization. Should be between 0 and 1')
parser.add_argument(
    '--style_interpolation_weights', type=str, default='',
    help='The weight for blending the style of multiple style images')

args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

output_dir = Path(args.output)
output_dir.mkdir(exist_ok = True, parents = True)

# --content_video should be given.
assert (args.content_video)
if args.content_video:
    content_path = Path(args.content_video)

# --style_path should be given
assert (args.style_path)
if args.style_path:
    style_path = Path(args.style_path)

decoder = net.decoder
vgg = net.vgg

decoder.eval()
vgg.eval()

decoder.load_state_dict(torch.load(args.decoder))
vgg.load_state_dict(torch.load(args.vgg))
vgg = nn.Sequential(*list(vgg.children())[:31])

vgg.to(device)
decoder.to(device)



content_tf = test_transform(args.content_size, args.crop)
style_tf = test_transform(args.style_size, args.crop)
        
#get video fps & video size
content_video = cv2.VideoCapture(args.content_video)
fps = int(content_video.get(cv2.CAP_PROP_FPS))
content_video_length = int(content_video.get(cv2.CAP_PROP_FRAME_COUNT))
# output_height = int(content_video.get(cv2.CAP_PROP_FRAME_HEIGHT))
# output_width = int(content_video.get(cv2.CAP_PROP_FRAME_WIDTH))
output_height = 512
output_width = int(content_video.get(cv2.CAP_PROP_FRAME_WIDTH))*512//int(content_video.get(cv2.CAP_PROP_FRAME_HEIGHT))


ckpt_path = "MODNet/models/P3M-10k-1-20-1/best_test_finetuned_model.pth"

device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
# device = torch.device("cpu")
modnet = MODNet(backbone_pretrained=False)
state_dict = torch.load(ckpt_path, map_location='cpu')

if list(state_dict.keys())[0].startswith("module."):
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}

modnet.load_state_dict(state_dict)

modnet = modnet.to(device)
# modnet.eval()

assert fps != 0, 'Fps is zero, Please enter proper video path'

pbar = tqdm(total = content_video_length)
if style_path.suffix in [".mp4", ".mpg", ".avi"]:

    style_video = cv2.VideoCapture(args.style_path)
    style_video_length = int(style_video.get(cv2.CAP_PROP_FRAME_COUNT))

    assert style_video_length==content_video_length, 'Content video and style video has different number of frames'

    output_video_path = output_name = output_dir / '{:s}_stylized_{:s}{:s}'.format(
                content_path.stem, style_path.stem, args.save_ext)
    writer = imageio.get_writer(output_video_path, mode='I', fps=fps)
    print('ok')
    
    while(True):
        torch.cuda.empty_cache()
        ret, content_img = content_video.read()
        if not ret:
            break

        im = preprocess_img(content_img)

        _, _, matte = modnet(im.to(device) if torch.cuda.is_available() else im, True)

        # resize and save matte
        # F.interpolate expects size=(height, width). Use (output_height, output_width)
        matte = F.interpolate(matte, size=(output_height, output_width), mode='area')
        matte = matte[0][0].data.cpu().numpy()

        matte_name = 'matte.png'
        Image.fromarray(((matte * 255).astype('uint8'))).save(os.path.join('results', matte_name))
        print(f"Saved matte: {matte_name}")
        _, style_img = style_video.read()
        # cv2 returns BGR arrays; convert to RGB for PIL/torch processing
        content_rgb = cv2.cvtColor(content_img, cv2.COLOR_BGR2RGB)
        style_rgb = cv2.cvtColor(style_img, cv2.COLOR_BGR2RGB)

        content = content_tf(Image.fromarray(content_rgb))
        style = style_tf(Image.fromarray(style_rgb))

        if args.preserve_color:
            style = coral(style, content)

        style = style.to(device).unsqueeze(0)
        content = content.to(device).unsqueeze(0)
        with torch.no_grad():
            output = style_transfer(vgg, decoder, content, style,
                                    args.alpha)
        # convert output tensor to HWC float in [0,1]
        output = output.cpu().numpy()  # shape (C,H,W)
        output = np.transpose(output.squeeze(0), (1,2,0)) if output.ndim==4 else np.transpose(output, (1,2,0))
        # normalize output to [0,1]
        if output.max() > 2.0:
            output = output / 255.0
        output = np.clip(output, 0.0, 1.0)
        output = cv2.resize((output * 255.0).astype(np.uint8), (output_width, output_height), interpolation=cv2.INTER_CUBIC)

        # compose using RGB content (convert BGR->RGB first), operate in float [0,1]
        content_rgb_f = (cv2.cvtColor(content_img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0)
        output_f = output.astype(np.float32) / 255.0
        matte_f = np.expand_dims(matte, axis=2)
        final_image = content_rgb_f * matte_f + output_f * (1.0 - matte_f)
        final_image = (np.clip(final_image, 0.0, 1.0) * 255.0).astype(np.uint8)
        writer.append_data(np.array(final_image))
        pbar.update(1)
        Image.fromarray(final_image).save("output.png")
        # print("Saved output image: output.png")
        # break

    style_video.release()
    content_video.release()

if style_path.suffix in [".jpg", ".png", ".JPG", ".PNG"]:

    output_video_path = output_dir / '{:s}_stylized_{:s}{:s}'.format(
                content_path.stem, style_path.stem, args.save_ext)
    writer = imageio.get_writer(output_video_path, mode='I', fps=fps)
    
    style_img = Image.open(style_path)
    while(True):
        torch.cuda.empty_cache()
        ret, content_img = content_video.read()
        if not ret:
            break
        # content_img is a numpy array from cv2; use cv2.resize instead of PIL.Image.resize
        content_img = cv2.resize(content_img, (output_width, output_height), interpolation=cv2.INTER_CUBIC)

        im = preprocess_img(content_img)
        print(im.shape)

        _, _, matte = modnet(im.to(device) if torch.cuda.is_available() else im, True)

        # resize and save matte
        # F.interpolate expects size=(height, width). Use (output_height, output_width)
        matte = F.interpolate(matte, size=(output_height, output_width), mode='area')
        matte = matte[0][0].data.cpu().numpy()

        matte_name = 'matte.png'
        # Image.fromarray(((matte * 255).astype('uint8'))).save(os.path.join('results', matte_name))
        # content_img is already resized and is BGR; convert to RGB for PIL
        content_rgb = cv2.cvtColor(content_img, cv2.COLOR_BGR2RGB)
        content = content_tf(Image.fromarray(content_rgb))
        style = style_tf(style_img)

        if args.preserve_color:
            style = coral(style, content)

        style = style.to(device).unsqueeze(0)
        content = content.to(device).unsqueeze(0)
        with torch.no_grad():
            output = style_transfer(vgg, decoder, content, style,
                                    args.alpha)
        # convert output tensor to HWC float in [0,1]
        output = output.cpu().numpy()
        output = np.transpose(output.squeeze(0), (1,2,0)) if output.ndim==4 else np.transpose(output, (1,2,0))
        if output.max() > 2.0:
            output = output / 255.0
        output = np.clip(output, 0.0, 1.0)
        output = cv2.resize((output * 255.0).astype(np.uint8), (output_width, output_height), interpolation=cv2.INTER_CUBIC)

        content_rgb_f = (cv2.cvtColor(content_img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0)
        output_f = output.astype(np.float32) / 255.0
        matte_f = np.expand_dims(matte, axis=2)
        final_image = content_rgb_f * matte_f + output_f * (1.0 - matte_f)
        final_image = (np.clip(final_image, 0.0, 1.0) * 255.0).astype(np.uint8)
        writer.append_data(np.array(final_image))
        pbar.update(1)
        Image.fromarray(final_image).save("output.png")
        print("Saved output image: output.png")
        # break
    content_video.release()