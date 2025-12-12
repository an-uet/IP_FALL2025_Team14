import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms

from PIL import Image
import time

from src.models.modnet import MODNet
from utils import compute_mse_mad


def test_modnet(ckpt_path, input_path, output_path):
    ref_size = 512
    result = []

    # define image to tensor transform
    im_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]
    )

    device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')
    # device = torch.device("cpu")
    modnet = MODNet(backbone_pretrained=False)
    state_dict = torch.load(ckpt_path, map_location='cpu')

    if list(state_dict.keys())[0].startswith("module."):
        print("Detected DataParallel ckpt → removing 'module.'")
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    else:
        print("Normal checkpoint")

    modnet.load_state_dict(state_dict)

    modnet = modnet.to(device)
    modnet.eval()

    im_names = os.listdir(input_path)
    if not os.path.exists(output_path):
        os.makedirs(output_path)


    for im_name in im_names:
        # print('Process image: {0}'.format(im_name))

        im = Image.open(os.path.join(input_path, im_name))
        
        im = np.asarray(im)
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

        # resize image for input
        im_b, im_c, im_h, im_w = im.shape
        if max(im_h, im_w) < ref_size or min(im_h, im_w) > ref_size:
            if im_w >= im_h:
                im_rh = ref_size
                im_rw = int(im_w / im_h * ref_size)
            elif im_w < im_h:
                im_rw = ref_size
                im_rh = int(im_h / im_w * ref_size)
        else:
            im_rh = im_h
            im_rw = im_w
        
        im_rw = im_rw - im_rw % 32
        im_rh = im_rh - im_rh % 32
        im = F.interpolate(im, size=(im_rh, im_rw), mode='area')

        # inference
        t1 = time.time()
        _, _, matte = modnet(im.to(device) if torch.cuda.is_available() else im, True)
        t2 = time.time()
        process_time = t2 - t1
        # print(f'Inference time: {t2 - t1} seconds.')
        result.append((im_name, process_time))
        # resize and save matte
        matte = F.interpolate(matte, size=(im_h, im_w), mode='area')
        matte = matte[0][0].data.cpu().numpy()
        matte_name = im_name.split('.')[0] + '.png'
        Image.fromarray(((matte * 255).astype('uint8')), mode='L').save(os.path.join(output_path, matte_name))
    result = pd.DataFrame(result, columns=['image_name', 'process_time'])
    result.to_csv(os.path.join(output_path, 'modnet_inference_time.csv'), index=False)
    print('avg process time per image:', float(np.mean([r[1] for r in result.values])))


if __name__ == "__main__":

    configs = ['1-10-1', '1-15-1', '1-20-1', '5-10-1']
    root_path = '../dataset/'
    result_folder = 'results'
    result = []

    for config in configs:
        ckpt_path = f"./models/P3M-10k-{config}/best_test_finetuned_model.pth"
        
        # P3M-10k test: P3M-500-P
        input_path = os.path.join(root_path, 'P3M-10k/validation/P3M-500-P/blurred_image')
        gt_path = os.path.join(root_path, 'P3M-10k/validation/P3M-500-P/mask')
        output_path = os.path.join(root_path, f'{result_folder}/modnet_{config}_P3M-500-P')
        test_modnet(ckpt_path, input_path, output_path)
        avg_mse, avg_mad = compute_mse_mad(output_path, gt_path, output_path)
        result.append((config, 'P3M-500-P', avg_mse, avg_mad))

        print('---------------------------------------')

        # P3M-10k test: P3M-500-NP
        input_path = os.path.join(root_path, 'P3M-10k/validation/P3M-500-NP/original_image')
        gt_path = os.path.join(root_path, 'P3M-10k/validation/P3M-500-NP/mask')
        output_path = os.path.join(root_path, f'{result_folder}/modnet_{config}_P3M-500-NP')
        test_modnet(ckpt_path, input_path, output_path)
        avg_mse, avg_mad = compute_mse_mad(output_path, gt_path, output_path)
        result.append((config, 'P3M-500-NP', avg_mse, avg_mad))

         # PPM-100
        input_path = os.path.join(root_path, 'PPM-100/image')
        gt_path = os.path.join(root_path, 'PPM-100/matte')
        output_path = os.path.join(root_path, f'{result_folder}/modnet_{config}_PPM-100')
        test_modnet(ckpt_path, input_path, output_path)
        avg_mse, avg_mad = compute_mse_mad(output_path, gt_path, output_path, change_type=True)
        result.append((config, 'PPM-100', avg_mse, avg_mad))
        print('---------------------------------------')

         # AIsegment
        input_path = os.path.join(root_path, 'AIsegment/test_1')
        gt_path = os.path.join(root_path, 'AIsegment/test/matte')
        output_path = os.path.join(root_path, f'{result_folder}/modnet_{config}_AIsegment')
        test_modnet(ckpt_path, input_path, output_path)
        avg_mse, avg_mad = compute_mse_mad(output_path, gt_path, output_path, change_type=True)
        result.append((config, 'AIsegment', avg_mse, avg_mad))
        print('---------------------------------------')
    result = pd.DataFrame(result, columns=['Configuration', 'Dataset', 'Average MSE', 'Average MAD'])
    result.to_csv(os.path.join(root_path, f'{result_folder}/modnet_evaluation_results.csv'), index=False)
    print('Final evaluation results saved.')



