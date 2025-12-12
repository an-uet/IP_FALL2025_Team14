import torch
from torchvision import models, transforms
from PIL import Image
import numpy as np
import cv2
import pandas as pd

import os
import time

def gen_mask_deeplabv3(model, device, dest_path, src_path):
    if not os.path.exists(f"{dest_path}"):
        os.makedirs(f"{dest_path}")

    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    
    list_img = os.listdir(src_path)
    process_time = []
    for img_name in list_img:
        t1 = time.time()
        img_path = os.path.join(src_path, img_name)
        print(img_path)
        
        img = Image.open(img_path).convert("RGB")

        orig_w, orig_h = img.size

        inp = preprocess(img).unsqueeze(0).to(device)
        with torch.no_grad():
            output = model(inp)["out"]  

        pred = torch.argmax(output.squeeze(), dim=0).cpu().numpy()
        
        PERSON_CLASS = 15  
        mask = (pred == PERSON_CLASS).astype(np.uint8) * 255 
        mask = cv2.resize(mask, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
        cv2.imwrite(f"{dest_path}/{img_name}", mask)

        t2 = time.time()
        process_time.append((img_name, t2 - t1))
        print(f"DONE! Saved mask to {img_name}")

    avg_process_time = np.mean([pt[1] for pt in process_time])
    print(f"Average processing time per image: {avg_process_time} seconds")

    return process_time

def compute_mse_mad(pred_src, gt_src):
    list_img = os.listdir(gt_src)
    results = []
    for img_name in list_img:
        pred_path = os.path.join(pred_src, img_name)
        gt_path = os.path.join(gt_src, img_name)
        
        pred = cv2.imread(pred_path, cv2.IMREAD_GRAYSCALE) / 255.0
        gt   = cv2.imread(gt_path,   cv2.IMREAD_GRAYSCALE) / 255.0
        
        if pred.shape != gt.shape:
            gt = cv2.resize(gt, (pred.shape[1], pred.shape[0]))

        mse = np.mean((pred - gt) ** 2)
        mad = np.mean(np.abs(pred - gt))

        results.append((img_name, mse, mad))

    avg_mse = np.mean([r[1] for r in results])
    avg_mad = np.mean([r[2] for r in results])
    print(f"Average MSE: {avg_mse}, Average MAD: {avg_mad}")

    return results

if __name__ == "__main__":
    model = models.segmentation.deeplabv3_resnet101(pretrained=True).eval()
    device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    print("Model loaded to", device)

    # generate masks for Distinctions-646 dataset
    src_dir = "../dataset/Distinctions-646-new/Train/FG"
    dest_path = '../results/deeplabv3_masks_Distinctions-646'
    process_time = gen_mask_deeplabv3(model, device, dest_path, src_dir)
    process_time = pd.DataFrame(process_time, columns=['image_name', 'process_time'])
    process_time.to_csv('../results/deeplabv3_masks_Distinctions-646_process_time.csv', index=False)
  
