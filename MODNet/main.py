import json
import os
import pandas as pd
import cv2

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

from modnet_dataset import ModNetDataLoader
from src.models.modnet import MODNet
from src.trainer import supervised_training_iter
from utils import compute_mse_mad



def compute_metrics_on_loader(loader, model, device):
    model.eval()
    total_mse = 0.0
    total_mad = 0.0
    total_images = 0

    with torch.no_grad():
        for img, trimap, matte in loader:
            img = img.to(device)
            matte = matte.to(device)
            _, _, pred_matte = model(img, False)

            if pred_matte.shape != matte.shape:
                pred_matte = torch.nn.functional.interpolate(
                    pred_matte, 
                    size=matte.shape[-2:], 
                    mode='bilinear'
                )

            pred_matte = pred_matte.clamp(0, 1)

            batch_size = img.size(0)
            mse = torch.mean((pred_matte - matte) ** 2).item()
            mad = torch.mean(torch.abs(pred_matte - matte)).item()

            total_mse += mse * batch_size
            total_mad += mad * batch_size
            total_images += batch_size

    avg_mse = total_mse / total_images if total_images > 0 else 0.0
    avg_mad = total_mad / total_images if total_images > 0 else 0.0
    model.train()
    return avg_mse, avg_mad


def train_modnet(config, dataset, test_dataset, val_dataset):
    batch_size = config.get('batch_size', 8)
    lr = config.get('lr', 0.01)
    epochs = config.get('epochs', 40)  
    mse_test_best = config.get('mse_test_best', 1e6)
    semantic_scale=config.get('semantic_scale', 1.0)
    detail_scale=config.get('detail_scale', 10.0)
    matte_scale=config.get('matte_scale', 1.0)
    output_dir = config.get('output_dir', 'models/P3M-10k-1-10-1')

    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    print(f"Dataset size: {len(dataset)}")
    print("DataLoader created.")

    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    print(f"Test Dataset size: {len(test_dataset)}")
    print("Test DataLoader created.")

    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    print(f"Validation Dataset size: {len(val_dataset)}")
    print("Validation DataLoader created.")
    
    result = []
    losses = []

    device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
    # device = torch.device("cpu")

    # modnet = torch.nn.DataParallel(MODNet()).cuda()
    # state_dict = torch.load(ckpt_path)
    modnet = MODNet(backbone_pretrained=True)
    # modnet.load_state_dict(state_dict)
    modnet = modnet.to(device)
    modnet.train()
    optimizer = torch.optim.SGD(modnet.parameters(), lr=lr, momentum=0.9)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=int(0.25 * epochs), gamma=0.1)

    print('Starting training...')
    test_mse, test_mad = compute_metrics_on_loader(test_loader, modnet, device)
    val_mse, val_mad = compute_metrics_on_loader(val_loader, modnet, device)
    print(f"start: Test MSE={test_mse}, Test MAD={test_mad}, Val MSE={val_mse}, Val MAD={val_mad}")
    result.append((-1, test_mse, test_mad, val_mse, val_mad)) 

    for epoch in range(0, epochs):
        if epoch != 0 and (epoch + 1) % 10 == 0:
            lr *= 0.1
            print(f"Learning rate adjusted to: {lr}")
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
        print(f"Starting epoch {epoch+1}/{epochs}...")
        for idx, (image, trimap, gt_matte) in enumerate(train_loader):
            image = image.to(device)
            trimap = trimap.to(device)
            gt_matte = gt_matte.to(device)
            semantic_loss, detail_loss, matte_loss = supervised_training_iter(modnet, optimizer, image, trimap, gt_matte, semantic_scale, detail_scale, matte_scale)
            print(f"Epoch [{epoch+1}/{epochs}], Step [{idx+1}/{len(train_loader)}], Semantic Loss: {semantic_loss:.4f}, Detail Loss: {detail_loss:.4f}, Matte Loss: {matte_loss:.4f}")
            losses.append((semantic_loss.item(), detail_loss.item(), matte_loss.item()))
        lr_scheduler.step()

        test_mse, test_mad = compute_metrics_on_loader(test_loader, modnet, device)
        val_mse, val_mad = compute_metrics_on_loader(val_loader, modnet, device)
        print(f"Epoch {epoch+1}:Test MSE={test_mse}, Test MAD={test_mad}, Val MSE={val_mse}, Val MAD={val_mad}")
        result.append((epoch, test_mse, test_mad, val_mse, val_mad))

        if test_mse < mse_test_best:
            mse_test_best = test_mse
            torch.save(modnet.state_dict(), f"{output_dir}/best_test_finetuned_model.pth")
            print(f"Best model saved at epoch {epoch+1} with Test MSE={test_mse}")
        
        if epoch % 5 == 0:
            torch.save(modnet.state_dict(), f"{output_dir}/finetuned_epoch_{epoch}.pth")
        print(f"Epoch {epoch} done.")
        
        losses_df = pd.DataFrame(losses, columns=['semantic_loss', 'detail_loss', 'matte_loss'])
        losses_df.to_csv(f'{output_dir}/training_losses.csv', index=False)
        result_df = pd.DataFrame(result, columns=['epoch', 'test_mse', 'test_mad', 'val_mse', 'val_mad'])
        result_df.to_csv(f'{output_dir}/training_metrics.csv', index=False)

    torch.save(modnet.state_dict(), f"{output_dir}/finetuned_epoch_{epoch}.pth")

    
    losses = pd.DataFrame(losses, columns=['semantic_loss', 'detail_loss', 'matte_loss'])
    losses.to_csv(f'{output_dir}/training_losses.csv', index=False)

if __name__ == "__main__":
    config = json.load(open('config/1-10-1.json'))
    dataset = ModNetDataLoader(
        img_dir="../dataset/P3M-10k/train/blurred_image",
        mask_dir="../dataset/P3M-10k/train/mask",
        trimap_dir = None,
        ref_size=512
    )

    test_dataset = ModNetDataLoader(
        img_dir="../dataset/P3M-10k/validation/P3M-500-P/blurred_image",
        mask_dir="../dataset/P3M-10k/validation/P3M-500-P/mask",
        trimap_dir = '../dataset/P3M-10k/validation/P3M-500-P/trimap',
        ref_size=512
    )

    val_dataset = ModNetDataLoader(
        img_dir="../dataset/PPM-100/image",
        mask_dir="../dataset/PPM-100/matte",
        trimap_dir = '',
        ref_size=512
    )

    train_modnet(config, dataset, test_dataset, val_dataset)