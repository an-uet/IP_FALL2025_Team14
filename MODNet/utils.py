import os
import pandas as pd
import cv2

import numpy as np
import os

# Compute MSE and MAD between 2 source folders of images with ground truth
def compute_metrics(pred_baseline, pred_finetune, gt_src, change_type=False):
    list_img = sorted(os.listdir(gt_src))
    results = []
    missing_pred = []
    missing_gt = []

    for img_name in list_img:
        # if change_type:
        #     pred_path1 = os.path.join(pred_baseline, img_name.replace('.jpg', '.png'))
        #     pred_path2 = os.path.join(pred_finetune, img_name.replace('.jpg', '.png'))
        # else:
        pred_path1 = os.path.join(pred_baseline, img_name)
        pred_path2 = os.path.join(pred_finetune, img_name)
        gt_path = os.path.join(gt_src, img_name)

        if not os.path.exists(gt_path):
            missing_gt.append(gt_path)
            print(f"Ground truth image missing: {gt_path}")
            continue

        if not os.path.exists(pred_path1):
            missing_pred.append(pred_path1)
            print(f"Predicted image 1 missing: {pred_path1}")
            continue

        if not os.path.exists(pred_path2):
            missing_pred.append(pred_path2)
            print(f"Predicted image 2 missing: {pred_path2}")
            continue

        pred_img1 = cv2.imread(pred_path1, cv2.IMREAD_GRAYSCALE)
        pred_img2 = cv2.imread(pred_path2, cv2.IMREAD_GRAYSCALE)
        gt_img = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)

        if pred_img1 is None:
            missing_pred.append(pred_path1)
            continue

        if pred_img2 is None:
            missing_pred.append(pred_path2)
            continue

        if gt_img is None:
            missing_gt.append(gt_path)
            continue

        # convert to float in [0,1]
        pred1 = pred_img1.astype(np.float32) / 255.0
        pred2 = pred_img2.astype(np.float32) / 255.0
        gt = gt_img.astype(np.float32) / 255.0
        gt1 = gt
        gt2 = gt

        # resize gt to match pred if shapes differ
        if pred1.shape != gt.shape and pred2.shape != gt.shape:
            gt1 = cv2.resize(gt, (pred1.shape[1], pred1.shape[0]), interpolation=cv2.INTER_LINEAR)
            gt2 = cv2.resize(gt, (pred2.shape[1], pred2.shape[0]), interpolation=cv2.INTER_LINEAR)
        mse = float(np.mean((pred1 - gt1) ** 2))
        mad = float(np.mean(np.abs(pred1 - gt1)))

        mse2 = float(np.mean((pred2 - gt2) ** 2))
        mad2 = float(np.mean(np.abs(pred2 - gt2)))

        results.append((img_name, mse, mad, mse2, mad2))
    

    if len(results) == 0:
        print("No image pairs found to compute metrics.")
        return []

    avg_mse = float(np.mean([r[1] for r in results]))
    avg_mad = float(np.mean([r[2] for r in results]))
    avg_mse2 = float(np.mean([r[3] for r in results]))
    avg_mad2 = float(np.mean([r[4] for r in results]))
    results = pd.DataFrame(results, columns=['Image', 'MSE_baseline', 'MAD_baseline', 'MSE_finetuning', 'MAD_finetuning'])
    results.to_csv('modnet_evaluation_results.csv', index=False)
    print(f"Average MSE_baseline: {avg_mse:.6f}, Average MAD_baseline: {avg_mad:.6f}")
    print(f"Average MSE fine tuning: {avg_mse2:.6f}, Average MAD fine tuning: {avg_mad2:.6f}")

    if missing_pred:
        print(f"Warning: {len(missing_pred)} predicted images missing. Examples: {missing_pred[:3]}")
    if missing_gt:
        print(f"Warning: {len(missing_gt)} ground-truth images missing. Examples: {missing_gt[:3]}")

    return results


def compute_mse_mad(pred_src, gt_src, out_dir, change_type=False):
    list_img = sorted(os.listdir(gt_src))
    results = []
    missing_pred = []
    missing_gt = []

    for img_name in list_img:
        if change_type:
            pred_path = os.path.join(pred_src, img_name.replace('.jpg', '.png'))
        #     pred_path2 = os.path.join(pred_finetune, img_name.replace('.jpg', '.png'))
        else:
            pred_path = os.path.join(pred_src, img_name)
        gt_path = os.path.join(gt_src, img_name)

        if not os.path.exists(gt_path):
            missing_gt.append(gt_path)
            print(f"Ground truth image missing: {gt_path}")
            continue

        if not os.path.exists(pred_path):
            missing_pred.append(pred_path)
            print(f"Predicted image missing: {pred_path}")
            continue

        pred_img = cv2.imread(pred_path, cv2.IMREAD_GRAYSCALE)
        gt_img = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)

        if pred_img is None:
            missing_pred.append(pred_path)
            continue

        if gt_img is None:
            missing_gt.append(gt_path)
            continue

        # convert to float in [0,1]
        pred = pred_img.astype(np.float32) / 255.0
        gt = gt_img.astype(np.float32) / 255.0
        gt1 = gt
        gt2 = gt

        # resize gt to match pred if shapes differ
        if pred.shape != gt.shape:
            gt = cv2.resize(gt, (pred.shape[1], pred.shape[0]), interpolation=cv2.INTER_LINEAR)
        mse = float(np.mean((pred - gt) ** 2))
        mad = float(np.mean(np.abs(pred - gt)))

        results.append((img_name, mse, mad))
    

    if len(results) == 0:
        print("No image pairs found to compute metrics.")
        return []

    avg_mse = float(np.mean([r[1] for r in results]))
    avg_mad = float(np.mean([r[2] for r in results]))
    results = pd.DataFrame(results, columns=['Image', 'MSE', 'MAD'])
    results.to_csv(os.path.join(out_dir, 'modnet_evaluation_results.csv'), index=False)
    print(f"Average MSE: {avg_mse:.6f}, Average MAD: {avg_mad:.6f}")

    if missing_pred:
        print(f"Warning: {len(missing_pred)} predicted images missing. Examples: {missing_pred[:3]}")
    if missing_gt:
        print(f"Warning: {len(missing_gt)} ground-truth images missing. Examples: {missing_gt[:3]}")

    return avg_mse, avg_mad


def preprocess_distinction_646(src_dir, dest_dir):
    count = 0
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)

    for img_name in os.listdir(src_dir):
        if img_name.startswith('h_'):
            count += 1
            print(f'Processing {img_name}...')
            src_path = os.path.join(src_dir, img_name)
            dest_path = os.path.join(dest_dir, img_name)
            cv2.imwrite(dest_path, cv2.imread(src_path))
    print(f'Total {count} images processed and saved to {dest_dir}.')
    
if __name__ == "__main__":
    src_root_path = './dataset/Distinctions-646/'
    dest_root_path = './dataset/Distinctions-646-new/'
    list_subsets = ['Train', 'Test']
    for subset in list_subsets:
        src = os.path.join(src_root_path, subset, 'FG')
        dest = os.path.join(dest_root_path, subset, 'FG')
        preprocess_distinction_646(src, dest)

        src = os.path.join(src_root_path, subset, 'GT')
        dest = os.path.join(dest_root_path, subset, 'GT')
        preprocess_distinction_646(src, dest)

        

