import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
from torchvision import transforms
from scipy.ndimage import morphology

def get_basename_no_ext(p):
    return os.path.splitext(os.path.basename(p))[0]


class ModNetDataLoader(Dataset):
    def __init__(self, img_dir, mask_dir, trimap_dir=None, ref_size=512):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.trimap_dir = trimap_dir
        self.ref_size = ref_size

        # match test_modnet(): image normalized to [-1,1]
        self.im_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

        # list files
        self.img_files = sorted(os.listdir(img_dir))

        # build mask map
        self.mask_dict = {get_basename_no_ext(f): f for f in os.listdir(mask_dir)}

        # trimap (optional)
        if trimap_dir:
            self.trimap_dict = {get_basename_no_ext(f): f for f in os.listdir(trimap_dir)}
        else:
            self.trimap_dict = None

        # check missing
        for img_name in self.img_files:
            b = get_basename_no_ext(img_name)
            if b not in self.mask_dict:
                print("⚠ missing mask:", img_name)
            if self.trimap_dict and b not in self.trimap_dict:
                print("⚠ missing trimap:", img_name)

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):

        img_name = self.img_files[idx]
        base = get_basename_no_ext(img_name)

        img_path = os.path.join(self.img_dir, img_name)
        mask_path = os.path.join(self.mask_dir, self.mask_dict[base])

        # ------------------ Load Image -------------------
        im = Image.open(img_path)
        im = np.asarray(im)

        # unify to RGB
        if len(im.shape) == 2:
            im = np.repeat(im[:, :, None], 3, axis=2)
        if im.shape[2] == 1:
            im = np.repeat(im, 3, axis=2)
        elif im.shape[2] == 4:
            im = im[:, :, :3]

        im = Image.fromarray(im)
        im = self.im_transform(im)       # → [-1,1]

        # ------------------ Load Mask ---------------------
        mask = Image.open(mask_path)
        mask = np.array(mask).astype(np.float32)

        # select alpha channel if RGBA
        if mask.ndim == 3:
            mask = mask[:, :, -1]

        mask = mask / 255.0              # → [0,1]
        mask = torch.from_numpy(mask).unsqueeze(0)  # shape (1,H,W)

        # ------------------ Load or Generate Trimap ------------
        if self.trimap_dict:
            tpath = os.path.join(self.trimap_dir, self.trimap_dict[base])
            trimap = np.array(Image.open(tpath)).astype(np.float32) / 255.0
            trimap = torch.from_numpy(trimap).unsqueeze(0)
        else:
            trimap = self.generate_trimap(mask.squeeze(0).numpy())

        # ------------------ Resize (same logic as test_modnet) -------------------
        im, trimap, mask = self.resize_all(im, trimap, mask)

        return im, trimap, mask

    def generate_trimap(self, alpha):
        fg = (alpha >= 1.0).astype(np.float32)
        unknown = morphology.distance_transform_edt(fg == 0) <= np.random.randint(3, 20)
        trimap = fg.copy()
        trimap[unknown] = 0.5
        return torch.from_numpy(trimap).float().unsqueeze(0)

    def resize_all(self, im, trimap, mask):

        ref = self.ref_size
        rh = ref - (ref % 32)
        rw = ref - (ref % 32)
    
        # img: bilinear/area
        im = F.interpolate(im.unsqueeze(0), size=(rh, rw), mode="area").squeeze(0)
    
        # mask: bilinear
        mask = F.interpolate(mask.unsqueeze(0), size=(rh, rw), mode="area").squeeze(0)
    
        # trimap: nearest to preserve discrete labels 0/0.5/1
        trimap = F.interpolate(trimap.unsqueeze(0), size=(rh, rw), mode="nearest").squeeze(0)
    
        return im, trimap, mask