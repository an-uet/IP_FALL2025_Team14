# Image Background Re-styling

This folder contains utilities and example scripts for background style transfer used in human portraint image and video . It combines two main components:

- MODNet: (foreground/background matte estimation)
- AdaIN: Style transfer

The scripts here are lightweight wrappers that load pretrained models, run
inference on images or videos, and compose stylized backgrounds while keeping
foreground subjects.

<video width="640" controls>
  <source src="output/video_test_stylized_scream.mp4" type="video/mp4">
  Your browser does not support the video tag.
</video>




## Contents

- `video_bg_transfer.py` — stylize a video using AdaIN + MODNet matte per-frame (video input or single style image).
- `bg_transfer.py` — single-image background transfer + style pipeline.
- `AdaIN/` — AdaIN implementation, pretrained `decoder.pth` and `vgg_normalised.pth` expected under `AdaIN/models/`.
- `MODNet/` — MODNet implementation and pretrained checkpoints under `MODNet/models/`.
- `output/` — default output directory used by scripts (created at runtime).

## Requirements

Recommended Python environment (example):

- Python 3.10+
- PyTorch (compatible with your CUDA) — e.g. 2.x
- torchvision
- opencv-python
- imageio[ffmpeg]
- pillow
- tqdm

Install dependencies (pip):

```bash
pip install -r requirements.txt
# If you plan to write MP4 with imageio, ensure ffmpeg backend is available:
pip install "imageio[ffmpeg]" imageio-ffmpeg
```

## Setup / pretrained models

1. Download AdaIN pretrained weights and put them in `AdaIN/models/`:
   - `decoder.pth`
   - `vgg_normalised.pth`

   Official release (example): https://github.com/naoto0804/pytorch-AdaIN/releases

2.  MODNet checkpoint: Place under `MODNet/models/` (example path used in scripts):
   - `MODNet/models/P3M-10k-1-20-1/best_finetuned_model.pth`

## Usage examples

All examples assume you are in the `image_processing/medical/` directory.

Single-image background transfer (compose a stylized background):

```bash
python bg_transfer.py
# This script currently loads hard-coded example paths in the __main__ block —
# modify the call at the bottom or adapt to your input files.
```

Stylize a video (style is a single image):

```bash
python test_video.py --content_video /path/to/content.mp4 \
    --style_path AdaIN/input/style/mondrian.jpg
```

Stylize a video using another video as style (frame-aligned):

```bash
python test_video.py --content_video /path/to/content.mp4 \
    --style_path /path/to/style_video.mp4
```

Key options for `test_video.py` (see `--help` for full list):

- `--content_size`, `--style_size` — minimum sizes for transforms (set 0 to keep original)
- `--alpha` — stylization strength (0.0-1.0)
- `--preserve_color` — preserve content color using CORAL
- `--output` — output directory (default `output`)

## Fine-tune MODNet

Quick guide for fine-tuning MODNet on your own dataset.

### Data requirements
- Input images should be RGB (e.g. `images/`) and you must provide matching
    ground-truth mattes/alpha maps (grayscale) of the same resolution.
- Example folder layout:
    - data/
        - images/   (RGB input images)
        - mattes/   (ground-truth alpha maps, values 0..255 or normalized 0..1)

### Example training command
You can config some parameters for train model in MODNet/config. For train model, run command:

```bash
python MODNet/main.py
```
### For test MODNet model
You can config the src of testing dataset and run command:
```bash 
python MODNet/test.py
```
