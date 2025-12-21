import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms

from PIL import Image
from flask import Flask, render_template, request, send_from_directory, url_for

from MODNet.src.models.modnet import MODNet
from utils import transfer_bg

import AdaIN.net as net
from AdaIN.function import adaptive_instance_normalization, coral


app = Flask(__name__)

UPLOAD_FOLDER = "upload"
OUTPUT_FOLDER = "static/output"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/transfer", methods=["POST"])
def transfer():
    content = request.files["content"]
    style = request.files["style"]

    content_path = os.path.join(UPLOAD_FOLDER, "content.jpg")
    style_path = os.path.join(UPLOAD_FOLDER, "style.jpg")
    output_path = os.path.join(OUTPUT_FOLDER, "output.png")


    content.save(content_path)
    style.save(style_path)

    # ==============================
    # MOCK STYLE TRANSFER (demo)
    # ==============================
    # Thay đoạn này bằng model thật của bạn
    ckpt_path = "MODNet/models/P3M-10k-1-20-1/best_test_finetuned_model.pth"

    device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
    # device = torch.device("cpu")
    modnet = MODNet(backbone_pretrained=False)
    state_dict = torch.load(ckpt_path, map_location='cpu')

    if list(state_dict.keys())[0].startswith("module."):
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}

    modnet.load_state_dict(state_dict)

    modnet = modnet.to(device)
    modnet.eval()

    print(1)
    output_img = transfer_bg(
        modnet,
        device,
        content_img=Image.open(content_path),
        style_img=Image.open(style_path)
    )
    output_img = Image.fromarray(output_img.astype(np.uint8))
    output_img.save(output_path)

    return {"output_url": output_path}



if __name__ == "__main__":
    app.run(debug=True)
