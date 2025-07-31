from fastapi import FastAPI,  UploadFile
from fastapi import Request
from PIL import Image
import numpy as np 
import torch
import torch.nn as nn
import torch.nn.functional as F
import open_clip
from PIL import Image
import os
from app.model import CategoryAwareAttributePredictor
from app.utils import load_config, load_models, custom_round
from app.infer import infer_image

app = FastAPI()

@app.on_event("startup")
def startup_event():
    config = load_config("config.yml")
    checkpoint_path = "ckpts/binary_checkpoint_epoch1.pth"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    clip_model, model, preprocess_val = load_models(config, checkpoint_path, device)

    app.state.model = model
    app.state.clip_model = clip_model
    app.state.preprocess = preprocess_val
    app.state.device = device
# config_path = "config.yml"
# checkpoint_path = "ckpts/binary_checkpoint_epoch1.pth"
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# config = load_config(config_path)
# clip_model, model, preprocess_val = load_models(config, checkpoint_path, device)

@app.post("/upload_image")
def predict(request: Request,file: UploadFile):
    try:
        image = Image.open(file.file).convert('RGB')
    except Exception as e:
        return {"error": f"Failed to process image: {str(e)}"}
    
    model = request.app.state.model
    clip_model = request.app.state.clip_model
    preprocess = request.app.state.preprocess
    device = request.app.state.device

    pred_label, probs = infer_image(clip_model, model, preprocess, image, device)
    print(f"Predicted label: {pred_label}, Probabilities: {probs}")
    return {'pred' : pred_label,
            'prob_no_scratches' : custom_round(probs[0][0]),
            'prob_scratches' : custom_round(probs[0][1]),
    }
