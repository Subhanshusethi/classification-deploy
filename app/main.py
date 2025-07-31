from fastapi import FastAPI,  UploadFile
from fastapi import Request
from PIL import Image
import numpy as np 
import torch
import torch.nn as nn
import torch.nn.functional as F
import open_clip
from PIL import Image
import time
from datetime import datetime
import json
import os
from app.model import CategoryAwareAttributePredictor
from app.utils import load_config, load_models, custom_round
from app.infer import infer_image

app = FastAPI()

import smtplib
from email.message import EmailMessage

def send_email_alert(subject, body):
    from_email = "sethisubhanshu@gmail.com"
    to_email = "subhanshusethi38@gmail.com"
    app_password = "zoliomsjhzmsfxkg"

    if not all([from_email, to_email, app_password]):
        print("Email alert skipped: missing env vars")
        return

    msg = EmailMessage()
    msg.set_content(body)
    msg["Subject"] = subject
    msg["From"] = from_email
    msg["To"] = to_email

    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as smtp:
            smtp.login(from_email, app_password)
            smtp.send_message(msg)
        print("Email alert sent")
    except Exception as e:
        print("Failed to send email:", e)

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
def predict(request: Request, file: UploadFile):
    start_time = time.time()
    try:
        image = Image.open(file.file).convert('RGB')
    except Exception as e:
        error_msg = f"🚨 Image processing failed\nFile: {file.filename}\nError: {str(e)}"
        send_email_alert("Model Alert: Image Error", error_msg)
        return {"error": f"Failed to process image: {str(e)}"}

    model = request.app.state.model
    clip_model = request.app.state.clip_model
    preprocess = request.app.state.preprocess
    device = request.app.state.device

    pred_label, probs = infer_image(clip_model, model, preprocess, image, device)
    end_time = time.time()

    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "image_name": file.filename,
        "predicted_label": pred_label,
        "probabilities": probs.tolist(),  # safer for JSON
        "latency": round(end_time - start_time, 4)
    }

    with open("inference_log.txt", "a") as log_file:
        log_file.write(json.dumps(log_entry) + "\n")

    return {
        'pred': pred_label,
        'prob_no_scratches': custom_round(probs[0][0]),
        'prob_scratches': custom_round(probs[0][1]),
    }