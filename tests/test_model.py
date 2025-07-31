import torch
from PIL import Image
from app.utils import load_config, load_models, custom_round
from app.infer import infer_image

def test_model_inference_runs():
    # Setup
    config = load_config("config.yml")
    checkpoint_path = "ckpts/binary_checkpoint_epoch1.pth"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    clip_model, model, preprocess = load_models(config, checkpoint_path, device)

    # Load test image
    image = Image.open("tests/test_image.png").convert("RGB")

    # Run inference
    pred_label, probs = infer_image(clip_model, model, preprocess, image, device)

    # Basic checks
    assert pred_label in ["scratches", "no_scratches"]
    assert probs.shape == (1, 2)
    assert abs(probs[0].sum() - 1.0) < 1e-4
