##################
# CATEGORY_MAPPING
##################
from model import CategoryAwareAttributePredictor
import yaml
import torch
import torch.nn.functional as F
import open_clip
CATEGORY_MAPPING = {
    "defect": {
        "scratch": "class"
    }
}
def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def load_models(config, checkpoint_path, device):
    # Create CLIP model and transforms
    clip_model, preprocess_train, preprocess_val = open_clip.create_model_and_transforms(
        config['model']['name'],
        pretrained=config['model']['pretrained'],
        device=device
    )
    clip_model = clip_model.float()
    
    # Define attribute_dims (binary classification: 2 classes)
    attribute_dims = {"defect_scratch": 2}
    
    model = CategoryAwareAttributePredictor(
        clip_dim=config['model']['clip_dim'],
        category_attributes=CATEGORY_MAPPING,
        attribute_dims=attribute_dims,
        hidden_dim=config['model']['hidden_dim'][0],
        dropout_rate=config['model']['dropout_rate'][0],
        num_hidden_layers=config['model']['num_hidden_layers'][0]
    ).to(device)

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    clip_model.load_state_dict(checkpoint['clip_model_state_dict'])

    model.eval()
    clip_model.eval()
    
    return clip_model, model, preprocess_val

def custom_round(number):
    if number > 0.5:
        return 1
    else:
        return 0