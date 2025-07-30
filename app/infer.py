import yaml
import torch
import torch.nn.functional as F
def infer_image(clip_model, model, preprocess, image, device):
    # image = Image.open(image).convert('RGB')
    image_tensor = preprocess(image).unsqueeze(0).to(device)

    category = "defect"  # known from training
    
    with torch.no_grad():
        image_features = clip_model.encode_image(image_tensor)
        predictions = model(image_features, category)
        logits = predictions["defect_scratch"]  # shape [1, 2]
        probs = F.softmax(logits, dim=1)
        pred_class = torch.argmax(probs, dim=1).item()

        class_names = ["no_scratches", "scratches"]
        pred_label = class_names[pred_class]
        
        return pred_label, probs.cpu().numpy()