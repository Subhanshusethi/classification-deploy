import torch
import torch.nn as nn
import torch.nn.functional as F
import open_clip

class CategoryAwareAttributePredictor(nn.Module):
    def __init__(self, clip_dim=768, category_attributes=None, attribute_dims=None, hidden_dim=512, dropout_rate=0.2, num_hidden_layers=1):
        super(CategoryAwareAttributePredictor, self).__init__()
        
        self.category_attributes = category_attributes
        self.attribute_predictors = nn.ModuleDict()
        
        for category, attributes in category_attributes.items():
            for attr_name in attributes.keys():
                key = f"{category}_{attr_name}"
                if key in attribute_dims:
                    layers = []
                    # Input layer
                    layers.append(nn.Linear(clip_dim, hidden_dim))
                    layers.append(nn.LayerNorm(hidden_dim))
                    layers.append(nn.ReLU())
                    layers.append(nn.Dropout(dropout_rate))
                    
                    # Additional hidden layers
                    current_dim = hidden_dim
                    for _ in range(num_hidden_layers - 1):
                        layers.append(nn.Linear(current_dim, current_dim // 2))
                        layers.append(nn.LayerNorm(current_dim // 2))
                        layers.append(nn.ReLU())
                        layers.append(nn.Dropout(dropout_rate))
                        current_dim = current_dim // 2

                    # Output layer
                    layers.append(nn.Linear(current_dim, attribute_dims[key]))
                    
                    self.attribute_predictors[key] = nn.Sequential(*layers)
    
    def forward(self, clip_features, category):
        results = {}
        category_attrs = self.category_attributes[category]
        clip_features = clip_features.float()
        
        for attr_name in category_attrs.keys():
            key = f"{category}_{attr_name}"
            if key in self.attribute_predictors:
                results[key] = self.attribute_predictors[key](clip_features)
        
        return results