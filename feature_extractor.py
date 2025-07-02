import numpy as np
import torch

class FeatureExtractor:
    def __init__(self, model, layer_name='avgpool'):
        self.model = model
        self.model.eval()
        self.features = []
        self.layer_name = layer_name
        self._register_hook()
    
    def _register_hook(self):
        def hook_fn(module, input, output):
            # Flatten the features
            self.features.append(output.detach().cpu().numpy().reshape(output.size(0), -1))
        
        # Register hook to the specified layer
        if self.layer_name == 'avgpool':
            self.model.avgpool.register_forward_hook(hook_fn)
        elif self.layer_name == 'fc':
            self.model.fc.register_forward_hook(hook_fn)
    
    def extract_features(self, dataloader):
        self.features = []
        all_gender = []
        all_age = []
        
        with torch.no_grad():
            for images, gender, age in dataloader:
                _ = self.model(images)  # Forward pass triggers hook
                all_gender.extend(gender.numpy())
                all_age.extend(age.numpy())
        
        features = np.vstack(self.features)
        return features, np.array(all_gender), np.array(all_age)