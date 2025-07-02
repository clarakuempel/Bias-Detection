import torchvision

def load_models():
    """Load pre-trained models for comparison"""
    models = {}
    
    # 1. Supervised ResNet18 (fix deprecation warning)
    models['supervised_resnet18'] = torchvision.models.resnet18(weights='IMAGENET1K_V1')
    
    # 2. Supervised ResNet50 (fix deprecation warning)
    # models['supervised_resnet50'] = torchvision.models.resnet50(weights='IMAGENET1K_V1')
    
    # # 3. For self-supervised, we'll use a pre-trained model
    # # Note: You might need to download specific self-supervised weights
    # # For now, we'll use a standard ResNet as placeholder
    # models['self_supervised'] = torchvision.models.resnet18(weights='IMAGENET1K_V1')
    
    return models