from dataloader import CelebASubset
from loadmodels import load_models
from feature_extractor import FeatureExtractor
from bias_detector import BiasDetector

import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import gc
import torch



def run_bias_analysis():
    # Data transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    dataset = CelebASubset(
        root_dir='data/img_align_celeba',  # Path to your image folder
        attr_file='data/list_attr_celeba.txt',  # Path to your attributes file
        transform=transform,
        max_samples=1000  # Start with 5000 images for faster testing
    )
    
    # Uncomment below if you want to test with dummy data first
    # print("Creating dummy dataset for testing...")
    # dataset = create_dummy_dataset(transform)
    
    dataloader = DataLoader(dataset, batch_size=8, shuffle=False)

    
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    gc.collect()

    models = load_models()

    bias_detector = BiasDetector()

    results={}

    for model_name, model in models.items():
        print(f"\nAnalyzing {model_name}...")

        extractor = FeatureExtractor(model, layer_name='avgpool')
        features, gender_labels, age_labels = extractor.extract_features(dataloader)

        print(f"Extracted features shape: {features.shape}")

        # gender bias
        gender_linear = bias_detector.linear_probe_bias(features, gender_labels)
        # gender_geometric = bias_detector.geometric_bias(features, gender_labels)
        # gender_subspace = bias_detector.subspace_bias(features, gender_labels)



        # age bias
        age_linear = bias_detector.linear_probe_bias(features, age_labels)
        # age_geometric = bias_detector.geometric_bias(features, age_labels)
        # age_subspace = bias_detector.subspace_bias(features, age_labels)

        results[model_name] = {
            'gender': {
                'linear_probe': gender_linear['test_accuracy'],
                # 'silhouette': gender_geometric['silhouette_score'],
                # 'pca_bias': gender_subspace['pca_bias_score']
            },
            'age': {
                'linear_probe': age_linear['test_accuracy'],
                # 'silhouette': age_geometric['silhouette_score'],
                # 'pca_bias': age_subspace['pca_bias_score']
            }
        }

        print(f"Gender bias - Linear probe: {gender_linear['test_accuracy']:.3f}")
        
        # print(f"Gender bias - Silhouette: {gender_geometric['silhouette_score']:.3f}")
        print(f"Age bias - Linear probe: {age_linear['test_accuracy']:.3f}")
        # print(f"Age bias - Silhouette: {age_geometric['silhouette_score']:.3f}")
    
    return results
    


if __name__ == "__main__":
    results = run_bias_analysis()
    breakpoint()
    