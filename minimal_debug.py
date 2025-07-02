# Fix for Mac OpenMP issue
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import sys
import numpy as np

def debug_step_by_step():
    """Debug your exact pipeline step by step"""
    
    print("=== STEP-BY-STEP DEBUG ===")
    
    # Step 1: Test imports
    print("1. Testing imports...")
    try:
        from dataloader import CelebASubset
        print("✅ CelebASubset imported")
        
        from loadmodels import load_models
        print("✅ load_models imported")
        
        from feature_extractor import FeatureExtractor
        print("✅ FeatureExtractor imported")
        
        from bias_detector import BiasDetector
        print("✅ BiasDetector imported")
        
        import torchvision.transforms as transforms
        from torch.utils.data import DataLoader
        print("✅ All imports successful")
        
    except Exception as e:
        print(f"❌ Import failed: {e}")
        return False
    
    # Step 2: Test dataset creation
    print("\n2. Testing dataset creation...")
    try:
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        dataset = CelebASubset(
            root_dir='data/img_align_celeba',
            attr_file='data/list_attr_celeba.txt',
            transform=transform,
            max_samples=100  # Start very small
        )
        print(f"✅ Dataset created with {len(dataset)} samples")
        
    except Exception as e:
        print(f"❌ Dataset creation failed: {e}")
        return False
    
    # Step 3: Test dataloader
    print("\n3. Testing dataloader...")
    try:
        dataloader = DataLoader(dataset, batch_size=8, shuffle=False)  # Small batch
        print("✅ DataLoader created")
        
        # Test loading one batch
        for batch in dataloader:
            images, gender, age = batch
            print(f"✅ First batch loaded - Images: {images.shape}, Gender: {gender.shape}")
            break
            
    except Exception as e:
        print(f"❌ DataLoader failed: {e}")
        return False
    
    # Step 4: Test model loading
    print("\n4. Testing model loading...")
    try:
        models = load_models()
        model_name = 'supervised_resnet18'
        model = models[model_name]
        print(f"✅ Model {model_name} loaded")
        
    except Exception as e:
        print(f"❌ Model loading failed: {e}")
        return False
    
    # Step 5: Test feature extractor creation
    print("\n5. Testing feature extractor...")
    try:
        extractor = FeatureExtractor(model, layer_name='avgpool')
        print("✅ FeatureExtractor created")
        
    except Exception as e:
        print(f"❌ FeatureExtractor creation failed: {e}")
        return False
    
    # Step 6: Test feature extraction (this might be the issue)
    print("\n6. Testing feature extraction...")
    try:
        print("Extracting features...")
        features, gender_labels, age_labels = extractor.extract_features(dataloader)
        print(f"✅ Feature extraction completed - Shape: {features.shape}")
        print(f"Gender distribution: {np.bincount(gender_labels)}")
        print(f"Age distribution: {np.bincount(age_labels)}")
        
    except Exception as e:
        print(f"❌ Feature extraction failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Step 7: Test bias detector creation
    print("\n7. Testing bias detector creation...")
    try:
        bias_detector = BiasDetector()
        print("✅ BiasDetector created")
        
    except Exception as e:
        print(f"❌ BiasDetector creation failed: {e}")
        return False
    
    # Step 8: Test linear probe (the suspected culprit)
    print("\n8. Testing linear probe with REAL data...")
    try:
        print("Running linear probe on gender...")
        gender_result = bias_detector.linear_probe_bias(features, gender_labels)
        print(f"✅ Gender linear probe completed: {gender_result['test_accuracy']:.3f}")
        
        print("Running linear probe on age...")
        age_result = bias_detector.linear_probe_bias(features, age_labels)
        print(f"✅ Age linear probe completed: {age_result['test_accuracy']:.3f}")
        
    except Exception as e:
        print(f"❌ Linear probe failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n🎉 ALL STEPS PASSED! The issue might be with larger datasets or specific data.")
    return True

if __name__ == "__main__":
    success = debug_step_by_step()
    
    if success:
        print("\nThe pipeline works with small data. Try increasing max_samples gradually.")
        print("The segfault might happen with larger datasets or specific data points.")
    else:
        print("\nFound the problematic step! Fix the failing step before proceeding.")