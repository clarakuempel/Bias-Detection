import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import silhouette_score
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA

class BiasDetector:
    def __init__(self):
        pass
    
    def linear_probe_bias(self, features, protected_attr, test_size=0.3, random_state=42):
        """
        Measure bias using linear probe accuracy
        Higher accuracy = more bias encoded
        """
        X_train, X_test, y_train, y_test = train_test_split(
            features, protected_attr, test_size=test_size, random_state=random_state
        )
        
        clf = LogisticRegression(random_state=random_state, max_iter=1000)
        clf.fit(X_train, y_train)
        
        train_acc = clf.score(X_train, y_train)
        test_acc = clf.score(X_test, y_test)
        
        return {
            'train_accuracy': train_acc,
            'test_accuracy': test_acc,
            'bias_score': test_acc  # Higher = more bias
        }
    
    def geometric_bias(self, features, protected_attr):
        """
        Measure bias using silhouette score
        Higher silhouette = more separated groups = more bias
        """
        if len(np.unique(protected_attr)) < 2:
            return {'silhouette_score': 0.0}
        
        # For small datasets, use all data
        if len(features) <= 1000:
            features_sample = features
            attr_sample = protected_attr
        else:
            # Sample subset if too large (silhouette_score can be slow)
            indices = np.random.choice(len(features), 1000, replace=False)
            features_sample = features[indices]
            attr_sample = protected_attr[indices]
        
        try:
            sil_score = silhouette_score(features_sample, attr_sample)
        except Exception as e:
            print(f"Error computing silhouette score: {e}")
            sil_score = 0.0
        
        return {'silhouette_score': sil_score}
    
    def subspace_bias(self, features, protected_attr):
        """
        Measure bias using PCA - how much variance explained by protected attribute
        """
        # Apply PCA
        pca = PCA(n_components=min(50, features.shape[1]))
        features_pca = pca.fit_transform(features)
        
        # Linear probe on PCA features
        probe_result = self.linear_probe_bias(features_pca, protected_attr)
        
        return {
            'pca_bias_score': probe_result['test_accuracy'],
            'variance_explained': np.sum(pca.explained_variance_ratio_[:10])  # Top 10 components
        }