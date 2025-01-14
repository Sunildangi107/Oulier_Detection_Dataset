import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import shuffle
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

def generate_shadow_features(X):
    """Generate shadow features by shuffling the original features."""
    shadow_features = X.apply(shuffle, axis=0)
    shadow_features.columns = [f'shadow_{col}' for col in X.columns]
    return shadow_features

def calculate_z_scores(X, y):
    """Calculate Z-scores using feature importance from a random forest model."""
    model = RandomForestClassifier(random_state=42, n_estimators=100)
    model.fit(X, y)
    feature_importances = model.feature_importances_
    z_scores = (feature_importances - np.mean(feature_importances)) / np.std(feature_importances)
    return z_scores

def feature_selection_method(X, y, max_iterations):
    """Feature selection method to remove irrelevant attributes."""
    selected_features = list(X.columns)
    original_features = X.copy()

    for iteration in range(max_iterations):
        print(f"Iteration {iteration + 1}:")

        # Generate shadow features
        shadow_features = generate_shadow_features(original_features)
        combined_features = pd.concat([original_features, shadow_features], axis=1)

        # Calculate Z-scores for original and shadow features
        z_scores = calculate_z_scores(combined_features, y)
        original_z_scores = z_scores[:len(original_features.columns)]
        shadow_z_scores = z_scores[len(original_features.columns):]

        # Determine threshold for feature removal
        max_shadow_score = max(shadow_z_scores)

        # Remove features with Z-score <= max_shadow_score
        to_remove = [feature for idx, feature in enumerate(selected_features)
                     if original_z_scores[idx] <= max_shadow_score]

        print(f"Removing features: {to_remove}")
        selected_features = [feature for feature in selected_features if feature not in to_remove]

        # Update the dataset with remaining features
        original_features = original_features[selected_features]

        if not to_remove:
            break  # Stop if no features are removed

    print("Final selected features:", selected_features)

    # Compute average gain of selected features (optional, based on feature importance)
    final_model = RandomForestClassifier(random_state=42, n_estimators=100)
    final_model.fit(original_features, y)
    avg_gain = np.mean(final_model.feature_importances_)
    print(f"Average gain of selected features: {avg_gain}")

    return selected_features

# Example Usage
if __name__ == "__main__":
    # Example Dataset
    from sklearn.datasets import make_classification

    X, y = make_classification(n_samples=1000, n_features=20, n_informative=15, 
                               n_redundant=5, random_state=42)
    X = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(X.shape[1])])

    max_iterations = 5
    selected_features = feature_selection_method(X, y, max_iterations)