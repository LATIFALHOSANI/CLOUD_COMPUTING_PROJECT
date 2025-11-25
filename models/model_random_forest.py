from sklearn.ensemble import RandomForestClassifier

def build_random_forest_model():
    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_split=4,
        min_samples_leaf=2,
        class_weight="balanced",   # handles imbalance
        random_state=42,
        n_jobs=-1,
    )
    return model