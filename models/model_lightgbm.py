from lightgbm import LGBMClassifier

def build_lightgbm_model(scale_pos_weight: float):
    model = LGBMClassifier(
        n_estimators=300,
        max_depth=-1,          # let LightGBM choose
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        class_weight=None,     # we use scale_pos_weight instead
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        n_jobs=-1,
    )
    return model