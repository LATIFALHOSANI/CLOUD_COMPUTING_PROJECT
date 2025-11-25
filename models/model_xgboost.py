from xgboost import XGBClassifier

def build_xgboost_model(scale_pos_weight: float) -> XGBClassifier:
    model = XGBClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=scale_pos_weight,
    random_state=42,
    n_jobs=-1,
    )
    return model