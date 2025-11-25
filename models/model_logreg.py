from sklearn.linear_model import LogisticRegression

def build_logreg_model():
    model = LogisticRegression(
        penalty="l2",
        C=0.1,                  # stronger regularization to avoid overfitting
        class_weight="balanced",
        max_iter=500,
        solver="lbfgs",
        n_jobs=-1,
    )
    return model
