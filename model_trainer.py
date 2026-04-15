from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import joblib

MODEL_MAP = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest": RandomForestClassifier(),
    "KNN": KNeighborsClassifier()
}

def build_pipeline(model_name):
    return Pipeline([
        ("scaler", StandardScaler()),
        ("model", MODEL_MAP[model_name])
    ])

def train_with_kfold(X, y, model_name):
    pipe = build_pipeline(model_name)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    results = cross_validate(
        pipe, X, y,
        cv=cv,
        scoring=["accuracy", "precision", "recall", "f1"]
    )

    pipe.fit(X, y)
    return pipe, results

def save_model(model):
    joblib.dump(model, "models/model.pkl")