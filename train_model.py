from pathlib import Path

import pandas as pd
from sklearn.metrics import classification_report, roc_auc_score, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import joblib

from credit_score import add_credit_score_column

_REPO_ROOT = Path(__file__).resolve().parent
_DATA_DIR = _REPO_ROOT / "data"
_MODEL_DIR = _REPO_ROOT / "model"
_DEFAULT_DATASET = _DATA_DIR / "ml_ready_dataset.csv"
_DEFAULT_MODEL = _MODEL_DIR / "credit_approval_model.joblib"
_DEFAULT_TEST_PRED = _DATA_DIR / "test_predictions.csv"

LEAKY_CREDIT_HISTORY_COLS = [
    # These are computed from the same credit STATUS history used to define TARGET in cleaning.py,
    # so including them makes the task trivial (label leakage) and can produce ~100% accuracy.
    "months_on_record",
    "max_dpd",
    "avg_dpd",
    "num_late_payments",
]


def load_data(path: str | Path | None = None, drop_leaky_cols: bool = True):
    path = Path(path) if path is not None else _DEFAULT_DATASET
    df = pd.read_csv(path)
    df = add_credit_score_column(df)
    drop_cols = ["TARGET", "ID"]
    if drop_leaky_cols:
        drop_cols.extend([c for c in LEAKY_CREDIT_HISTORY_COLS if c in df.columns])
    X = df.drop(columns=drop_cols)
    y = df["TARGET"]
    return X, y


def build_preprocessor(X: pd.DataFrame):
    # In this dataset, all non-ID/TARGET columns are already numeric-encoded
    numeric_features = list(X.columns)
    numeric_transformer = Pipeline(steps=[("scaler", StandardScaler())])

    preprocessor = ColumnTransformer(
        transformers=[("num", numeric_transformer, numeric_features)]
    )
    return preprocessor


def build_models(preprocessor):
    smote = SMOTE(random_state=42)

    log_reg = ImbPipeline(
        steps=[
            ("preprocess", preprocessor),
            ("smote", smote),
            (
                "clf",
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=1000,
                    solver="lbfgs",
                ),
            ),
        ]
    )

    rf = ImbPipeline(
        steps=[
            ("preprocess", preprocessor),
            ("smote", smote),
            (
                "clf",
                RandomForestClassifier(
                    n_estimators=300,
                    max_depth=None,
                    min_samples_split=4,
                    min_samples_leaf=2,
                    n_jobs=-1,
                    class_weight="balanced_subsample",
                    random_state=42,
                ),
            ),
        ]
    )

    return {"log_reg": log_reg, "random_forest": rf}


def train_and_evaluate():
    X, y = load_data(drop_leaky_cols=True)
    print(f"Training with {X.shape[1]} features (leaky credit-history columns dropped).")
    # Deterministic split by row order:
    # first 75% rows -> train, last 25% rows -> test
    split_idx = int(len(X) * 0.75)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    preprocessor = build_preprocessor(X_train)
    models = build_models(preprocessor)

    best_model_name = None
    best_auc = -1.0
    best_model = None
    best_y_pred = None
    best_y_proba = None

    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(X_train, y_train)
        y_proba = model.predict_proba(X_test)[:, 1]
        y_pred = model.predict(X_test)

        auc = roc_auc_score(y_test, y_proba)
        acc = accuracy_score(y_test, y_pred)
        print(f"{name} ROC AUC: {auc:.4f}")
        print(f"{name} Accuracy: {acc:.4f}")
        print(classification_report(y_test, y_pred, digits=4))

        if auc > best_auc:
            best_auc = auc
            best_model_name = name
            best_model = model
            best_y_pred = y_pred
            best_y_proba = y_proba

    print(f"\nBest model: {best_model_name} with ROC AUC={best_auc:.4f}")
    best_acc = accuracy_score(y_test, best_y_pred)
    print(f"Best model Accuracy (compare 0/1 PRED to TARGET): {best_acc:.4f}")

    _MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_model, _DEFAULT_MODEL)
    print(f"Saved best model to {_DEFAULT_MODEL}")

    pd.DataFrame(
        {
            "TARGET": y_test.reset_index(drop=True),
            "PRED": pd.Series(best_y_pred).reset_index(drop=True),
            "PROB_UNSAFE": pd.Series(best_y_proba).reset_index(drop=True),
        }
    ).to_csv(_DEFAULT_TEST_PRED, index=False)
    print(f"Saved test predictions to {_DEFAULT_TEST_PRED}")


def _blend_risk(prob_unsafe: float, credit_score: int, weight_credit_score: float = 0.65) -> float:
    """
    Blend model risk with credit score. Higher credit score -> lower blended risk.
    weight_credit_score is the share of the decision driven by the credit score.
    """
    credit_norm = (credit_score - 300) / 550  # 300-850 -> 0-1
    credit_norm = max(0.0, min(1.0, credit_norm))
    return (weight_credit_score * (1 - credit_norm)) + ((1 - weight_credit_score) * prob_unsafe)


def predict_single(
    applicant_features: dict,
    model_path: str | Path | None = None,
):
    path = Path(model_path) if model_path is not None else _DEFAULT_MODEL
    model = joblib.load(path)
    # Ensure the input dict has all feature columns except ID and TARGET
    from credit_score import compute_credit_score

    credit_score = compute_credit_score(applicant_features)
    features_with_score = dict(applicant_features)
    features_with_score["CREDIT_SCORE"] = credit_score
    X = pd.DataFrame([features_with_score])
    proba_unsafe = model.predict_proba(X)[:, 1]
    prediction = model.predict(X)[0]
    blended_prob_unsafe = _blend_risk(float(proba_unsafe[0]), credit_score)
    # Assuming TARGET=1 means unsafe / should reject
    decision = "reject" if blended_prob_unsafe >= 0.4 else "approve"
    return {
        "credit_score": credit_score,
        "prob_unsafe": float(proba_unsafe[0]),
        "prob_unsafe_blended": float(blended_prob_unsafe),
        "prediction": int(prediction),
        "decision": decision,
    }


if __name__ == "__main__":
    train_and_evaluate()
