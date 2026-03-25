import matplotlib.pyplot as plt
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
)
from sklearn.svm import LinearSVC

LABELS = {1: "World", 2: "Sports", 3: "Business", 4: "Sci/Tech"}


class LR_TFIDF:
    def __init__(self):
        self.model = LogisticRegression(max_iter=1000)

    def fit(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        return self.model.predict(X_test)


class SVM_TFIDF:
    def __init__(self):
        self.model = LinearSVC()

    def fit(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        return self.model.predict(X_test)


def train_model(model_type: str, X_train, y_train):
    """
    Train a TF-IDF model with Logistic Regression or SVM.
    """
    if model_type == "logistic":
        model = LR_TFIDF()
    elif model_type == "svm":
        model = SVM_TFIDF()
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    model.fit(X_train, y_train)

    return model


def vectorize_datasets(train_df, dev_df, test_df, text_col: str = "description"):
    """Fit TF-IDF on train and transform dev/test."""
    vectorizer = TfidfVectorizer(max_features=5000, stop_words="english")
    X_train = vectorizer.fit_transform(train_df[text_col])
    X_dev = vectorizer.transform(dev_df[text_col])
    X_test = vectorizer.transform(test_df[text_col])
    return X_train, X_dev, X_test


def extract_labels(train_df, dev_df, test_df, label_col: str = "label"):
    """Extract zero-based labels for train/dev/test."""
    y_train = train_df[label_col].values - 1
    y_dev = dev_df[label_col].values - 1
    y_test = test_df[label_col].values - 1
    return y_train, y_dev, y_test


def error_analysis(test_df: pd.DataFrame, y_test, X_test, model) -> None:
    """Display misclassified examples for TF-IDF models."""
    true_labels_text = [
        LABELS.get(int(i) + 1, str(int(i) + 1)) for i in y_test
    ]

    pred_numeric = model.predict(X_test)
    pred_labels_text = [
        LABELS.get(int(i) + 1, str(int(i) + 1)) for i in pred_numeric
    ]

    df_predictions = pd.DataFrame(
        {
            "text": test_df["description"].values,
            "true_label": true_labels_text,
            "pred_label": pred_labels_text,
        }
    )

    errors = df_predictions[
        df_predictions["true_label"] != df_predictions["pred_label"]
    ]

    for e in errors.head(20).itertuples():
        print(f"Text: {e.text}")
        print(f"True Label: {e.true_label}, Predicted Label: {e.pred_label}")
        print("\n" + "-" * 50 + "\n")


def evaluate_model(model, X_test, y_test, dataset_name: str = "Test"):
    """Evaluate TF-IDF model and return metrics."""
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average="macro")
    conf_matrix = confusion_matrix(y_test, y_pred)

    print("Results:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Macro-F1: {macro_f1:.4f}")
    disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix)

    disp.plot(xticks_rotation="vertical")
    plt.title(
        f"Confusion Matrix: {model.__class__.__name__} on {dataset_name}"
    )
    plt.show()

    return {"accuracy": accuracy, "macro_f1": macro_f1}
