from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC


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
