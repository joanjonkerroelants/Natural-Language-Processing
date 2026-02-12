from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC


class LR_TFIDF:
    def __init__(self):
        self.model = LogisticRegression()

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
