from typing import cast

from sklearn.metrics import classification_report, confusion_matrix


def get_metrics(model, X_test, y_test):
    y_pred = model.predict(X_test)

    report = classification_report(y_test, y_pred, output_dict=True)
    report = cast(dict, report)

    macro_f1 = report["macro avg"]["f1-score"]
    accuracy = report["accuracy"]
    conf_m = confusion_matrix(y_test, y_pred)

    return {
        "accuracy": accuracy,
        "f1_score": macro_f1,
        "confusion_matrix": conf_m,
    }
