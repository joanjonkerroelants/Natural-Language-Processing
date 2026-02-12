from typing import cast

import matplotlib.pyplot as plt
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    confusion_matrix,
)


def get_metrics(model, X_test, y_test, labels):
    y_pred = model.predict(X_test)

    report = classification_report(y_test, y_pred, output_dict=True)
    report = cast(dict, report)

    macro_f1 = report["macro avg"]["f1-score"]
    accuracy = report["accuracy"]
    conf_m = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(
        confusion_matrix=conf_m, display_labels=labels
    )
    disp.plot(xticks_rotation="vertical")
    plt.show()

    return {
        "accuracy": accuracy,
        "f1_score": macro_f1,
        "confusion_matrix": conf_m,
    }
