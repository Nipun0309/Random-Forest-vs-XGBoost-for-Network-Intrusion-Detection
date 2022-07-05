from sklearn import metrics
from sklearn.preprocessing import LabelBinarizer


def multiclass_roc_auc_score(y_test, y_pred, average="macro"):
    lb = LabelBinarizer()
    lb.fit(y_test)
    y_test = lb.transform(y_test)
    y_pred = lb.transform(y_pred)
    x = metrics.roc_auc_score(y_test, y_pred, average=average)
    return x
