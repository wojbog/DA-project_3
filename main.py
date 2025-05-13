from sklearn.linear_model import LogisticRegression
import shap
# xkboost
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from yellowbrick.classifier import ClassificationReport, ROCAUC
import numpy as np
import pandas as pd

# train split
from sklearn.model_selection import train_test_split

dataset_file = "./data/car evaluation.csv"
df = pd.read_csv(dataset_file, header=None)
df.columns = [
        "buying_price"
        "maintanace_price",
        "number_of_doors",
        "capacity",
        "boot_size",
        "safety",
        "evaluation"
]

X = df.drop(columns=["category"])
y = (df["category"] - 1) // 2


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0xC0FFEE)

model = LogisticRegression()
model.fit(X_train, y_train)

classes = ["Category 1", "Category 2"]

visualizers = [
    ClassificationReport(model, classes=classes, recall=False),
    ROCAUC(model, classes=classes, size=(1080, 720)),
]

# explainer = ClassifierExplainer(model, test_X, test_y)
# ExplainerDashboard(explainer).run()
for v in visualizers:
    v.fit(X_train, y_train)
    v.score(X_test, y_test)
    v.show()

# print coefficients
coefs = model.coef_[0]


explainer = shap.Explainer(model, X_train)
shap_values = explainer(X_test[:3])
shap.plots.waterfall(shap_values[0])



