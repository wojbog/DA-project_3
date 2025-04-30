from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from yellowbrick.classifier import ClassificationReport, ROCAUC
import numpy as np
import pandas as pd
from explainerdashboard import ClassifierExplainer, ExplainerDashboard

# train split
from sklearn.model_selection import train_test_split

dataset_file = "./data/car evaluation.csv"
df = pd.read_csv(dataset_file, header=None)
df.columns = [
        "dupa1",
        "dupa2",
        "dupa3",
        "dupa4",
        "dupa5",
        "dupa6",
        "category"
]

X = df.drop(columns=["category"])
y = (df["category"] - 1) // 2


train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.2, random_state=0xC0FFEE)

model = LogisticRegression()
model.fit(train_X, train_y)

visualizers = [
    ClassificationReport(model, classes=[0, 1]),
    ROCAUC(model, classes=[0, 1], size=(1080, 720)),
]

explainer = ClassifierExplainer(model, test_X, test_y)
# ExplainerDashboard(explainer).run()
# for v in visualizers:
#     v.fit(train_X,train_y)
#     v.score(test_X, test_y)
#     v.show()

# # # y_hat = model.predict(data)
# # # y_hat_proba = model.predict_proba(data)
# # # print(y_hat)
# # # accuracy = accuracy_score(labels, y_hat)

# # # f1 = f1_score(labels, y_hat)
# # # print((labels == y_hat).sum() / len(labels))
# # # print(f"Accuracy: {accuracy}")
# # # print(f"F1 Score: {f1}")
# # # print(f"ROC AUC Score: {roc_auc_score(labels, y_hat_proba[:, 1])}")





