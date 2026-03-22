import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from xgboost import XGBClassifier

df = pd.read_csv("cpu_utilization_asg_misconfiguration.csv")
data = df["value"].values

threshold = 80

incident_now = []

for value in data:
    if value > threshold:
        incident_now.append(1)
    else:
        incident_now.append(0)

incident_now = np.array(incident_now)

W = 20
H = 5

X = []
y = []

last_start = len(data) - W - H + 1

for i in range(last_start):
    past_window = data[i:i + W]
    future_window = incident_now[i + W:i + W + H]

    label = 0

    for item in future_window:
        if item == 1:
            label = 1
            break

    X.append(past_window)
    y.append(label)

X = np.array(X)
y = np.array(y)

split_index = int(len(X) * 0.7)
X_train = X[:split_index]
X_test = X[split_index:]
y_train = y[:split_index]
y_test = y[split_index:]

print("Number of original time steps:", len(data))
print("Threshold for incident:", threshold)
print("Number of incident time steps:", incident_now.sum())
print("Total sliding-window samples:", len(X))


lr_model = LogisticRegression(max_iter=1000)
lr_model.fit(X_train, y_train)

rf_model = RandomForestClassifier(n_estimators=100, random_state=1)
rf_model.fit(X_train, y_train)

xgb_model = XGBClassifier(
    n_estimators=100,
    max_depth=3,
    learning_rate=0.1,
    eval_metric="logloss",
    random_state=1
)
xgb_model.fit(X_train, y_train)

lr_pred = lr_model.predict(X_test)
rf_pred = rf_model.predict(X_test)
xgb_pred = xgb_model.predict(X_test)

lr_accuracy = accuracy_score(y_test, lr_pred)
lr_precision = precision_score(y_test, lr_pred, zero_division=0)
lr_recall = recall_score(y_test, lr_pred, zero_division=0)
lr_f1 = f1_score(y_test, lr_pred, zero_division=0)

print("\nLogistic Regression")
print("Accuracy :", round(lr_accuracy, 3))
print("Precision:", round(lr_precision, 3))
print("Recall   :", round(lr_recall, 3))
print("F1-score :", round(lr_f1, 3))

rf_accuracy = accuracy_score(y_test, rf_pred)
rf_precision = precision_score(y_test, rf_pred, zero_division=0)
rf_recall = recall_score(y_test, rf_pred, zero_division=0)
rf_f1 = f1_score(y_test, rf_pred, zero_division=0)

print("\nRandom Forest")
print("Accuracy :", round(rf_accuracy, 3))
print("Precision:", round(rf_precision, 3))
print("Recall   :", round(rf_recall, 3))
print("F1-score :", round(rf_f1, 3))

xgb_accuracy = accuracy_score(y_test, xgb_pred)
xgb_precision = precision_score(y_test, xgb_pred, zero_division=0)
xgb_recall = recall_score(y_test, xgb_pred, zero_division=0)
xgb_f1 = f1_score(y_test, xgb_pred, zero_division=0)

print("\nXGBoost")
print("Accuracy :", round(xgb_accuracy, 3))
print("Precision:", round(xgb_precision, 3))
print("Recall   :", round(xgb_recall, 3))
print("F1-score :", round(xgb_f1, 3))