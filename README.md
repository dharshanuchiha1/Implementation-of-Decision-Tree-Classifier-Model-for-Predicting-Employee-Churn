# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the required libraries for data handling, plotting, and Decision Tree classification.

2.Load the Employee dataset and convert categorical data into numerical format using get_dummies().

3.Split the dataset into input features (X) and target output (y), then divide the data into training and testing sets.

4.Create and train the DecisionTreeClassifier model using the training data, then predict the output for test data.

5.Calculate the accuracy of the model and display the Decision Tree diagram using plot_tree().

## Program:
```python
/*
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by: Dharshan G
RegisterNumber:  212225230054
*/
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score
data = pd.read_csv("Employee.csv")
data = pd.get_dummies(data, drop_first=True)
X = data.iloc[:, :-1]
y = data.iloc[:, -1]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
plt.figure(figsize=(20,10))

plot_tree(
    model,
    feature_names=X.columns,
    filled=True
)

plt.show()
```

## Output:
![decision tree classifier model](sam.png)
<img width="586" height="553" alt="Screenshot 2026-05-13 103522" src="https://github.com/user-attachments/assets/05e522c4-73e9-4a0e-8c6a-b9d63c68232e" />
<img width="1257" height="639" alt="Screenshot 2026-05-13 103611" src="https://github.com/user-attachments/assets/4e3c18a8-1adc-43e5-9cfd-2d8cbea1f83f" />


## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
