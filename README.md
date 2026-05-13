# SGD-Classifier
## AIM:
To write a program to predict the type of species of the Iris flower using the SGD Classifier.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Load the Iris dataset and split it into training and testing data.
2. Standardize the feature values using StandardScaler.
3. Train the SGDClassifier model using the training dataset.
4. Predict the test data results and evaluate the model using accuracy and classification report.
 

## Program:
```

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report


iris = load_iris()
X = iris.data
y = iris.target


scaler = StandardScaler()
X = scaler.fit_transform(X)


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
model = SGDClassifier(max_iter=1000, random_state=42)
model.fit(X_train, y_train)


y_pred = model.predict(X_test)


print("Accuracy:", accuracy_score(y_test, y_pred))


print("\nClassification Report:\n", classification_report(y_test, y_pred))


/*
Program to implement the prediction of iris species using SGD Classifier.
Developed by: MOHAMMED AMEER F
RegisterNumber:  212225040248
*/
```

## Output:

<img width="661" height="275" alt="WhatsApp Image 2026-05-13 at 08 49 53" src="https://github.com/user-attachments/assets/32433c2d-f3c1-4b03-9afe-de6c1cbecba3" />


## Result:
Thus, the program to implement the prediction of the Iris species using SGD Classifier is written and verified using Python programming.
