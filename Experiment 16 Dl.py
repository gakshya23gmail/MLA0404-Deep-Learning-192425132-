import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
X, y = make_classification(n_samples=200,
                           n_features=2,
                           n_informative=2,
                           n_redundant=0,
                           random_state=42)
model = MLPClassifier(hidden_layer_sizes=(3,3),
                      activation='identity',     
                      learning_rate_init=0.03,  
                      max_iter=1000,
                      random_state=42)
model.fit(X, y)
y_pred = model.predict(X)
accuracy = accuracy_score(y, y_pred)
print("Accuracy:", accuracy)
plt.scatter(X[:,0], X[:,1], c=y, cmap='viridis', edgecolor='k')
plt.title("Neural Network Two-Class Classification")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()
