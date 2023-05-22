from pwn import *
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

features = np.genfromtxt('data.csv', delimiter=',', dtype=float)[:, :24]
labels = np.genfromtxt('data.csv', delimiter=',', dtype=str)[:, -1]

training_fetures, testing_features, training_labels, testing_labels = train_test_split(
    features, labels, test_size=0.2, random_state=42)

clf = RandomForestClassifier()
clf.fit(training_fetures, training_labels)

test_score = clf.score(testing_features, testing_labels)

print("Testing accuracy:", test_score)

unique_labels = np.unique(labels)

prediction_labels = clf.predict(testing_features)
cm = confusion_matrix(testing_labels, prediction_labels)
plt.figure(figsize=(7, 7))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=unique_labels, yticklabels=unique_labels)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.tight_layout()
plt.show()
