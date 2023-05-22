from pwn import *
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import numpy as np

features = np.genfromtxt('data.csv', delimiter=',', dtype=float)[:, :24]
labels = np.genfromtxt('data.csv', delimiter=',', dtype=str)[:, -1]

training_fetures, testing_fetures, training_labels, testing_labels = train_test_split(
    features, labels, test_size=0.2, random_state=42)

clf = RandomForestClassifier()
clf.fit(training_fetures, training_labels)

test_score = clf.score(testing_fetures, testing_labels)

print("Testing accuracy:", test_score)

importances = clf.feature_importances_

# Displaying feature importance for each column
for i, importance in enumerate(importances):
    print(f"Column #{i+1}: Feature Importance: {importance}")
