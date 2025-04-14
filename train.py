import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# Example: X is an array of feature vectors, y is the corresponding labels.
# You must create these from your 500 samples.
X = np.load('features.npy')  # shape (500, num_features)
y = np.load('labels.npy')    # binary labels: 1 if overlap, 0 otherwise

# Split into training and testing sets.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a classifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Evaluate
y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred))
