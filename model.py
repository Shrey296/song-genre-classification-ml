import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay

df = pd.read_csv("Currents.csv")

# Removes rows with missing Genres
df = df.dropna(subset=["Genres"])

# Broad Genre Categorization
def categorize(g):
    g = g.lower()
    if any(k in g for k in ["rap", "hip hop", "trap"]):
        return "Rap/HipHop"
    if any(k in g for k in ["rock", "metal", "punk", "grunge", "garage"]):
        return "Rock"
    if "pop" in g:
        return "Pop"
    if "r&b" in g or "soul" in g:
        return "R&B/Soul"
    if any(k in g for k in ["anime", "j-pop", "j-rock"]):
        return "Anime/JPop"
    if any(k in g for k in ["indie", "lo-fi", "bedroom"]):
        return "Indie"
    return "Other"

df["BroadGenre"] = df["Genres"].apply(categorize)

# 12 Audio Features
feature_cols = [
    "Danceability", "Energy", "Key", "Mode", "Loudness",
    "Speechiness", "Acousticness", "Instrumentalness",
    "Liveness", "Valence", "Tempo", "Time Signature"
]

df = df.dropna(subset=feature_cols)

X = df[feature_cols]
y = df["BroadGenre"]

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Standardization
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Logistic Regression Model
model = LogisticRegression(max_iter=5000)
model.fit(X_train_scaled, y_train)

# Predictions
y_pred = model.predict(X_test_scaled)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Model Accuracy:", accuracy)

print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred, labels=model.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
disp.plot(xticks_rotation=45)
plt.title("Confusion Matrix")
plt.tight_layout()
plt.show()

# Feature Importance
importance = np.mean(np.abs(model.coef_), axis=0)
importance_series = pd.Series(importance, index=feature_cols).sort_values()

plt.figure(figsize=(8,6))
plt.barh(importance_series.index, importance_series.values)
plt.title("Feature Importance (Logistic Regression)")
plt.xlabel("Coefficient Magnitude")
plt.tight_layout()
plt.show()
