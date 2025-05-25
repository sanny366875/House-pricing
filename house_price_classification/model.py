import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from collections import Counter

# Load dataset
df = pd.read_csv('house_data.csv')

# Encode categorical variables
label_enc_location = LabelEncoder()
df['Location'] = label_enc_location.fit_transform(df['Location'])

label_enc_price = LabelEncoder()
df['PriceCategory'] = label_enc_price.fit_transform(df['PriceCategory'])

# Features and target
X = df[['Area', 'Bedrooms', 'Bathrooms', 'Location']]
y = df['PriceCategory']

# Create a copy to avoid SettingWithCopyWarning
X_scaled = X.copy()

# Standardize numerical columns
scaler = StandardScaler()
X_scaled[['Area', 'Bedrooms', 'Bathrooms']] = scaler.fit_transform(X_scaled[['Area', 'Bedrooms', 'Bathrooms']])

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.3, random_state=42, stratify=y
)

# Train Logistic Regression with class balancing
model = LogisticRegression(max_iter=5000, class_weight='balanced')
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluation
print("Accuracy on test set:", accuracy_score(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(
    y_test, y_pred,
    labels=np.unique(y),
    target_names=label_enc_price.classes_,
    zero_division=1
))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=label_enc_price.classes_,
            yticklabels=label_enc_price.classes_)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# Safe cross-validation: determine minimum samples per class
class_counts = Counter(y)
min_class_size = min(class_counts.values())
safe_cv = min(5, min_class_size)

# Cross-validation
cv_scores = cross_val_score(model, X_scaled, y, cv=safe_cv, scoring='accuracy')
print(f"\nCross-validation (cv={safe_cv}) scores:", cv_scores)
print("Average cross-validation accuracy:", cv_scores.mean())
