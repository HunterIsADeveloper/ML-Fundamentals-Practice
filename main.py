import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV, KFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier

# Want to predict the species of penguin
palmer_penguins_ds = pd.read_csv('palmerpenguins_extended.csv')

param_grid = {
    'n_estimators': [100, 200, 300], # Number of trees in the forest
    'max_depth': [None, 10, 20, 30], # Maximum depth of the tree
    'max_features': ['auto', 'sqrt', 'log2'], # Number of features to consider at every split
    'min_samples_split': [2, 5, 10], # Minimum number of samples required to split an internal node
    'min_samples_leaf': [1, 2, 4], # Minimum number of samples required to be at a leaf node
    'bootstrap': [True, False] # Whether bootstrap samples are used when building trees
}

# Drop unneeded columns
palmer_penguins_ds = palmer_penguins_ds.drop(columns=['sex', "life_stage", "health_metrics"])

# Encode columns which are still categorical
scaler = StandardScaler()
label_encoder = LabelEncoder()
palmer_penguins_ds['island'] = label_encoder.fit_transform(palmer_penguins_ds['island'])
palmer_penguins_ds['diet'] = label_encoder.fit_transform(palmer_penguins_ds['diet'])

x = palmer_penguins_ds.loc[:, palmer_penguins_ds.columns != 'species']
y = palmer_penguins_ds['species']

# Test Sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Validation Sets
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.25, random_state=42)

# Encode target variable
y_train = label_encoder.fit_transform(y_train)
y_val = label_encoder.transform(y_val)
y_test = label_encoder.transform(y_test)

# Scale features
x_train = scaler.fit_transform(x_train)
x_val = scaler.transform(x_val)
x_test = scaler.transform(x_test)

rf_model = RandomForestClassifier(random_state=42)
grid_search = GridSearchCV(rf_model, param_grid, cv=10, scoring='accuracy', n_jobs=-1, verbose=2)

grid_search.fit(x_train, y_train)

best_rf = grid_search.best_estimator_

print("Best Hyperparameters:", grid_search.best_params_)
print(f"Best Cross-Validation Accuracy: {grid_search.best_score_:.4f}")

y_pred = best_rf.predict(x_test)

test_accuracy = accuracy_score(y_test, y_pred)
print(f"Test Accuracy: {test_accuracy:.4f}")
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

feature_importances = best_rf.feature_importances_
feature_names = x.columns
sorted_indices = np.argsort(feature_importances)[::-1]
labels = np.array(feature_names)[sorted_indices].tolist()

plt.figure(figsize=(10, 6))
plt.bar(range(len(feature_importances)), feature_importances[sorted_indices], align='center')
plt.xticks(range(len(feature_importances)), labels, rotation=90)
plt.title("Random Forest Feature Importances")
plt.xlabel("Feature")
plt.ylabel("Importance")
plt.tight_layout()
plt.show()