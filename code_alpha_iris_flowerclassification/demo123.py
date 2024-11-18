from sklearn.datasets import load_iris
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
# Load dataset
iris = load_iris()

# Convert to pandas DataFrame
iris_df = pd.DataFrame(data=iris.data, columns=iris.feature_names)

# Add target column (species)
iris_df['species'] = iris.target
iris_df['species'] = iris_df['species'].map({0: 'setosa', 1: 'versicolor', 2: 'virginica'})

# Preview the first few rows
print(iris_df.head())
X = iris_df.drop('species', axis=1)  # Features (measurement data)
y = iris_df['species']  # Target (iris species)
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
from sklearn.ensemble import RandomForestClassifier

# Initialize the model
model = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the model
model.fit(X_train, y_train)
from sklearn.neighbors import KNeighborsClassifier

# Initialize the model
knn = KNeighborsClassifier(n_neighbors=5)

# Train the model
knn.fit(X_train, y_train)
# Evaluate the model on the test set
accuracy = model.score(X_test, y_test)
print(f"Accuracy of the model: {accuracy * 100:.2f}%")
accuracy = knn.score(X_test, y_test)
print(f"Accuracy of KNN model: {accuracy * 100:.2f}%")
# Example: Predict species for a new flower with specific measurements
new_data = [[5.1, 3.5, 1.4, 0.2]]  # Example: measurements for sepal length, sepal width, petal length, petal width
predicted_species = model.predict(new_data)

print(f"Predicted species: {predicted_species[0]}")
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Make predictions
y_pred = model.predict(X_test)

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

# Plot the confusion matrix
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=iris.target_names, yticklabels=iris.target_names)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()
