from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn import datasets

# 1. Load a sample dataset (Iris flowers)
iris = datasets.load_iris()
X, y = iris.data, iris.target

# 2. Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 3. Initialize and train the model
model = GaussianNB()
model.fit(X_train, y_train)

# 4. Make predictions and check accuracy
accuracy = model.score(X_test, y_test)
print(f"Model Accuracy: {accuracy * 100:.2f}%")

# 5. Predict a single new flower
sample = [[5.1, 3.5, 1.4, 0.2]] 
print(f"Prediction for sample: {iris.target_names[model.predict(sample)][0]}")