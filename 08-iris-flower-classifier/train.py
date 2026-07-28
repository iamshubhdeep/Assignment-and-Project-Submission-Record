import joblib
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier

def train_and_save():
    # 1. Grab the dataset (features and labels)
    X, y = load_iris(return_X_y=True)
    
    # 2. Initialize and fit the random forest classifier
    print("Training the model...")
    model = RandomForestClassifier(random_state=42)
    model.fit(X, y)
    
    # 3. Export the trained model to disk
    # Using 'iris_model.pkl' (with an underscore) is safer for system paths
    joblib.dump(model, "iris_model.pkl")
    print("Model saved successfully!")

if __name__ == "__main__":
    train_and_save()