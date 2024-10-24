import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Load dataset from local storage
def load_data(file_path):
    data = pd.read_csv(file_path)
    print("Data Loaded Successfully...")
    return data

# Preprocess the data
def preprocess_data(data):
    # Check for missing values
    if data.isnull().sum().sum() > 0:
        data = data.dropna()  # Remove missing values

    # Features and target variable
    X = data.drop('Species', axis=1)
    y = data['Species']

    # Feature scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y

# Main function to load and preprocess data
def main(file_path):
    data = load_data(file_path)
    X, y = preprocess_data(data)
    print("Preprocessing done!")
    return X, y

if __name__ == "__main__":
    iris_file_path = 'Iris.csv'  # Update with your local path
    X, y = main(iris_file_path)
