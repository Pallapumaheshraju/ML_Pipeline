from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
import data_preprocess as dp

# Train and evaluate model
def train_and_evaluate(X, y):
    # Split the dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train model
    model = LogisticRegression()
    model.fit(X_train, y_train)
    print("till here")
    # Predictions
    y_pred = model.predict(X_test)

    # Evaluate model
    accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)

    print(f'Accuracy: {accuracy:.2f}')
    print('Confusion Matrix:')
    print(conf_matrix)

# Main function
def main():
    print("Start")
    iris_file_path = 'Iris.csv'  # Update with your local path
    X, y = dp.main(iris_file_path)
    train_and_evaluate(X, y)
    print("Done")

if __name__ == "__main__":
    print("Hello")
    main()