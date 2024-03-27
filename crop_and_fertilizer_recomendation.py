import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

def load_data(filename):
    """Loads data from a CSV file containing N, P, K, and PH properties.

    Args:
        filename (str): The path to the CSV file.

    Returns:
        pandas.DataFrame: The loaded data with N, P, K, PH, fertilizer, and croptype columns.
    """

    data = pd.read_csv(filename)
    return data

def train_model(data, test_size=0.2, random_state=42):
    """Trains a Random Forest model on the provided data.

    Args:
        data (pandas.DataFrame): The data to train the model on.
        test_size (float, optional): The proportion of data to use for testing. Defaults to 0.2.
        random_state (int, optional): The random seed for splitting data. Defaults to 42.

    Returns:
        tuple: A tuple containing the trained model and the split data (X_train, X_test, y_train, y_test).
    """

    # Extract features (N, P, K, PH) and target variables (fertilizer, croptype)
    features = data.drop(columns=['fertilizer', 'croptype'])
    target_fertilizer = data['fertilizer']
    target_croptype = data['croptype']

    # Split data into training and testing sets
    X_train, X_test, y_train_fertilizer, y_test_fertilizer = train_test_split(
        features, target_fertilizer, test_size=test_size, random_state=random_state
    )
    X_train, X_test, y_train_croptype, y_test_croptype = train_test_split(
        X_train, target_croptype, test_size=test_size/2, random_state=random_state
    )  # Use smaller test size for croptype to avoid overfitting

    # Train separate Random Forest models for fertilizer and croptype
    model_fertilizer = RandomForestClassifier()
    model_fertilizer.fit(X_train, y_train_fertilizer)

    model_croptype = RandomForestClassifier()
    model_croptype.fit(X_train, y_train_croptype)

    return (model_fertilizer, model_croptype), (X_train, X_test, y_train_fertilizer, y_test_fertilizer, X_train, X_test, y_train_croptype, y_test_croptype)

def predict(model, features):
    """Makes predictions using a trained model.

    Args:
        model (tuple): The trained model (fertilizer, croptype models)
        features (pandas.DataFrame or list): The features (N, P, K, PH) for prediction.

    Returns:
        tuple: A tuple containing the predicted fertilizer type and crop type.
    """

    model_fertilizer, model_croptype = model

    if isinstance(features, pd.DataFrame):
        features = features.values.flatten()  # Convert DataFrame to a 1D array

    fertilizer_prediction = model_fertilizer.predict([features])[0]
    croptype_prediction = model_croptype.predict([features])[0]

    return fertilizer_prediction, croptype_prediction

def get_user_input():
    """Prompts the user for N, P, K, and PH values.

    Returns:
        list: A list containing the user-provided N, P, K, and PH values.
    """

    while True:
        try:
            N = float(input("Enter N value: "))
            P = float(input("Enter P value: "))
            K = float(input("Enter K value: "))
            PH = float(input("Enter PH value: "))
            return [N, P, K, PH]
        except ValueError:
            print("Invalid input. Please enter numbers only.")

def main():
    """Loads data, trains models, makes predictions, and handles user input."""

    data_file = "your_data.csv"  # Replace with the path to your CSV file
    model, (X_train, X_test, y_train_fertilizer, y_test_fertilizer,X_train, X_test, y_train_croptype, y_test_croptype) = train_model(load_data(data_file))
