import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.spatial.distance import cosine
from sentence_transformers import SentenceTransformer, util
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import joblib


def calculate_semantic_similarity(reference, student):
    # ...
    model = SentenceTransformer("bert-base-nli-mean-tokens")

    # Encode reference and student answers
    reference_embedding = model.encode(reference)
    student_embedding = model.encode(student)

    # Compute cosine similarity between embeddings
    semantic_similarity_score = util.pytorch_cos_sim(
        reference_embedding, student_embedding
    ).item()

    return semantic_similarity_score


def calculate_lexical_similarity(reference, student):
    # ...
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([reference, student])

    # Compute cosine similarity between TF-IDF vectors
    lexical_similarity_score = 1 - cosine(
        tfidf_matrix[0].toarray().flatten(), tfidf_matrix[1].toarray().flatten()
    )

    return lexical_similarity_score


# Function to extract features from a pair of reference and student answers
def extract_features(reference_answer, student_answer):
    # Calculate semantic similarity
    semantic_similarity_score = calculate_semantic_similarity(
        reference_answer, student_answer
    )

    # Calculate lexical similarity
    lexical_similarity_score = calculate_lexical_similarity(
        reference_answer, student_answer
    )

    # Combine features into a single vector
    features = [semantic_similarity_score, lexical_similarity_score]

    return features


# Function to train a Random Forest Regressor and save the model
def train_random_forest_and_save_model(X, y, model_save_path):
    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Create a Random Forest Regressor
    model = RandomForestRegressor()

    # Train the model
    model.fit(X_train, y_train)

    # Save the trained model
    joblib.dump(model, model_save_path)

    # Evaluate on the test set
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    print("Mean Squared Error on Test Set:", mse)

    return model


def main():
    # Load your dataset
    data = pd.read_csv(
        "C:\\Users\\gokul\\OneDrive\\Desktop\\MainProject\\1-3-24\\dataset.csv"
    )
    print(data.columns)
    # Extract features and labels from the dataset
    X = []
    y = []
    for index, row in data.iterrows():
        reference_answer, student_answer = (
            row["reference answer"],
            row[" student answer"],
        )
        features = extract_features(reference_answer, student_answer)
        score = row["score"]
        X.append(features)
        y.append(score)

    # Train a Random Forest Regressor and save the model
    model_save_path = "random_forest_model.pkl"
    model = train_random_forest_and_save_model(X, y, model_save_path)

    print("Model saved successfully.")


if __name__ == "__main__":
    main()
