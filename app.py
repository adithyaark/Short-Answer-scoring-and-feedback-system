import streamlit as st
import os
import pandas as pd
import tempfile
import fitz
import pytesseract
from PIL import Image
from sklearn.ensemble import RandomForestRegressor
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.spatial.distance import cosine
from sentence_transformers import SentenceTransformer, util
import feedback

# Define custom CSS styles to increase text size
custom_css = """
    <style>
    body {
        font-size: 25px; /* Adjust the base font size as needed */
    }
    .stMarkdown p {
        font-size: 25px !important; /* Increase the font size of Markdown paragraphs */
    }
    .stMarkdown ul {
        font-size: 25px !important; /* Increase the font size of Markdown lists */
    }
    .stText {
        font-size: 25px !important; /* Increase the font size of regular text */
    }
    .stButton {
        font-size: 25px !important; /* Increase the font size of buttons */
    }
    .stTextInput>div>div>input {
        font-size: 25px !important; /* Increase the font size of text input fields */
    }
    .stTable {
        width: 100%; /* Set the width of the table to 100% */
    }

    </style>
"""

# Apply custom CSS styles to the Streamlit app
st.markdown(custom_css, unsafe_allow_html=True)


# Function to extract features from a pair of reference and student answers
def extract_text_from_pdf(pdf_path):
    try:
        pdf_document = fitz.open(pdf_path)
        text = ""
        for page_number in range(pdf_document.page_count):
            page = pdf_document[page_number]
            text += page.get_text()
        return text
    except Exception as e:
        st.error(f"Error: {e}")
        return ""


# Function to extract text from image
def extract_text_from_image(image_path):
    try:
        text = pytesseract.image_to_string(Image.open(image_path), lang="eng")
        return text
    except Exception as e:
        st.error(f"Error: {e}")
        return ""


# Function to split reference and student answers
def split_reference_and_student_answers(text):
    split_point = text.find("Student:")
    if split_point != -1:
        reference_answer = text[:split_point]
        student_answer = text[split_point + len("Student:") :].strip()
    else:
        reference_answer = text
        student_answer = ""
    return reference_answer, student_answer


# Function to calculate semantic similarity
def calculate_semantic_similarity(reference, student):
    model = SentenceTransformer("bert-base-nli-mean-tokens")
    reference_embedding = model.encode(reference)
    student_embedding = model.encode(student)
    semantic_similarity_score = util.pytorch_cos_sim(
        reference_embedding, student_embedding
    ).item()
    return semantic_similarity_score


# Function to calculate lexical similarity
def calculate_lexical_similarity(reference, student):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([reference, student])
    lexical_similarity_score = 1 - cosine(
        tfidf_matrix[0].toarray().flatten(), tfidf_matrix[1].toarray().flatten()
    )
    return lexical_similarity_score


# Function to extract features
def extract_features(reference_answer, student_answer):
    semantic_similarity_score = calculate_semantic_similarity(
        reference_answer, student_answer
    )
    lexical_similarity_score = calculate_lexical_similarity(
        reference_answer, student_answer
    )
    return [semantic_similarity_score, lexical_similarity_score]


# Load the pre-trained Random Forest model
@st.cache_data
def load_model(model_path):
    return joblib.load(model_path)


# Function to predict score
def predict_score(model, features):
    return model.predict([features])[0]


def main():
    st.title("GradeAid")

    # File uploader for PDF
    uploaded_file = st.file_uploader("Upload PDF", type=["pdf"])

    if uploaded_file is not None:
        # Create a temporary file to store the uploaded PDF content
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_file.write(uploaded_file.read())

            # Load the PDF from the temporary file
        pdf_text = extract_text_from_pdf(temp_file.name)

        if pdf_text:
            reference_answer, student_answer = split_reference_and_student_answers(
                pdf_text
            )
            st.subheader("Reference answer:")
            st.write(reference_answer)
            st.subheader("Student answer:")
            st.write(student_answer)

            features = extract_features(reference_answer, student_answer)

            # Load pre-trained model
            model = load_model("random_forest_model.pkl")

            predicted_score = predict_score(model, features)
            st.subheader("Predicted Score:")
            st.markdown(
                f"<h2 style='color: green; text-align: center;'>{predicted_score}/5</h2>",
                unsafe_allow_html=True,
            )

            # Detailed analysis button
            if st.button("Show Feedback"):
                st.write("Keywords missing in your answer are \n\n")
                df = pd.DataFrame(
                    feedback.findMissingKeywords(reference_answer, student_answer),
                    columns=["Missing Keywords"],
                )
                df.index += 1
                st.table(df)
            # Remove the temporary file
        os.unlink(temp_file.name)


if __name__ == "__main__":
    main()
