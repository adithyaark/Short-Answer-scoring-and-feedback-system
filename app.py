import streamlit as st
import os
import pandas as pd
import tempfile
import fitz
import pytesseract
from PIL import Image
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


# Function to extract text from PDF
def extract_text_from_pdf(pdf_path):
    try:
        pdf_document = fitz.open(pdf_path)
        text = ""
        for page_number in range(pdf_document.page_count):
            page = pdf_document[page_number]
            text += page.get_text()
        return text
    except Exception as e:
        st.error(f"Error extracting PDF: {e}")
        return ""


# Function to extract text from image
def extract_text_from_image(image_path):
    try:
        text = pytesseract.image_to_string(Image.open(image_path), lang="eng")
        return text
    except Exception as e:
        st.error(f"Error extracting image: {e}")
        return ""


# Load and cache the SentenceTransformer model
@st.cache_resource
def load_sentence_transformer():
    return SentenceTransformer("all-MiniLM-L6-v2")


# Function to calculate semantic similarity using cached model
def calculate_semantic_similarity(reference, student):
    model = load_sentence_transformer()
    reference_embedding = model.encode(reference, convert_to_tensor=True)
    student_embedding = model.encode(student, convert_to_tensor=True)
    semantic_similarity_score = util.pytorch_cos_sim(
        reference_embedding, student_embedding
    ).item()
    return semantic_similarity_score


# Function to calculate score based on semantic similarity
def calculate_score(reference_answer, student_answer):
    """
    Calculate score using semantic similarity.
    Score = similarity × 10
    """
    if not student_answer.strip():
        return 0.0
    
    with st.spinner("Calculating score..."):
        semantic_similarity = calculate_semantic_similarity(reference_answer, student_answer)
        score = semantic_similarity * 10
        return min(score, 10.0)  # Cap score at 10


def main():
    st.title("GradeAid")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Reference Answer")
        reference_file = st.file_uploader("Upload Reference Answer PDF", type=["pdf"], key="reference")

    with col2:
        st.subheader("Student Answer")
        student_file = st.file_uploader("Upload Student Answer PDF", type=["pdf"], key="student")

    if reference_file is not None and student_file is not None:
        # Process reference answer
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as ref_temp:
            ref_temp.write(reference_file.read())
            ref_temp_path = ref_temp.name

        # Process student answer
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as stu_temp:
            stu_temp.write(student_file.read())
            stu_temp_path = stu_temp.name

        try:
            # Extract text from both PDFs
            reference_answer = extract_text_from_pdf(ref_temp_path)
            student_answer = extract_text_from_pdf(stu_temp_path)

            if reference_answer and student_answer:
                # Display extracted text
                st.subheader("Reference Answer Content:")
                st.write(reference_answer)
                
                st.subheader("Student Answer Content:")
                st.write(student_answer)

                # Calculate score using semantic similarity
                predicted_score = calculate_score(reference_answer, student_answer)
                
                st.subheader("Predicted Score:")
                st.markdown(
                    f"<h2 style='color: green; text-align: center;'>{predicted_score:.2f}/10</h2>",
                    unsafe_allow_html=True,
                )

                # Detailed analysis button
                if st.button("Show Feedback"):
                    with st.spinner("Generating feedback..."):
                        try:
                            missing_keywords = feedback.findMissingKeywords(reference_answer, student_answer)
                            
                            if missing_keywords:
                                st.write("**Keywords missing in your answer:**\n")
                                df = pd.DataFrame(missing_keywords, columns=["Missing Keywords"])
                                df.index += 1
                                st.table(df)
                            else:
                                st.success("No keywords missing!")
                        except Exception as e:
                            st.error(f"Error generating feedback: {e}")

        finally:
            # Remove temporary files
            if os.path.exists(ref_temp_path):
                os.unlink(ref_temp_path)
            if os.path.exists(stu_temp_path):
                os.unlink(stu_temp_path)


if __name__ == "__main__":
    main()