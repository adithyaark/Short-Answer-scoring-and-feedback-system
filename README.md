# Short Answer Scoring and Feedback System

This project is a simple web application that automatically evaluates student short-answer responses by comparing them with a reference answer.

It uses **semantic similarity** to calculate marks and provides **keyword-based feedback** to help students understand what concepts they missed.

The application is built using **Python and Streamlit**.

---

## What This Project Does

- Accepts **Reference Answer** and **Student Answer** as PDF files
- Extracts text from PDFs
- Calculates a score out of **10** based on meaning (not exact words)
- Shows **missing keywords** as feedback
- Runs completely in a browser using Streamlit

---

## Tools and Technologies Used

- **Python** – Core programming language
- **Streamlit** – Web interface
- **SentenceTransformer** – Semantic similarity model
- **PyMuPDF (fitz)** – PDF text extraction
- **Pandas** – Display feedback in tables
- **Cortical.io API** – Keyword extraction (for feedback)
- **Virtual Environment (venv)** – Dependency isolation

---
<img width="1919" height="982" alt="Screenshot 2026-02-03 225808" src="https://github.com/user-attachments/assets/de008a19-1350-4e48-98b1-e0da6c1f47e6" />
![Uploading Screenshot 2026-02-03 225837.png…]()


