import os

import spacy
import random
import streamlit as st
import fitz  # PyMuPDF for PDF extraction
from io import BytesIO
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from sentence_transformers import SentenceTransformer, util

# Load SpaCy model
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    with st.spinner("Downloading SpaCy model..."):
        spacy.cli.download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")


@st.cache_resource
def load_semantic_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

semantic_model = load_semantic_model()


# Function to extract text from PDF
def extract_text_from_pdf(pdf_file):
    pdf_document = fitz.open(stream=pdf_file.read(), filetype="pdf")
    text = ""
    for page_num in range(len(pdf_document)):
        page = pdf_document.load_page(page_num)
        text += page.get_text()
    return text

# Helper: Split long text into smaller chunks
def split_text_into_chunks(text, max_chars=80000):
    """Split text into chunks to avoid SpaCy's max_length limit."""
    return [text[i:i+max_chars] for i in range(0, len(text), max_chars)]

def generate_mcqs(text, num_questions=5, difficulty="Medium"):
    if not text or not isinstance(text, str) or not text.strip():
        return []

    # Process in chunks and store all sentences/entities/nouns
    all_sentences = []
    all_entities = []
    all_nouns = []

    chunks = split_text_into_chunks(text)
    for chunk in chunks:
        doc = nlp(chunk)
        all_sentences.extend([sent.text for sent in doc.sents])
        all_entities.extend(list(doc.ents))
        all_nouns.extend([token for token in doc if token.pos_ == "NOUN"])

    # Randomly pick sentences
    selected_sentences = random.sample(all_sentences, min(num_questions, len(all_sentences)))
    mcqs = []

    for sentence in selected_sentences:
        sent_doc = nlp(sentence)

        # Step 1: Pick target (entity > noun fallback)
        entities = [ent for ent in sent_doc.ents if ent.label_ in ["PERSON", "ORG", "GPE", "DATE"]]
        if entities:
            target = entities[0].text
            target_label = entities[0].label_
        else:
            nouns = [token.text for token in sent_doc if token.pos_ == "NOUN"]
            if not nouns:
                continue
            target = nouns[0]
            target_label = None

        # Step 2: Create question stem
        if difficulty == "Easy":
            question_stem = sentence.replace(target, "_______")
        elif difficulty == "Medium":
            words = sentence.split()
            idx = words.index(target) if target in words else -1
            if idx > 0:
                words[max(0, idx-1):min(len(words), idx+2)] = ["_______"]
            question_stem = " ".join(words)
        elif difficulty == "Hard":
            question_stem = f"Fill in the blank: _______ (Topic related to '{target_label if target_label else 'the text'}')"

        # Step 3: Build distractors
        distractors = []

        # Initial candidate pool based on difficulty and target label
        if difficulty == "Easy":
            candidates = list(set(ent.text for ent in all_entities if ent.text != target))
        else:
            if target_label:
                candidates = list(
                    set(ent.text for ent in all_entities if ent.label_ == target_label and ent.text != target))
            else:
                candidates = list(set(token.text for token in all_nouns if token.text != target))

        # If not enough candidates, broaden the pool
        if len(candidates) < 3:
            broader_candidates = list(set(
                [ent.text for ent in all_entities if ent.text != target] +
                [token.text for token in all_nouns if token.text != target]
            ))
            candidates = list(set(candidates + broader_candidates))

        # Generate top similar distractors
        if candidates:
            target_emb = semantic_model.encode(target, convert_to_tensor=True)
            candidate_embs = semantic_model.encode(candidates, convert_to_tensor=True)
            similarities = util.cos_sim(target_emb, candidate_embs)[0]
            sorted_candidates = [candidates[i] for i in similarities.argsort(descending=True)]
            distractors = [word for word in sorted_candidates if word != target][:3]

        # Still not enough? Fill with random from all nouns/entities
        while len(distractors) < 3:
            fallback_pool = list(set(
                [ent.text for ent in all_entities if ent.text != target] +
                [token.text for token in all_nouns if token.text != target]
            ))
            if fallback_pool:
                distractors.append(random.choice(fallback_pool))
            else:
                distractors.append("Option")

        # Remove duplicates & trim to exactly 3
        distractors = list(dict.fromkeys(distractors))[:3]

        # Step 4: Finalize MCQ
        answer_choices = [target] + distractors[:3]
        random.shuffle(answer_choices)
        correct_answer = chr(65 + answer_choices.index(target))
        mcqs.append((question_stem, answer_choices, correct_answer))

    return mcqs


# Function to create a text file from MCQs
def create_text(mcqs):
    text_content = ""
    for idx, (question, choices, correct_answer) in enumerate(mcqs):
        text_content += f"Q{idx+1}: {question}\n"
        for i, choice in enumerate(choices):
            text_content += f"    {chr(65+i)}. {choice}\n"
        text_content += f"Correct Answer: {correct_answer}\n\n"
    return text_content

# Initialize Session State for tracking user answers
if "user_answers" not in st.session_state:
    st.session_state.user_answers = {}

if "check_clicked" not in st.session_state:
    st.session_state.check_clicked = False

if "mcqs" not in st.session_state:
    st.session_state.mcqs = []

# Sidebar for navigation
st.sidebar.title("Navigation")
st.sidebar.header("Instructions")
st.sidebar.write("1. Upload a PDF file to extract text.")
st.sidebar.write("2. Select the difficulty level and the number of questions.")
st.sidebar.write("3. Generate MCQs and check your answers.")
st.sidebar.write("4. Download the MCQs as a PDF or text file.")

# Sidebar options for settings
st.sidebar.header("Settings")
difficulty = st.sidebar.selectbox("Select difficulty level", options=["Easy", "Medium", "Hard"])
num_questions = st.sidebar.selectbox("Select number of questions", options=list(range(1, 11)), index=4)

# Streamlit UI for PDF upload
st.title("Enhanced PDF to MCQ Generator")
uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")

if uploaded_file is not None:
    with st.spinner("Extracting text from PDF..."):
        pdf_text = extract_text_from_pdf(uploaded_file)
        source_text = pdf_text

    st.subheader("Extracted Text")
    st.write(pdf_text[:2000])  # Display the first 2000 characters

    # Generate MCQs Button
    if source_text and source_text.strip() and st.button("Generate MCQs"):
        with st.spinner("Generating MCQs..."):
            mcqs = generate_mcqs(source_text, num_questions, difficulty)
            st.session_state.mcqs = mcqs
            st.session_state.user_answers = {idx: None for idx in range(len(mcqs))}
            st.session_state.check_clicked = False
            st.success("MCQs generated successfully!")
    elif uploaded_file and not source_text.strip():
        st.warning("No text could be extracted from this PDF. It might be a scanned image.")

    # MCQ Display
    if "mcqs" in st.session_state:
        mcqs = st.session_state.mcqs
        st.subheader("Generated MCQs")

        for idx, (question, choices, correct_answer) in enumerate(mcqs):
            st.write(f"**Q{idx + 1}.** {question}")

            # Add a placeholder to the choices
            choices_with_placeholder = ["Select an answer"] + choices

            # Store the user's selected answer in session state
            user_answer = st.radio(
                f"Choose an option for Q{idx + 1}:",
                choices_with_placeholder,
                index=0 if st.session_state.user_answers[idx] is None else choices_with_placeholder.index(
                    st.session_state.user_answers[idx]),
                key=f"radio_{idx}"
            )

            # Update user answers directly after radio selection
            st.session_state.user_answers[idx] = user_answer

        # Check Answers Button
        if st.button("Check Answers"):
            st.session_state.check_clicked = True

        # Show Results
        if st.session_state.check_clicked:
            st.subheader("Results")
            correct_count = 0
            for idx, (question, choices, correct_answer) in enumerate(mcqs):
                st.write(f"**Q{idx + 1}.** {question}")
                selected_answer = st.session_state.user_answers[idx]

                if selected_answer == "Select an answer" or selected_answer is None:
                    st.write(f"⚠️ **No option selected for this question!**")
                else:
                    correct_answer_text = \
                    [choice for choice in choices if chr(65 + choices.index(choice)) == correct_answer][0]
                    if selected_answer == correct_answer_text:
                        st.write("✅ **Correct!**")
                        correct_count += 1
                    else:
                        st.write(f"❌ **Wrong!** The correct answer is **{correct_answer_text}**.")
                st.write("---")
            st.write(f"Your score: {correct_count}/{len(mcqs)}")

        # Download MCQs as Text
        st.subheader("Download MCQs")
        text_content = create_text(st.session_state.mcqs)
        st.download_button(
            label="Download as Text",
            data=text_content,
            file_name="mcqs.txt",
            mime="text/plain"
        )
