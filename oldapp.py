import os
import torch

# MUST COME BEFORE OTHER IMPORTS
torch.classes.__path__ = [os.path.join(torch.__path__[0], "classes")]
os.environ["STREAMLIT_SERVER_ENABLE_STATIC_FILE_HANDLER"] = "false"

import streamlit as st
import tempfile
import hypernetx as hnx
from pdf_processor import extract_text, preprocess_text
from hypergraph_processor import HypergraphProcessor, HyperGNN
from question_generator import QuestionGenerator

# Set up page config
st.set_page_config(layout="wide", page_title="HyperGraph Quiz Generator")

# Application title
st.title("HyperGraph Quiz Generator")
st.write("Upload a PDF and generate AI-powered quizzes using hypergraph relationships")

# Initialize session state
if 'hypergraph' not in st.session_state:
    st.session_state.hypergraph = None
if 'questions' not in st.session_state:
    st.session_state.questions = None
if 'processing_stage' not in st.session_state:
    st.session_state.processing_stage = None

# Sidebar controls
with st.sidebar:
    st.header("PDF Upload")
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
    
    st.header("Quiz Configuration")
    difficulty = st.select_slider(
        "Question Difficulty",
        options=["easy", "medium", "hard"],
        value="medium"
    )
    num_questions = st.slider(
        "Number of Questions",
        min_value=1,
        max_value=20,
        value=10
    )
    
    process_btn = st.button("Generate Quiz")

# Main processing pipeline
if process_btn and uploaded_file:
    with st.status("Processing PDF...", expanded=True) as status:
        # Step 1: Process PDF
        st.write("Extracting text and preprocessing...")
        text_pages = extract_text(uploaded_file)
        processed_text = preprocess_text(text_pages)
        
        # Step 2: Build Hypergraph
        st.write("Building knowledge hypergraph...")
        processor = HypergraphProcessor()
        lda_params = processor.optimize_lda(processed_text)
        hypergraph = processor.build_hypergraph()
        
        # Step 3: Train HyperGNN
        st.write("Training Hypergraph Neural Network...")
        hypergnn = HyperGNN(hypergraph, processor.lda_model)
        model, embeddings = hypergnn.train()
        
        # Step 4: Generate Questions
        st.write("Generating quiz questions...")
        generator = QuestionGenerator(hypergraph, embeddings)
        questions = generator.generate(
            num_questions=num_questions,
            difficulty=difficulty
        )
        
        st.session_state.hypergraph = hypergraph
        st.session_state.questions = questions
        status.update(label="Processing complete!", state="complete")

# Display Hypergraph Visualization
if st.session_state.hypergraph:
    if st.button("Show Knowledge Hypergraph"):
        with st.expander("Document Knowledge Hypergraph", expanded=True):
            fig = plt.figure(figsize=(12, 8))
            hnx.drawing.draw(st.session_state.hypergraph)
            st.pyplot(fig)

# Quiz Interface
if st.session_state.questions:
    st.header("Generated Quiz")
    
    answers = {}
    score = 0
    
    with st.form("quiz_form"):
        for i, q in enumerate(st.session_state.questions):
            st.subheader(f"Question {i+1}")
            st.markdown(f"**{q['question']}**")
            
            options = list(q['options'].items())
            user_ans = st.radio(
                "Select your answer:",
                options,
                format_func=lambda x: f"{x[0]}. {x[1]}",
                key=f"q_{i}"
            )
            answers[i] = user_ans[0]
        
        submitted = st.form_submit_button("Submit Quiz")
    
    if submitted:
        st.header("Quiz Results")
        score = 0
        
        for i, q in enumerate(st.session_state.questions):
            st.subheader(f"Question {i+1}")
            correct = q['correct']
            user_ans = answers.get(i, " ")
            
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"**Your Answer:** {user_ans}. {q['options'][user_ans]}")
            with col2:
                st.markdown(f"**Correct Answer:** {correct}. {q['options'][correct]}")
            
            if user_ans == correct:
                score += 1
                st.success("Correct!")
            else:
                st.error("Incorrect")
            
            st.markdown(f"**Explanation:** {q['explanation']}")
            st.divider()
        
        total = len(st.session_state.questions)
        st.success(f"Final Score: {score}/{total} ({(score/total)*100:.1f}%)")

# Add footer
st.markdown("---")
st.markdown("Built with Streamlit • HyperNetX • Ollama")