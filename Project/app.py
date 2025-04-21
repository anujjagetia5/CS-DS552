import streamlit as st
from PyPDF2 import PdfReader
import base64
import os
import tempfile
from summarizer import get_fast_summarizer, summarize_multiple_texts

# Set page configuration with custom theme
st.set_page_config(
    page_title="DocSum", 
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        font-weight: 700;
        margin-bottom: 1rem;
        text-align: center;
    }
    .sub-header {
        font-size: 1.5rem;
        font-weight: 600;
        color: #0D47A1;
        border-bottom: 2px solid #1E88E5;
        padding-bottom: 0.5rem;
        margin-top: 1.5rem;
    }
    .success-box {
        background-color: #E3F2FD;
        border-radius: 5px;
        padding: 20px;
        border-left: 5px solid #1E88E5;
    }
    .summary-title {
        background-color: #1E88E5;
        color: white;
        padding: 10px;
        border-radius: 5px 5px 0 0;
        text-align: center;
        font-weight: 600;
    }
    .summary-content {
        border: 1px solid #1E88E5;
        border-radius: 0 0 5px 5px;
        padding: 15px;
        background-color: white;
    }
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        background-color: #1E88E5;
        color: white;
    }
    .sidebar-header {
        text-align: center;
        font-weight: 600;
        color: #0D47A1;
        border-bottom: 2px solid #1E88E5;
        padding-bottom: 0.5rem;
    }
    .info-box {
        background-color: #E8F5E9;
        border-radius: 5px;
        padding: 10px;
        border-left: 5px solid #4CAF50;
    }
</style>
""", unsafe_allow_html=True)

# Application Header
st.markdown("<h1 class='main-header'>Your Personal PDF Summarizer</h1>", unsafe_allow_html=True)

# App description
st.markdown("""
<div class="info-box">
Upload your PDFs and get AI-generated summaries in seconds. 
Perfect for research papers, reports, articles, and more!
</div>
""", unsafe_allow_html=True)

# Clear cache function
def clear_cache():
    # Clear Streamlit cache
    st.cache_data.clear()
    st.cache_resource.clear()
    # Display success message
    st.success("Cache cleared successfully!")

# Sidebar with improved styling
st.sidebar.markdown("<h3 class='sidebar-header'>Summary Settings</h3>", unsafe_allow_html=True)
num_pdfs = st.sidebar.slider("Number of PDFs to upload", 1, 5, 1)
summary_length = st.sidebar.select_slider(
    "Summary length", 
    options=["Very Short", "Short", "Medium", "Long", "Very Long"],
    value="Medium"
)

# Map summary length to token counts
summary_length_map = {
    "Very Short": (30, 80),
    "Short": (50, 120),
    "Medium": (100, 200),
    "Long": (150, 300),
    "Very Long": (200, 400)
}

min_length, max_length = summary_length_map[summary_length]

# Performance settings
st.sidebar.markdown("<h3 class='sidebar-header'>Advanced Settings</h3>", unsafe_allow_html=True)
use_fast_model = st.sidebar.checkbox("Use faster model (less accurate)", value=True)
max_pages_per_pdf = st.sidebar.slider("Max pages per PDF", 1, 400, 20, help="Maximum number of pages to process from each PDF")

# Add sampling strategy for large documents
sampling_strategy = st.sidebar.selectbox(
    "Sampling strategy for large documents",
    options=["Process all pages", "Focus on beginning & end", "Uniform sampling", "Focus on beginning"],
    index=1,
    help="How to sample pages when processing large documents"
)

# Map friendly names to internal strategy codes
sampling_strategy_map = {
    "Process all pages": None,
    "Focus on beginning & end": "bookend",
    "Uniform sampling": "uniform",
    "Focus on beginning": "front_heavy"
}

# Add a button to clear cache
if st.sidebar.button("Clear Cache", help="Clear cached PDF data and models"):
    clear_cache()

# Add author info
st.sidebar.markdown("---")
st.sidebar.markdown("<div style='text-align: center'>Created by Anuj</div>", unsafe_allow_html=True)

# Constants
CHUNK_SIZE = 1500

# Load Summarizer (cached)
@st.cache_resource
def load_model():
    return get_fast_summarizer() if use_fast_model else None

summarizer = load_model()

# Function to extract text from PDF with progress
@st.cache_data
def extract_text_from_pdf(uploaded_file, max_pages, sampling_strategy=None):
    from pdf_utils import extract_text_from_pdf_parallel
    
    # Convert sampling strategy to internal code
    strategy_code = sampling_strategy_map.get(sampling_strategy)
    
    # Use the enhanced parallel extraction
    text, total_pages, processed_pages = extract_text_from_pdf_parallel(
        uploaded_file, 
        max_pages=max_pages,
        workers=4,
        sampling_strategy=strategy_code
    )
    
    return text, total_pages, processed_pages

# Create columns for the main content
col1, col2 = st.columns([2, 3])

with col1:
    st.markdown("<h3 class='sub-header'>Upload Documents</h3>", unsafe_allow_html=True)
    
    # Multiple PDF upload
    uploaded_files = []
    for i in range(num_pdfs):
        uploaded_file = st.file_uploader(f"Upload PDF #{i+1}", type=["pdf"], key=f"pdf_{i}")
        if uploaded_file:
            uploaded_files.append(uploaded_file)

# Initialize session state for summary storage and regeneration
if 'current_summary' not in st.session_state:
    st.session_state.current_summary = ""

if 'all_texts' not in st.session_state:
    st.session_state.all_texts = []

if 'pdf_info' not in st.session_state:
    st.session_state.pdf_info = []

# Function to generate summary
def generate_summary(is_regenerate=False):
    if st.session_state.all_texts:
        with st.spinner("Generating summary... This may take a few moments"):
            # For regeneration, adjust parameters slightly to get variation
            if is_regenerate:
                import random
                # Adjust length parameters slightly
                variation_min = random.randint(-20, 20)
                variation_max = random.randint(-20, 20)
                adj_min_length = max(min_length + variation_min, 30)  # Don't go below 30
                adj_max_length = max(max_length + variation_max, adj_min_length + 20)  # Ensure max > min
                
                # Summarize with slightly different parameters
                summary = summarize_multiple_texts(
                    st.session_state.all_texts, 
                    summarizer=summarizer,
                    max_length=adj_max_length,
                    min_length=adj_min_length,
                    use_fast_model=use_fast_model,
                    # Add randomness parameter
                    randomness=True
                )
            else:
                # Original summary generation
                summary = summarize_multiple_texts(
                    st.session_state.all_texts, 
                    summarizer=summarizer,
                    max_length=max_length,
                    min_length=min_length,
                    use_fast_model=use_fast_model,
                    randomness=False
                )
            
            # Store the summary in session state
            st.session_state.current_summary = summary
            
            # Force a rerun to display the new summary
            st.experimental_rerun()

# Process PDFs if uploaded
if uploaded_files:
    # Process each PDF
    st.session_state.all_texts = []
    st.session_state.pdf_info = []
    
    # Process PDFs with a progress bar
    progress_bar = st.progress(0)
    
    for idx, uploaded_file in enumerate(uploaded_files):
        # Display PDF name
        st.markdown(f"<h4>PDF #{idx+1}: {uploaded_file.name}</h4>", unsafe_allow_html=True)
        
        # Extract text with progress indication
        with st.spinner(f"Extracting text from PDF #{idx+1}..."):
            extracted_text, total_pages, processed_pages = extract_text_from_pdf(
                uploaded_file, 
                max_pages_per_pdf,
                sampling_strategy
            )
            
            # Show info about the PDF
            st.markdown(
                f"<div class='info-box'>Processed {processed_pages} pages out of {total_pages} total pages</div>", 
                unsafe_allow_html=True
            )
            
            # Display PDF preview (smaller iframe for speed)
            with st.expander("View PDF"):
                base64_pdf = base64.b64encode(uploaded_file.read()).decode("utf-8")
                pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="300px" type="application/pdf"></iframe>'
                st.markdown(pdf_display, unsafe_allow_html=True)
            
            # Reset file pointer
            uploaded_file.seek(0)
            
            # Add text to the collection
            if extracted_text.strip():
                st.session_state.all_texts.append(extracted_text)
                st.session_state.pdf_info.append(f"{uploaded_file.name} ({processed_pages} of {total_pages} pages)")
            else:
                st.warning(f"No text could be extracted from PDF #{idx+1}")
        
        # Update progress bar
        progress_bar.progress((idx + 1) / len(uploaded_files))
    
    # Clear progress bar
    progress_bar.empty()
    
    # Generate summary button
    if st.session_state.all_texts and st.session_state.current_summary == "":
        if st.button("Generate Summary", key="generate_initial"):
            generate_summary()

# Display summary in the second column
with col2:
    st.markdown("<h3 class='sub-header'>Summary Output</h3>", unsafe_allow_html=True)
    
    if st.session_state.current_summary:
        # Display the title
        st.markdown('<div class="summary-title">🧠 Document Summary</div>', unsafe_allow_html=True)
        
        # Display the summary content
        st.markdown(f'<div class="summary-content">{st.session_state.current_summary}</div>', unsafe_allow_html=True)
        
        # Add metadata about the summarization
        st.caption(f"Summary generated from: {', '.join(st.session_state.pdf_info)}")
        
        # Add regenerate and download buttons in a row
        col_regen, col_download = st.columns(2)
        
        with col_regen:
            if st.button("🔄 Regenerate Summary", key="regenerate"):
                generate_summary(is_regenerate=True)
        
        with col_download:
            st.download_button(
                label="📥 Download Summary",
                data=st.session_state.current_summary,
                file_name="pdf_summary.txt",
                mime="text/plain"
            )
    else:
        if uploaded_files:
            st.info("Upload your documents and click 'Generate Summary' to create a summary.")
        else:
            st.info("Upload your documents on the left to get started.")
