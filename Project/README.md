# 📄 PDF Summarizer

A web app to upload multiple PDFs and get summaries using transformer models.

## 🔧 Setup

```bash
pip install -r requirements.txt
streamlit run app.py
```

## 🚀 Features

- Upload multiple PDFs (up to 5)
- Extract text from all documents
- Generate a combined summary
- Download the summary as a text file
- Clear cache functionality for privacy
- Adjustable summary length
- PDFs with around 400 pages can be summarized at once.

## 📋 Usage

1. Use the slider to select how many PDFs you want to upload
2. Upload your PDFs in the file uploaders
3. Adjust the summary length using the sentences slider
4. View the generated summary
5. Download the summary if needed
6. Clear the cache when done using the button in the sidebar

## 🧠 Technical Details

This app uses the BART-CNN model from Facebook/Meta for text summarization via the Hugging Face Transformers library.
