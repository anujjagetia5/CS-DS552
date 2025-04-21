import nltk
import os
import ssl
import sys

def download_nltk_data():
    """Download NLTK data and configure paths."""
    print("Starting NLTK setup...")
    
    # Create directory for NLTK data
    nltk_data_dir = os.path.join(os.path.dirname(__file__), 'nltk_data')
    os.makedirs(nltk_data_dir, exist_ok=True)
    
    # Add the directory to NLTK's search path
    nltk.data.path.insert(0, nltk_data_dir)
    print(f"NLTK data directory set to: {nltk_data_dir}")
    print(f"NLTK search path: {nltk.data.path}")
    
    # Handle SSL certificate verification issues
    try:
        _create_unverified_https_context = ssl._create_unverified_context
    except AttributeError:
        pass
    else:
        ssl._create_default_https_context = _create_unverified_https_context
    
    # Download resources with explicit error handling
    resources = ['punkt', 'stopwords']
    for resource in resources:
        try:
            print(f"Downloading {resource}...")
            nltk.download(resource, download_dir=nltk_data_dir, quiet=False)
            print(f"Successfully downloaded {resource}")
        except Exception as e:
            print(f"Error downloading {resource}: {e}")
    
    # Verify the downloads
    for resource in resources:
        try:
            nltk.data.find(f"{'tokenizers/' if resource == 'punkt' else 'corpora/'}{resource}")
            print(f"Verified {resource} is available")
        except LookupError as e:
            print(f"WARNING: Failed to verify {resource}: {e}")
    
    # List contents of nltk_data directory
    if os.path.exists(nltk_data_dir):
        print(f"Contents of {nltk_data_dir}:")
        for root, dirs, files in os.walk(nltk_data_dir):
            for d in dirs:
                print(f"  Directory: {os.path.join(root, d)}")
            for f in files:
                print(f"  File: {os.path.join(root, f)}")
    else:
        print(f"WARNING: {nltk_data_dir} does not exist after download attempt")
    
    print("NLTK setup completed")

if __name__ == "__main__":
    download_nltk_data()
