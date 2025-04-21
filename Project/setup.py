import nltk
import os
import ssl

def download_nltk_data():
    # Create directory for NLTK data
    nltk_data_dir = os.path.join(os.path.dirname(__file__), 'nltk_data')
    os.makedirs(nltk_data_dir, exist_ok=True)
    
    # Add the directory to NLTK's search path
    nltk.data.path.insert(0, nltk_data_dir)
    
    # Handle SSL certificate verification issues that can occur on some platforms
    try:
        _create_unverified_https_context = ssl._create_unverified_context
    except AttributeError:
        pass
    else:
        ssl._create_default_https_context = _create_unverified_https_context
    
    # Download all required resources
    resources = ['punkt', 'stopwords']
    for resource in resources:
        try:
            nltk.download(resource, download_dir=nltk_data_dir)
            print(f"Successfully downloaded {resource}")
        except Exception as e:
            print(f"Error downloading {resource}: {e}")
    
    # List all downloaded data
    print("NLTK data path:", nltk.data.path)
    print("Files in nltk_data directory:", os.listdir(nltk_data_dir) if os.path.exists(nltk_data_dir) else "Directory not found")
