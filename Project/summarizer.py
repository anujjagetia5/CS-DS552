from transformers import pipeline
import nltk
import os
import re
import threading
import random
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed

# Global variables for the summarization pipeline
_summarizer = None
_fast_summarizer_initialized = False
_lock = threading.Lock()

# Define fallback stopwords in case NLTK fails
FALLBACK_STOPWORDS = {
    'a', 'an', 'the', 'and', 'or', 'but', 'if', 'then', 'else', 'when', 'at', 'from', 
    'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 
    'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 
    'off', 'over', 'under', 'is', 'am', 'are', 'was', 'were', 'be', 'been', 'being', 
    'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'i', 'my', 'me', 
    'we', 'our', 'you', 'your', 'he', 'his', 'him', 'she', 'her', 'it', 'its', 'they', 
    'them', 'their', 'this', 'that', 'these', 'those'
}

# Define a fallback sentence tokenizer
def fallback_sent_tokenize(text):
    """Simple sentence tokenizer when NLTK is not available."""
    print("Using fallback sentence tokenizer")
    # Split by periods, exclamation points, and question marks followed by a space or newline
    sentences = re.split(r'(?<=[.!?])\s+', text)
    # Filter out empty sentences and those that are too short
    return [s.strip() for s in sentences if s.strip() and len(s.strip()) > 10]

# Fallback word tokenizer
def fallback_word_tokenize(text):
    """Simple word tokenizer when NLTK is not available."""
    print("Using fallback word tokenizer")
    # Split by whitespace and punctuation
    words = re.findall(r'\b\w+\b', text.lower())
    return words

def ensure_nltk():
    """Ensure NLTK resources are available, with robust error handling."""
    print("Ensuring NLTK resources...")
    try:
        # Set up NLTK data directory
        nltk_data_dir = os.path.join(os.path.dirname(__file__), 'nltk_data')
        os.makedirs(nltk_data_dir, exist_ok=True)
        
        # Add to NLTK's search path if not already there
        if nltk_data_dir not in nltk.data.path:
            nltk.data.path.insert(0, nltk_data_dir)
        
        print(f"NLTK data path: {nltk.data.path}")
        
        # Try to download and use the resources
        try:
            print("Attempting to download punkt...")
            nltk.download('punkt', download_dir=nltk_data_dir, quiet=False)
            print("Attempting to download stopwords...")
            nltk.download('stopwords', download_dir=nltk_data_dir, quiet=False)
            
            # Test if they work
            print("Testing if punkt works...")
            from nltk.tokenize import sent_tokenize
            test_sent = sent_tokenize("This is a test. This is another test.")
            print(f"Punkt test result: {test_sent}")
            
            print("Testing if stopwords works...")
            from nltk.corpus import stopwords
            test_stops = stopwords.words('english')
            print(f"Stopwords test sample: {test_stops[:5] if test_stops else 'No stopwords found'}")
            
            return True
        except Exception as e:
            print(f"NLTK resource initialization error: {e}")
            return False
            
    except Exception as e:
        print(f"NLTK setup error: {e}")
        return False

def get_summarizer():
    """Get or initialize the transformer-based summarizer."""
    global _summarizer
    if _summarizer is None:
        # Using a smaller, faster model than BART-large-CNN
        _summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    return _summarizer

def get_fast_summarizer():
    """Initialize and get the fast extractive summarizer."""
    global _fast_summarizer_initialized
    if not _fast_summarizer_initialized:
        # Initialize NLTK components
        with _lock:
            if not _fast_summarizer_initialized:
                try:
                    nltk.data.find('tokenizers/punkt')
                    nltk.data.find('corpora/stopwords')
                except LookupError:
                    # If not found, download them
                    ensure_nltk()
                _fast_summarizer_initialized = True
    return True

def clean_text(text):
    """Clean and preprocess text."""
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    # Remove page markers
    text = re.sub(r'--- Page \d+ ---', '', text)
    return text.strip()

def sentence_similarity(sent1, sent2, stopwords=None):
    """Calculate similarity between two sentences."""
    if stopwords is None:
        try:
            from nltk.corpus import stopwords as nltk_stopwords
            stopwords = set(nltk_stopwords.words('english'))
        except Exception as e:
            print(f"Using fallback stopwords due to error: {e}")
            stopwords = FALLBACK_STOPWORDS
    
    # Convert sentences to lowercase and tokenize
    try:
        from nltk.tokenize import word_tokenize
        sent1_words = [w.lower() for w in word_tokenize(sent1) if w.lower() not in stopwords]
        sent2_words = [w.lower() for w in word_tokenize(sent2) if w.lower() not in stopwords]
    except Exception as e:
        print(f"Using fallback word tokenization due to error: {e}")
        sent1_words = [w for w in fallback_word_tokenize(sent1) if w not in stopwords]
        sent2_words = [w for w in fallback_word_tokenize(sent2) if w not in stopwords]
    
    # Create word vectors
    all_words = list(set(sent1_words + sent2_words))
    vector1 = [0] * len(all_words)
    vector2 = [0] * len(all_words)
    
    # Build the vectors
    for w in sent1_words:
        if w in all_words:
            vector1[all_words.index(w)] += 1
            
    for w in sent2_words:
        if w in all_words:
            vector2[all_words.index(w)] += 1
    
    # Calculate cosine similarity
    try:
        if sum(vector1) > 0 and sum(vector2) > 0:
            dot_product = sum(a*b for a, b in zip(vector1, vector2))
            norm_vec1 = sum(a*a for a in vector1) ** 0.5
            norm_vec2 = sum(b*b for b in vector2) ** 0.5
            
            if norm_vec1 > 0 and norm_vec2 > 0:
                return dot_product / (norm_vec1 * norm_vec2)
            return 0
        return 0
    except Exception as e:
        print(f"Error calculating sentence similarity: {e}")
        return 0

def build_similarity_matrix(sentences, stop_words):
    """Build similarity matrix for all sentences."""
    # Create an empty similarity matrix
    similarity_matrix = np.zeros((len(sentences), len(sentences)))
    
    for i in range(len(sentences)):
        for j in range(len(sentences)):
            if i != j:
                similarity_matrix[i][j] = sentence_similarity(
                    sentences[i], sentences[j], stop_words)
    
    return similarity_matrix

def extract_key_phrases(text, num_phrases=5):
    """Extract key phrases from text using simple frequency analysis."""
    try:
        # Get stopwords safely
        try:
            from nltk.corpus import stopwords
            stop_words = set(stopwords.words('english'))
        except Exception as e:
            print(f"Using fallback stopwords due to error: {e}")
            stop_words = FALLBACK_STOPWORDS
        
        # Tokenize safely
        try:
            from nltk.tokenize import word_tokenize
            words = word_tokenize(text.lower())
        except Exception as e:
            print(f"Using fallback word tokenization due to error: {e}")
            words = fallback_word_tokenize(text)
        
        words = [word for word in words if word.isalnum() and word not in stop_words]
        
        # Get word frequencies
        try:
            from nltk import FreqDist
            freq_dist = FreqDist(words)
            most_common = freq_dist.most_common(20)  # Get top 20 words
        except Exception as e:
            print(f"Using fallback frequency distribution due to error: {e}")
            # Fallback to simple counting
            word_freq = {}
            for word in words:
                word_freq[word] = word_freq.get(word, 0) + 1
            most_common = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:20]
        
        # Extract bigrams and trigrams around key words
        key_phrases = []
        
        # Try to use NLTK's sent_tokenize, fallback to our own if it fails
        try:
            from nltk.tokenize import sent_tokenize
            sentences = sent_tokenize(text)
        except Exception as e:
            print(f"Using fallback sentence tokenization due to error: {e}")
            sentences = fallback_sent_tokenize(text)
        
        for sentence in sentences:
            sent_lower = sentence.lower()
            for word, _ in most_common:
                if word in sent_lower:
                    # Find a window around the key word
                    try:
                        from nltk.tokenize import word_tokenize
                        words = word_tokenize(sent_lower)
                    except Exception:
                        words = sent_lower.split()
                    
                    try:
                        idx = words.index(word)
                        start = max(0, idx - 2)
                        end = min(len(words), idx + 3)
                        phrase = " ".join(words[start:end])
                        key_phrases.append(phrase)
                        
                        if len(key_phrases) >= num_phrases:
                            break
                    except ValueError:
                        continue
            if len(key_phrases) >= num_phrases:
                break
                
        return key_phrases[:num_phrases]
    except Exception as e:
        print(f"Error extracting key phrases: {e}")
        return []

def extractive_summarize(text, num_sentences=5, randomness=False):
    """Generate an extractive summary using the TextRank algorithm."""
    try:
        # Get stopwords safely
        try:
            from nltk.corpus import stopwords
            stop_words = set(stopwords.words('english'))
        except Exception as e:
            print(f"Using fallback stopwords due to error: {e}")
            stop_words = FALLBACK_STOPWORDS
        
        # Clean and split the text into sentences
        cleaned_text = clean_text(text)
        
        # Try NLTK's sentence tokenizer, fallback to our own if it fails
        try:
            from nltk.tokenize import sent_tokenize
            sentences = sent_tokenize(cleaned_text)
        except Exception as e:
            print(f"Using fallback sentence tokenizer due to error: {e}")
            sentences = fallback_sent_tokenize(cleaned_text)
        
        # Handle very short texts
        if len(sentences) <= num_sentences:
            return text
        
        # Build similarity matrix
        sentence_similarity_matrix = build_similarity_matrix(sentences, stop_words)
        
        # Apply PageRank-like algorithm
        sentence_similarity_graph = np.array(sentence_similarity_matrix)
        scores = np.array([sum(sentence_similarity_graph[i]) for i in range(len(sentences))])
        
        # Add random variation for regeneration if requested
        if randomness:
            # Add small random variations to scores
            random_factors = np.array([random.uniform(0.7, 1.3) for _ in range(len(scores))])
            scores = scores * random_factors
        
        ranked_sentences = [item[0] for item in sorted(enumerate(scores), key=lambda item: -item[1])]
        
        # Select top sentences preserving the original order
        selected_indices = sorted(ranked_sentences[:num_sentences])
        
        # If randomness is enabled, potentially swap some sentences with others
        if randomness and len(sentences) > num_sentences + 3:
            # Potentially replace some selected sentences with others
            for i in range(min(2, num_sentences)):
                if random.random() < 0.5:  # 50% chance to replace
                    # Find a sentence that's not in the top but close
                    replacement_candidates = ranked_sentences[num_sentences:num_sentences+5]
                    if replacement_candidates:
                        replace_idx = random.choice(replacement_candidates)
                        # Find which to replace (randomly select from current)
                        to_replace_idx = random.choice(range(len(selected_indices)))
                        selected_indices[to_replace_idx] = replace_idx
            
            # Re-sort to maintain document order
            selected_indices = sorted(selected_indices)
        
        summary = ' '.join([sentences[i] for i in selected_indices])
        
        return summary
    except Exception as e:
        print(f"Error in extractive summarization: {e}")
        # Fallback to a simpler approach
        sentences = fallback_sent_tokenize(text)
        num_sentences = min(num_sentences, len(sentences))
        summary = ' '.join(sentences[:num_sentences])
        return summary

def rephrase_sentence(sentence):
    """Simple sentence rephrasing by changing word order and structure."""
    # This is a very basic implementation - you might want to use more advanced NLP
    # for proper rephrasing in a production system
    
    try:
        from nltk.tokenize import word_tokenize
        from nltk import pos_tag
        words = word_tokenize(sentence)
    except Exception:
        # Fallback to simple word splitting
        words = sentence.split()
        # Cannot do POS tagging, so return the sentence as is
        return sentence
    
    # Check if sentence is long enough to rephrase
    if len(words) < 5:
        return sentence
        
    try:
        # Get sentence parts
        tagged = pos_tag(words)
        
        # Different simple rephrasing strategies
        strategies = [
            # Strategy 1: If sentence starts with "The", try moving an adjective
            lambda s, t: (s.startswith("The") and any(tag.startswith("JJ") for _, tag in t[1:4])),
            
            # Strategy 2: Try passive to active voice (very simplified)
            lambda s, t: ("was" in words or "were" in words) and "by" in words,
            
            # Strategy 3: Add a starter phrase
            lambda s, t: True  # Always applicable
        ]
        
        # Choose a strategy based on sentence
        applicable_strategies = [i for i, strategy in enumerate(strategies) 
                                if strategy(sentence, tagged)]
        
        if not applicable_strategies:
            return sentence
            
        strategy_idx = random.choice(applicable_strategies)
        
        # Apply the chosen strategy
        if strategy_idx == 0:
            # Move adjective
            for i, (word, tag) in enumerate(tagged[1:4], 1):
                if tag.startswith("JJ"):
                    new_sent = f"This {word} " + " ".join([w for j, (w, _) in enumerate(tagged) 
                                                        if j != i])
                    return new_sent
                    
        elif strategy_idx == 1:
            # Very simplified passive->active conversion
            # This won't work well for many sentences, just an example
            try:
                verb_idx = next(i for i, w in enumerate(words) if w in ["was", "were"])
                by_idx = next(i for i, w in enumerate(words) if w == "by")
                
                if by_idx > verb_idx and by_idx < len(words)-1:
                    subject = " ".join(words[by_idx+1:])
                    action = words[verb_idx+1:by_idx]
                    object_phrase = " ".join(words[:verb_idx])
                    return f"{subject} {' '.join(action)} {object_phrase}"
            except (StopIteration, IndexError):
                pass
                
        elif strategy_idx == 2:
            # Add a starter phrase
            starters = [
                "To summarize, ",
                "In essence, ",
                "Notably, ",
                "Specifically, ",
                "Importantly, ",
                "It's worth mentioning that ",
                "Key point: ",
                "To highlight, "
            ]
            return random.choice(starters) + sentence
                
        return sentence
    except Exception as e:
        print(f"Error in sentence rephrasing: {e}")
        return sentence

def summarize_text_chunk(text, summarizer=None, max_length=150, min_length=40, use_fast_model=True, randomness=False):
    """Summarize a single chunk of text."""
    if not text.strip():
        return ""
    
    # Use extractive summarization if fast model is selected or as fallback
    if use_fast_model:
        # Calculate the target number of sentences based on the min/max length
        target_sentences = max(3, min(10, max_length // 20))
        summary = extractive_summarize(text, num_sentences=target_sentences, randomness=randomness)
        
        # If randomness is enabled, potentially rephrase some sentences
        if randomness:
            try:
                try:
                    from nltk.tokenize import sent_tokenize
                    sentences = sent_tokenize(summary)
                except Exception:
                    sentences = fallback_sent_tokenize(summary)
                    
                new_sentences = []
                
                for sentence in sentences:
                    # 70% chance to rephrase each sentence
                    if random.random() < 0.7:
                        new_sentences.append(rephrase_sentence(sentence))
                    else:
                        new_sentences.append(sentence)
                        
                summary = ' '.join(new_sentences)
            except Exception as e:
                print(f"Error in sentence rephrasing: {e}")
                # Just use the original summary if rephrasing fails
        
        return summary
    
    # Use transformer model if available and preferred
    if summarizer:
        try:
            # Limit text length to prevent model errors
            if len(text) > 1024 * 3:
                text = text[:1024 * 3]
            
            # Generate summary with optional randomness
            if randomness:
                # Add some randomness to model generation
                output = summarizer(text, 
                                   max_length=max_length, 
                                   min_length=min_length, 
                                   do_sample=True,  # Enable sampling
                                   top_p=0.85,      # Use nucleus sampling
                                   temperature=1.2)  # Increase randomness
            else:
                output = summarizer(text, 
                                   max_length=max_length, 
                                   min_length=min_length, 
                                   do_sample=False)
                
            return output[0]['summary_text']
        except Exception as e:
            print(f"Error in transformer summarization: {e}")
            # Fallback to extractive summarization
            return extractive_summarize(text, num_sentences=5, randomness=randomness)
    else:
        # Fallback to extractive summarization
        return extractive_summarize(text, num_sentences=5, randomness=randomness)

def process_chunk_for_summary(chunk, summarizer=None, max_length=150, min_length=40, use_fast_model=True, randomness=False):
    """Process a single chunk for summarization."""
    return summarize_text_chunk(chunk, summarizer, max_length, min_length, use_fast_model, randomness)

def summarize_text_in_chunks(text, summarizer=None, chunk_size=2000, max_length=150, min_length=40, 
                            use_fast_model=True, randomness=False):
    """Break text into chunks and summarize each chunk."""
    # For very large texts, we need to be more aggressive in chunking
    # and summarizing to prevent memory issues and speed up processing
    
    # Estimate text size and adjust approach
    very_large_text = len(text) > 100000  # About 100KB
    
    # Use different chunking strategies based on text size
    if very_large_text:
        # For very large texts, use page markers if available
        if "--- Page" in text:
            chunks = []
            current_chunk = ""
            lines = text.split('\n')
            
            for line in lines:
                if line.startswith("--- Page") and current_chunk:
                    chunks.append(current_chunk)
                    current_chunk = line + "\n"
                else:
                    if len(current_chunk) + len(line) < chunk_size:
                        current_chunk += line + "\n"
                    else:
                        if current_chunk:
                            chunks.append(current_chunk)
                        current_chunk = line + "\n"
            
            if current_chunk:
                chunks.append(current_chunk)
                
        else:
            # Fall back to paragraph chunking for large texts
            paragraphs = text.split('\n\n')
            chunks = []
            current_chunk = ""
            
            for paragraph in paragraphs:
                if len(current_chunk) + len(paragraph) < chunk_size:
                    current_chunk += paragraph + "\n\n"
                else:
                    if current_chunk:
                        chunks.append(current_chunk)
                    current_chunk = paragraph + "\n\n"
            
            if current_chunk:
                chunks.append(current_chunk)
    else:
        # Standard chunking for normal sized texts
        paragraphs = text.split('\n\n')
        chunks = []
        current_chunk = ""
        
        for paragraph in paragraphs:
            if len(current_chunk) + len(paragraph) < chunk_size:
                current_chunk += paragraph + "\n\n"
            else:
                if current_chunk:
                    chunks.append(current_chunk)
                current_chunk = paragraph + "\n\n"
        
        if current_chunk:
            chunks.append(current_chunk)
    
    # Fallback to simple chunking if we still have no proper chunks
    if len(chunks) <= 1 and len(text) > chunk_size:
        chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
    
    # For very large texts, use parallel processing
    if very_large_text and len(chunks) > 4:
        return summarize_chunks_parallel(
            chunks, 
            summarizer=summarizer, 
            max_length=max_length, 
            min_length=min_length,
            use_fast_model=use_fast_model,
            randomness=randomness
        )
    
    # Standard sequential processing for normal texts
    summary_parts = []
    for chunk in chunks:
        if not chunk.strip():
            continue
        summary = summarize_text_chunk(
            chunk, 
            summarizer=summarizer, 
            max_length=max_length, 
            min_length=min_length,
            use_fast_model=use_fast_model,
            randomness=randomness
        )
        if summary:
            summary_parts.append(summary)
    
    # Combine the summaries
    combined_summary = " ".join(summary_parts)
    
    # If the combined summary is still too long, summarize it again
    if len(combined_summary) > chunk_size and len(summary_parts) > 1:
        return summarize_text_chunk(
            combined_summary, 
            summarizer=summarizer, 
            max_length=max_length * 2, 
            min_length=min_length,
            use_fast_model=use_fast_model,
            randomness=randomness
        )
    
    return combined_summary

def summarize_chunks_parallel(chunks, summarizer=None, max_length=150, min_length=40, 
                             use_fast_model=True, randomness=False):
    """Summarize chunks in parallel for faster processing."""
    with ThreadPoolExecutor(max_workers=min(8, len(chunks))) as executor:
        futures = []
        for chunk in chunks:
            if chunk.strip():
                future = executor.submit(
                    process_chunk_for_summary,
                    chunk,
                    summarizer,
                    max_length,
                    min_length, 
                    use_fast_model,
                    randomness
                )
                futures.append(future)
        
        # Collect results
        summary_parts = []
        for future in as_completed(futures):
            summary = future.result()
            if summary:
                summary_parts.append(summary)
    
    # Combine the summaries
    combined_summary = " ".join(summary_parts)
    
    # If the combined summary is still too long, summarize it again
    if len(combined_summary) > 2000 and len(summary_parts) > 1:
        return summarize_text_chunk(
            combined_summary, 
            summarizer=summarizer, 
            max_length=max_length * 2, 
            min_length=min_length,
            use_fast_model=use_fast_model,
            randomness=randomness
        )
    
    return combined_summary

def summarize_multiple_texts(texts, summarizer=None, max_length=150, min_length=40, use_fast_model=True, randomness=False):
    """Summarize multiple texts and combine them efficiently."""
    if not texts:
        return "No text available to summarize."
    
    # Calculate total text size to adjust approach
    total_size = sum(len(text) for text in texts)
    very_large_corpus = total_size > 200000  # About 200KB
    
    # For a single text, just summarize it directly
    if len(texts) == 1:
        return summarize_text_in_chunks(
            texts[0], 
            summarizer=summarizer, 
            max_length=max_length, 
            min_length=min_length,
            use_fast_model=use_fast_model,
            randomness=randomness
        )
    
    # For multiple texts with large corpus, use parallel processing
    if very_large_corpus:
        with ThreadPoolExecutor(max_workers=min(5, len(texts))) as executor:
            futures = []
            for text in texts:
                if text.strip():
                    future = executor.submit(
                        summarize_text_in_chunks,
                        text, 
                        summarizer, 
                        2000,  # chunk_size
                        max_length, 
                        min_length,
                        use_fast_model,
                        randomness
                    )
                    futures.append(future)
            
            # Collect results
            individual_summaries = []
            for future in as_completed(futures):
                summary = future.result()
                if summary:
                    individual_summaries.append(summary)
    else:
        # For moderate-sized corpus, process sequentially
        individual_summaries = []
        for text in texts:
            if not text.strip():
                continue
            summary = summarize_text_in_chunks(
                text, 
                summarizer=summarizer, 
                max_length=max_length, 
                min_length=min_length,
                use_fast_model=use_fast_model,
                randomness=randomness
            )
            if summary:
                individual_summaries.append(summary)
    
    # Extract key phrases from each summary for a better final summary
    all_key_phrases = []
    for summary in individual_summaries:
        key_phrases = extract_key_phrases(summary, num_phrases=3)
        all_key_phrases.extend(key_phrases)
    
    # Add key phrases to the combined text to emphasize important content
    combined_text = "\n\n".join(individual_summaries)
    if all_key_phrases:
        key_phrase_text = "Key points: " + ". ".join(all_key_phrases) + "."
        combined_text = key_phrase_text + "\n\n" + combined_text
    
    # Create a final summary of all the individual summaries
    # For very large corpus, increase the final summary length
    final_max_length = max(250, max_length * 2) if very_large_corpus else max(200, max_length * 1.5)
    final_min_length = min(150, min_length * 2) if very_large_corpus else min(100, min_length * 1.5)
    
    final_summary = summarize_text_chunk(
        combined_text, 
        summarizer=summarizer, 
        max_length=int(final_max_length), 
        min_length=int(final_min_length),
        use_fast_model=use_fast_model,
        randomness=randomness
    )
    
    return final_summary

# Initialize NLTK at module load time
print("Initializing summarizer module...")
ensure_nltk()
print("Summarizer module initialized")
