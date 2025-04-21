import tempfile
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from PyPDF2 import PdfReader
import time
import math

def extract_text_from_pdf_bytes(pdf_bytes, max_pages=400, sampling_strategy=None):
    """
    Extract text from PDF bytes directly without saving to disk.
    
    Args:
        pdf_bytes: The PDF file bytes
        max_pages: Maximum number of pages to process
        sampling_strategy: Strategy for sampling pages from large documents
                          None: Process all pages up to max_pages
                          'uniform': Sample pages uniformly throughout the document
                          'front_heavy': Process more pages from the beginning
                          'bookend': Process beginning and end more heavily
    """
    try:
        reader = PdfReader(pdf_bytes)
        num_pages = len(reader.pages)
        actual_pages_to_process = min(max_pages, num_pages)
        
        # Determine which pages to process based on strategy
        pages_to_process = []
        
        if num_pages > max_pages and sampling_strategy:
            if sampling_strategy == 'uniform':
                # Uniform sampling throughout the document
                if max_pages > 1:
                    step = num_pages / max_pages
                    pages_to_process = [int(i * step) for i in range(max_pages)]
                    # Ensure we include the first and last page
                    if 0 not in pages_to_process:
                        pages_to_process[0] = 0
                    if num_pages - 1 not in pages_to_process:
                        pages_to_process[-1] = num_pages - 1
                else:
                    pages_to_process = [0]  # Just the first page
                    
            elif sampling_strategy == 'front_heavy':
                # Sample more from the beginning, less from the end
                # 60% from first third, 30% from middle, 10% from last third
                first_third = int(max_pages * 0.6)
                middle_third = int(max_pages * 0.3)
                last_third = max_pages - first_third - middle_third
                
                first_segment = num_pages // 3
                second_segment = 2 * (num_pages // 3)
                
                # Select pages from each segment
                if first_third > 0:
                    step1 = first_segment / first_third
                    segment1 = [int(i * step1) for i in range(first_third)]
                else:
                    segment1 = []
                    
                if middle_third > 0:
                    step2 = (second_segment - first_segment) / middle_third
                    segment2 = [int(first_segment + i * step2) for i in range(middle_third)]
                else:
                    segment2 = []
                    
                if last_third > 0:
                    step3 = (num_pages - second_segment) / last_third
                    segment3 = [int(second_segment + i * step3) for i in range(last_third)]
                else:
                    segment3 = []
                
                pages_to_process = segment1 + segment2 + segment3
                
                # Ensure we include the first and last page
                if 0 not in pages_to_process:
                    pages_to_process[0] = 0
                if num_pages - 1 not in pages_to_process:
                    pages_to_process[-1] = num_pages - 1
                    
            elif sampling_strategy == 'bookend':
                # Focus on beginning and end of document
                # 40% from beginning, 20% from middle, 40% from end
                begin_pages = int(max_pages * 0.4)
                middle_pages = int(max_pages * 0.2)
                end_pages = max_pages - begin_pages - middle_pages
                
                # Beginning pages
                if begin_pages > 0:
                    begin_step = (num_pages // 4) / begin_pages
                    begin_segment = [int(i * begin_step) for i in range(begin_pages)]
                else:
                    begin_segment = []
                
                # Middle pages
                if middle_pages > 0:
                    middle_start = num_pages // 3
                    middle_end = 2 * (num_pages // 3)
                    middle_step = (middle_end - middle_start) / middle_pages
                    middle_segment = [int(middle_start + i * middle_step) for i in range(middle_pages)]
                else:
                    middle_segment = []
                
                # End pages
                if end_pages > 0:
                    end_start = 3 * (num_pages // 4)
                    end_step = (num_pages - end_start) / end_pages
                    end_segment = [int(end_start + i * end_step) for i in range(end_pages)]
                else:
                    end_segment = []
                
                pages_to_process = begin_segment + middle_segment + end_segment
                
                # Ensure we include the first and last page
                if 0 not in pages_to_process:
                    pages_to_process.insert(0, 0)
                if num_pages - 1 not in pages_to_process:
                    pages_to_process.append(num_pages - 1)
            
            # Remove any duplicates and sort
            pages_to_process = sorted(list(set(pages_to_process)))
            
        else:
            # Default: process pages in order up to max_pages
            pages_to_process = list(range(min(max_pages, num_pages)))
        
        # Extract text from selected pages
        text = ""
        for page_num in pages_to_process:
            page = reader.pages[page_num]
            page_text = page.extract_text()
            if page_text:
                # Add page number for reference
                text += f"\n--- Page {page_num + 1} ---\n"
                text += page_text + "\n"
        
        return text, num_pages, len(pages_to_process)
    except Exception as e:
        print(f"Error extracting text from PDF: {e}")
        return "", 0, 0

def process_pdf_chunk(pdf_file, chunk_index, chunk_size, max_pages, sampling_strategy):
    """Process a chunk of pages from a PDF file."""
    try:
        reader = PdfReader(pdf_file)
        total_pages = len(reader.pages)
        
        # Calculate which pages to process in this chunk
        start_page = chunk_index * chunk_size
        end_page = min(start_page + chunk_size, total_pages, max_pages)
        
        # Skip if this chunk is beyond the max pages
        if start_page >= max_pages:
            return ""
            
        # Determine which pages within this chunk to process based on strategy
        pages_to_process = list(range(start_page, end_page))
        if sampling_strategy and sampling_strategy != 'none':
            # Implement chunk-based sampling if needed
            pass  # For now, process all pages in the chunk
        
        # Extract text from the pages
        text = ""
        for page_num in pages_to_process:
            if page_num < total_pages:  # Safety check
                page = reader.pages[page_num]
                page_text = page.extract_text()
                if page_text:
                    text += f"\n--- Page {page_num + 1} ---\n" + page_text + "\n"
        
        return text
    except Exception as e:
        print(f"Error processing PDF chunk {chunk_index}: {e}")
        return ""

def extract_text_from_pdf_parallel(pdf_file, max_pages=400, workers=4, sampling_strategy=None):
    """Extract text from a PDF using parallel processing for large documents."""
    try:
        # Quick check of total pages
        reader = PdfReader(pdf_file)
        total_pages = len(reader.pages)
        
        # For small documents, use the regular method
        if total_pages <= 50 or workers <= 1:
            return extract_text_from_pdf_bytes(pdf_file, max_pages, sampling_strategy)
        
        # For large documents with sampling strategy, use the specialized method
        if sampling_strategy:
            return extract_text_from_pdf_bytes(pdf_file, max_pages, sampling_strategy)
            
        # For large documents without sampling, use parallel processing
        pages_to_process = min(max_pages, total_pages)
        chunk_size = math.ceil(pages_to_process / workers)
        num_chunks = math.ceil(pages_to_process / chunk_size)
        
        # Reset file pointer if needed
        if hasattr(pdf_file, 'seek'):
            pdf_file.seek(0)
        
        # Process chunks in parallel
        chunks = []
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [
                executor.submit(
                    process_pdf_chunk, 
                    pdf_file, 
                    i, 
                    chunk_size,
                    max_pages,
                    sampling_strategy
                )
                for i in range(num_chunks)
            ]
            
            for future in as_completed(futures):
                chunk_text = future.result()
                if chunk_text:
                    chunks.append(chunk_text)
        
        # Combine all chunks
        text = "".join(chunks)
        return text, total_pages, len(chunks) * chunk_size
    except Exception as e:
        print(f"Error in parallel PDF processing: {e}")
        # Fallback to regular processing
        return extract_text_from_pdf_bytes(pdf_file, max_pages, sampling_strategy)

def extract_text_from_multiple_pdfs(pdf_files, max_pages=400, parallel=True, sampling_strategy='bookend'):
    """Extract text from multiple PDF files efficiently."""
    if parallel and len(pdf_files) > 1:
        # Use parallel processing for multiple PDFs
        with ThreadPoolExecutor(max_workers=min(5, len(pdf_files))) as executor:
            futures = [
                executor.submit(
                    extract_text_from_pdf_parallel, 
                    pdf, 
                    max_pages,
                    workers=2,  # Use 2 workers per PDF to avoid overloading
                    sampling_strategy=sampling_strategy
                )
                for pdf in pdf_files
            ]
            
            results = []
            for future in as_completed(futures):
                results.append(future.result())
                
        return [result[0] for result in results], [result[1] for result in results], [result[2] for result in results]
    else:
        # Sequential processing
        texts = []
        total_pages = []
        processed_pages = []
        
        for pdf_file in pdf_files:
            text, num_pages, pages_processed = extract_text_from_pdf_parallel(
                pdf_file, 
                max_pages, 
                workers=4,  # Use more workers for single PDF processing
                sampling_strategy=sampling_strategy
            )
            texts.append(text)
            total_pages.append(num_pages)
            processed_pages.append(pages_processed)
            
        return texts, total_pages, processed_pages

def save_uploaded_file(uploaded_file):
    """Save an uploaded file to a temporary file and return the path."""
    try:
        # Create a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp:
            # Write the contents of the uploaded file to the temporary file
            tmp.write(uploaded_file.read())
            # Return the path of the temporary file
            return tmp.name
    except Exception as e:
        print(f"Error saving uploaded file: {e}")
        return None
    
def clean_temp_files(file_paths):
    """Remove temporary files."""
    for file_path in file_paths:
        try:
            if file_path and os.path.exists(file_path):
                os.remove(file_path)
        except Exception as e:
            print(f"Error removing temporary file {file_path}: {e}")