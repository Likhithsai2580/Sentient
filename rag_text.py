import docx
import PyPDF2
import os
from typing import List, Tuple

def read_text_file(file_path: str) -> str:
    """
    Read and return the content from a plain text file.

    This function opens a text file in read mode with UTF-8 encoding and reads its entire content.

    Args:
        file_path (str): The path to the text file.

    Returns:
        str: The content of the text file as a single string.
    """
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()

def read_pdf_file(file_path: str) -> str:
    """
    Read and return the text content from a PDF file.

    This function uses PyPDF2 library to extract text from each page of the PDF document.

    Args:
        file_path (str): The path to the PDF file.

    Returns:
        str: The extracted text content from the PDF file.
    """
    text = ""
    with open(file_path, 'rb') as file: # Open PDF file in binary read mode
        pdf_reader = PyPDF2.PdfReader(file) # Create a PdfReader object
        for page in pdf_reader.pages: # Iterate over each page in the PDF
            text += page.extract_text() + "\n" # Extract text from the page and append to the text variable
    return text

def read_docx_file(file_path: str) -> str:
    """
    Read and return the text content from a DOCX file.

    This function utilizes the docx library to read text from all paragraphs in a Word document.

    Args:
        file_path (str): The path to the DOCX file.

    Returns:
        str: The extracted text content from the DOCX file, with paragraphs joined by newline characters.
    """
    doc = docx.Document(file_path) # Open the DOCX file
    return "\n".join([paragraph.text for paragraph in doc.paragraphs]) # Extract text from each paragraph and join with newline

def read_document(file_path: str) -> str:
    """
    Read and return the text content from a document file, automatically detecting the file type.

    This function determines the document type based on the file extension and calls the appropriate
    reader function (for .txt, .pdf, or .docx files).

    Args:
        file_path (str): The path to the document file.

    Returns:
        str: The extracted text content from the document file.

    Raises:
        ValueError: If the file extension is not supported.
    """
    _, file_extension = os.path.splitext(file_path) # Split file path into name and extension
    file_extension = file_extension.lower() # Convert extension to lowercase for case-insensitive comparison

    if file_extension == '.txt': # Check if file is a text file
        return read_text_file(file_path) # Read as text file
    elif file_extension == '.pdf': # Check if file is a PDF file
        return read_pdf_file(file_path) # Read as PDF file
    elif file_extension == '.docx': # Check if file is a DOCX file
        return read_docx_file(file_path) # Read as DOCX file
    else:
        raise ValueError(f"Unsupported file format: {file_extension}") # Raise error for unsupported formats


def split_text(text: str, chunk_size: int = 1000) -> List[str]:
    """
    Split a long text into chunks of a specified size, attempting to preserve sentence boundaries.

    This function splits the input text into sentences and then combines sentences into chunks
    such that each chunk is approximately of the given `chunk_size`. It tries to avoid breaking
    sentences in the middle.

    Args:
        text (str): The input text to be split.
        chunk_size (int, optional): The maximum size of each chunk in characters. Defaults to 1000.

    Returns:
        List[str]: A list of text chunks.
    """
    sentences = text.replace('\n', ' ').split('. ') # Replace newlines with spaces and split into sentences
    chunks: List[str] = [] # Initialize list to store text chunks
    current_chunk: List[str] = [] # Initialize list to hold sentences for the current chunk
    current_size = 0 # Initialize current chunk size

    for sentence in sentences: # Iterate through each sentence
        sentence = sentence.strip() # Remove leading/trailing whitespace from sentence
        if not sentence: # Skip empty sentences
            continue

        # Ensure proper sentence ending if it was removed by split
        if not sentence.endswith('.'):
            sentence += '.'

        sentence_size = len(sentence) # Get the size of the current sentence

        # Check if adding this sentence would exceed the chunk size
        if current_size + sentence_size > chunk_size and current_chunk: # If adding sentence exceeds chunk size and current chunk is not empty
            chunks.append(' '.join(current_chunk)) # Join sentences in current chunk and add to chunks list
            current_chunk = [sentence] # Start a new chunk with the current sentence
            current_size = sentence_size # Reset current size to the size of the current sentence
        else:
            current_chunk.append(sentence) # Add sentence to the current chunk
            current_size += sentence_size # Update current chunk size

    # Add the last chunk if it contains any sentences
    if current_chunk: # If there are sentences in the current chunk
        chunks.append(' '.join(current_chunk)) # Join sentences and add the last chunk to the chunks list

    return chunks # Return the list of text chunks

import chromadb
from chromadb.utils import embedding_functions

# Initialize ChromaDB client with persistence, storing database in "chroma_db" directory
client = chromadb.PersistentClient(path="chroma_db")

# Configure sentence transformer embeddings using SentenceTransformerEmbeddingFunction from ChromaDB utils
# This uses the "all-MiniLM-L6-v2" model for generating embeddings
sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)

# Create or get an existing ChromaDB collection named "documents_collection"
# If the collection already exists, it will be retrieved; otherwise, a new one is created.
# The embedding function defined above is associated with this collection.
collection = client.get_or_create_collection(
    name="documents_collection",
    embedding_function=sentence_transformer_ef
)

def process_document(file_path: str) -> Tuple[List[str], List[str], List[dict]]:
    """
    Process a single document file, read its content, split it into chunks, and prepare data for ChromaDB.

    This function reads a document from the given file path, splits its content into text chunks,
    and prepares IDs, text chunks, and metadata for adding to a ChromaDB collection.

    Args:
        file_path (str): The path to the document file to be processed.

    Returns:
        Tuple[List[str], List[str], List[dict]]: A tuple containing:
            - List[str]: A list of unique IDs for each chunk.
            - List[str]: A list of text chunks extracted from the document.
            - List[dict]: A list of metadata dictionaries, each corresponding to a text chunk.
                         Metadata includes the source file name and chunk index.
    """
    try:
        # Read the document content using the appropriate function based on file type
        content = read_document(file_path)

        # Split the document content into smaller text chunks
        chunks = split_text(content)

        # Extract the base file name from the file path
        file_name = os.path.basename(file_path)
        # Create metadata for each chunk, including source file name and chunk index
        metadatas = [{"source": file_name, "chunk": i} for i in range(len(chunks))]
        # Generate unique IDs for each chunk based on file name and chunk index
        ids = [f"{file_name}_chunk_{i}" for i in range(len(chunks))]

        return ids, chunks, metadatas # Return IDs, chunks, and metadata
    except Exception as e:
        print(f"Error processing {file_path}: {str(e)}") # Print error message if processing fails
        return [], [], [] # Return empty lists in case of error

def add_to_collection(collection: chromadb.Collection, ids: List[str], texts: List[str], metadatas: List[dict]) -> None:
    """
    Add batches of documents to a ChromaDB collection.

    This function adds documents (text chunks with associated metadata and IDs) to the specified ChromaDB collection
    in batches to handle potentially large number of documents efficiently.

    Args:
        collection (chromadb.Collection): The ChromaDB collection to add documents to.
        ids (List[str]): A list of unique IDs for each document.
        texts (List[str]): A list of document texts.
        metadatas (List[dict]): A list of metadata dictionaries, each corresponding to a document.
    """
    if not texts: # If there are no texts to add, return immediately
        return

    batch_size = 100 # Define batch size for adding documents to the collection
    for i in range(0, len(texts), batch_size): # Iterate through texts in batches
        end_idx = min(i + batch_size, len(texts)) # Calculate the end index for the current batch
        collection.add( # Add a batch of documents to the ChromaDB collection
            documents=texts[i:end_idx], # List of text chunks for the current batch
            metadatas=metadatas[i:end_idx], # List of metadata for the current batch
            ids=ids[i:end_idx] # List of IDs for the current batch
        )

def process_and_add_documents(collection: chromadb.Collection, folder_path: str) -> None:
    """
    Process all document files in a specified folder and add them to a ChromaDB collection.

    This function iterates through all files in the given folder, checks if they are files (not directories),
    processes each document file using `process_document` function, and adds the processed chunks to the
    ChromaDB collection using `add_to_collection` function.

    Args:
        collection (chromadb.Collection): The ChromaDB collection to add documents to.
        folder_path (str): The path to the folder containing document files.
    """
    files = [os.path.join(folder_path, file)
             for file in os.listdir(folder_path)
             if os.path.isfile(os.path.join(folder_path, file))] # List all files in the folder

    for file_path in files: # Iterate through each file path
        print(f"Processing {os.path.basename(file_path)}...") # Print message indicating file processing start
        ids, texts, metadatas = process_document(file_path) # Process the document file
        add_to_collection(collection, ids, texts, metadatas) # Add processed chunks to the collection
        print(f"Added {len(texts)} chunks to collection") # Print message indicating number of chunks added


# Initialize ChromaDB collection (we'll cover this in detail in the next section)
collection = client.get_or_create_collection(
    name="documents_collection",
    embedding_function=sentence_transformer_ef
)

# Define the folder path where document files are stored
folder_path = "docs"
# Process all documents in the specified folder and add them to the ChromaDB collection
process_and_add_documents(collection, folder_path)


def semantic_search(collection: chromadb.Collection, query: str, n_results: int = 10) -> dict:
    """
    Perform a semantic search on a ChromaDB collection.

    This function queries the ChromaDB collection using the provided query text and retrieves documents
    that are semantically similar to the query.

    Args:
        collection (chromadb.Collection): The ChromaDB collection to search in.
        query (str): The query text for semantic search.
        n_results (int, optional): The number of search results to return. Defaults to 10.

    Returns:
        dict: A dictionary containing the search results from ChromaDB.
              The structure of this dictionary is defined by ChromaDB's query response format.
    """
    results = collection.query( # Perform query on the ChromaDB collection
        query_texts=[query], # The query text as a list
        n_results=n_results # Number of results to retrieve
    )
    return results # Return the query results

def get_context_with_sources(results: dict) -> Tuple[str, List[str]]:
    """
    Extract context and source information from ChromaDB search results.

    This function processes the results from a semantic search performed on ChromaDB,
    extracts the document text chunks to form a combined context, and formats source information
    (file name and chunk index) for each retrieved chunk.

    Args:
        results (dict): The dictionary of search results returned by ChromaDB's query function.

    Returns:
        Tuple[str, List[str]]: A tuple containing:
            - str: A combined context string formed by joining all retrieved document chunks with newline separators.
            - List[str]: A list of source information strings, each indicating the source file and chunk index.
    """
    # Combine document chunks into a single context string, separated by newlines
    context = "\n\n".join(results['documents'][0])

    # Format source information for each retrieved document chunk using metadata
    sources = [
        f"{meta['source']} (chunk {meta['chunk']})"
        for meta in results['metadatas'][0]
    ]

    return context, sources # Return the combined context and list of sources

import ollama

def query_llama3(context: str, user_query: str) -> str:
    """
    Use the LLaMA 3 model from Ollama to generate a response based on a given context and user query.

    This function sends a prompt to the Ollama API with a context and a user query, instructing LLaMA 3
    to act as an AI assistant and answer the query based on the provided context.

    Args:
        context (str): The context text retrieved from document search, used to inform the AI's answer.
        user_query (str): The user's question or query.

    Returns:
        str: The text response generated by the LLaMA 3 model.
    """
    prompt = f"""You are an AI assistant answering questions based on provided document context.

    Context:
    {context}

    User Query:
    {user_query}

    Answer:
    """
    response = ollama.chat(model='llama3.2:3b', messages=[{"role": "user", "content": prompt}]) # Send prompt to Ollama LLaMA 3 model
    return response['message']['content'] # Return the content of the model's message

# Perform a semantic search in the document collection
query = "What is the Experience in the Resume?"
results = semantic_search(collection, query)

# Extract the context and source information from the search results
context, sources = get_context_with_sources(results)

print("Context from Document Store:\n", context)

# Generate a response using LLaMA 3, based on the retrieved context and the user query
response = query_llama3(context, query)

# Print the AI generated response and the sources of the context
print("\nAI Response:\n", response)
print("\nSources:", sources)