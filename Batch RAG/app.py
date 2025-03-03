
import os
import docx
import PyPDF2
import chromadb
from chromadb.utils import embedding_functions
import ollama
from rag_audio import *
from rag_image import *

# Initialize ChromaDB client with persistence
client = chromadb.PersistentClient(path="chroma_db")

# Configure sentence transformer embeddings
sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)

# Create or get existing collection
collection = client.get_or_create_collection(
    name="documents_collection",
    embedding_function=sentence_transformer_ef
)

def read_text_file(file_path: str):
    """Read content from a text file"""
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()

def read_pdf_file(file_path: str):
    """Read content from a PDF file"""
    text = ""
    with open(file_path, 'rb') as file:
        pdf_reader = PyPDF2.PdfReader(file)
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"
    return text

def read_docx_file(file_path: str):
    """Read content from a Word document"""
    doc = docx.Document(file_path)
    return "\n".join([paragraph.text for paragraph in doc.paragraphs])

def read_audio_file(file_path: str):
    """Read content from an audio file"""
    text = extract_text_from_audio(
            file_path, # Path to the audio/video file (replace with your file)
            language="en-US", # Language for speech recognition (English - US)
            chunk_size=60000  # Chunk size of 60 seconds (set to 0 or None to use silence-based splitting)
        )
    
    return text

def  read_image_file(file_path: str):
    """Read content from an image file"""
    try:
        reader = AdvancedSceneTextReader() # Initialize the AdvancedSceneTextReader

        # Extract text from the image using the reader
        extracted_text = reader.extract_text(file_path)

        # Check if any text was extracted and print the result
        if not extracted_text:
            print("No text was extracted from the image.") # Inform user if no text was extracted
        else:
            return extracted_text

    except Exception as e:
        print(f"Error: {str(e)}") # Print any error that occurred during the process

def read_document(file_path: str):
    """Read document content based on file extension"""
    _, file_extension = os.path.splitext(file_path)
    file_extension = file_extension.lower()

    if file_extension == '.txt':
        return read_text_file(file_path)
    elif file_extension == '.pdf':
        return read_pdf_file(file_path)
    elif file_extension == '.docx':
        return read_docx_file(file_path)
    elif file_extension == '.mp4' or file_extension == '.mp3' or file_extension == '.wav':
        return read_audio_file(file_path)
    elif file_extension == '.png' or file_extension == '.jpg' or file_extension == '.jpeg':
        return read_image_file(file_path)
    else:
        raise ValueError(f"Unsupported file format: {file_extension}")

def split_text(text: str, chunk_size: int = 1000):
    """Split text into chunks while preserving sentence boundaries"""
    sentences = text.replace('\n', ' ').split('. ')
    chunks = []
    current_chunk = []
    current_size = 0

    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue

        # Ensure proper sentence ending
        if not sentence.endswith('.'):
            sentence += '.'

        sentence_size = len(sentence)

        # Check if adding this sentence would exceed chunk size
        if current_size + sentence_size > chunk_size and current_chunk:
            chunks.append(' '.join(current_chunk))
            current_chunk = [sentence]
            current_size = sentence_size
        else:
            current_chunk.append(sentence)
            current_size += sentence_size

    # Add the last chunk if it exists
    if current_chunk:
        chunks.append(' '.join(current_chunk))

    return chunks

def process_document(file_path: str):
    """Process a single document and prepare it for ChromaDB"""
    try:
        # Read the document
        content = read_document(file_path)

        print(content)

        # Split into chunks
        chunks = split_text(content)

        # Prepare metadata
        file_name = os.path.basename(file_path)
        metadatas = [{"source": file_name, "chunk": i} for i in range(len(chunks))]
        ids = [f"{file_name}_chunk_{i}" for i in range(len(chunks))]

        return ids, chunks, metadatas
    except Exception as e:
        print(f"Error processing {file_path}: {str(e)}")
        return [], [], []

def add_to_collection(ids, texts, metadatas):
    """Add documents to collection in batches"""
    if not texts:
        return

    batch_size = 100
    for i in range(0, len(texts), batch_size):
        end_idx = min(i + batch_size, len(texts))
        collection.add(
            documents=texts[i:end_idx],
            metadatas=metadatas[i:end_idx],
            ids=ids[i:end_idx]
        )
def query_llama3(context: str, user_query: str):
    """Use LLaMA 3 from Ollama to generate a response based on retrieved context."""
    prompt = f"""You are an AI assistant answering questions based on provided document context.
    
    Context:
    {context}
    
    User Query:
    {user_query}
    
    Answer:
    """
    try:
        response = ollama.chat(model='llama3.2:3b', messages=[{"role": "user", "content": prompt}])
        if 'message' in response and 'content' in response['message']:
            return response['message']['content']
        else:
            return "Error: Response format unexpected."
    except Exception as e:
        return f"Error generating response: {str(e)}"

def semantic_search(query: str, n_results: int = 10):
    """Perform semantic search on the collection."""
    results = collection.query(query_texts=[query], n_results=n_results)
    if not results or 'documents' not in results or not results['documents'][0]:
        return None
    return results

def get_context_with_sources(results):
    """Extract context and source information from search results."""
    if results is None:
        return "", []
    
    context = "\n\n".join(results['documents'][0])
    sources = [
        f"{meta['source']} (chunk {meta['chunk']})"
        for meta in results['metadatas'][0]
    ]
    return context, sources

def process_files_from_folder(folder_path):
    """Process all supported files from a folder"""
    supported_extensions = ['.txt', '.pdf', '.docx', '.mp4', '.mp3','.wav', '.png', '.jpg', '.jpeg']
    processed_count = 0
    
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if os.path.isfile(file_path):
            _, extension = os.path.splitext(filename)
            if extension.lower() in supported_extensions:
                print(f"Processing {filename}...")
                ids, texts, metadatas = process_document(file_path)
                add_to_collection(ids, texts, metadatas)
                processed_count += 1
    
    return processed_count

def main():
    """Main function to run the script"""
    while True:
        print("\n===== Document RAG Terminal =====")
        print("1. Process a single document")
        print("2. Process all documents in a folder")
        print("3. Ask a question")
        print("4. Exit")
        choice = input("Enter your choice (1-4): ")
        
        if choice == '1':
            file_path = input("Enter document path: ")
            if os.path.exists(file_path):
                ids, texts, metadatas = process_document(file_path)
                add_to_collection(ids, texts, metadatas)
                print(f"Processed document: {os.path.basename(file_path)}")
                print(f"Added {len(texts)} chunks to the collection")
            else:
                print("File not found!")
                
        elif choice == '2':
            folder_path = input("Enter folder path: ")
            if os.path.isdir(folder_path):
                count = process_files_from_folder(folder_path)
                print(f"Processed {count} documents from {folder_path}")
            else:
                print("Folder not found!")
                
        elif choice == '3':
            query = input("Enter your question: ")
            results = semantic_search(query)
            context, sources = get_context_with_sources(results)
            
            if not context:
                print("No relevant documents found.")
                continue
            
            print("\nGenerating response...")
            response = query_llama3(context, query)
            
            print("\n===== Response =====")
            print(response)
            print("\n===== Sources =====")
            for source in sources:
                print(f"- {source}")
                
        elif choice == '4':
            print("Exiting...")
            break
            
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main()