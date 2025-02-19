from markitdown import MarkItDown
from uuid import uuid4
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from transformers import AutoTokenizer, AutoModel
import torch


load_dotenv()


# Pinecone configuration
PINECONE_API_KEY = "pcsk_2zyKeU_2ZvxUD78Cxg1eRnooJXDYj1XBmsSN1qn8eDng6QirxtPQNgmm3UiFFtGktcBamq"
PINECONE_ENV = "us-east-1"  # Change to the appropriate environment region
PINECONE_INDEX_NAME = "index10"  # Specify the index name
DIMENSION = 3072  # Change according to your embeddings model dimension
DATA_PATH = "data"  # Path to the directory containing your PDF files

# Initialize the MarkItDown converter
md_converter = MarkItDown()

# Function to convert a document to Markdown
def convert_to_markdown(file_path):
    try:
        result = md_converter.convert(file_path)
        return result.text_content
    except Exception as e:
        print(f"Error converting {file_path}: {e}")
        return None

# Function to split text into chunks
def split_text_into_chunks(text, chunk_size=1000, chunk_overlap=100, length_function=len):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap, length_function=length_function)
    chunks = text_splitter.split_text(text)
    return chunks

# Initialize Hugging Face model and tokenizer for embeddings
model_name = 'sentence-transformers/all-MiniLM-L6-v2'  # Example model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

# Function to get Hugging Face embeddings
def get_huggingface_embeddings(text):
    # Tokenize the input text
    inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True)
    
    # Generate embeddings using the model
    with torch.no_grad():
        outputs = model(**inputs)
    
    # The embeddings are typically the last hidden state
    embeddings = outputs.last_hidden_state.mean(dim=1)  # Average pooling across tokens
    
    return embeddings.squeeze().numpy()  # Convert the embeddings to a numpy array

# Initialize Pinecone using the updated method
pc = Pinecone(api_key=PINECONE_API_KEY)

# Create the Pinecone index (if it does not exist already)
if PINECONE_INDEX_NAME not in pc.list_indexes():
    pc.create_index(
        name=PINECONE_INDEX_NAME,
        dimension=DIMENSION,
        metric="cosine",  # You can also use 'euclidean' or 'dotproduct'
        spec=ServerlessSpec(
            cloud="aws",
            region=PINECONE_ENV
        )
    )
    
# Connect to the index
index = pc.Index(PINECONE_INDEX_NAME)

# Directory containing your documents
docs_directory = "./data"

# Supported file extensions
supported_extensions = ('.pdf', '.docx', '.xlsx', '.txt', '.pptx', '.csv', '.jpg', '.jpeg', '.png', '.mp3', '.html', '.json', '.xml', '.zip')

# Process each file in the directory
for filename in os.listdir(docs_directory):
    if filename.lower().endswith(supported_extensions):
        file_path = os.path.join(docs_directory, filename)
        print(f"Processing {file_path}...")

        # Convert document to Markdown
        markdown_text = convert_to_markdown(file_path)
        if markdown_text:
            # Split the Markdown text into chunks
            chunks = split_text_into_chunks(markdown_text)
            print(f"Document {filename} split into {len(chunks)} chunks.")
            # You can now use these chunks for your RAG pipeline
            # Create unique IDs for each chunk and generate embeddings
            uuids = [str(uuid4()) for _ in range(len(chunks))]
            vectors = [(uuid, get_huggingface_embeddings(doc.page_content)) for uuid, doc in zip(uuids, chunks)]

            # Insert vectors into Pinecone
            index.upsert(vectors=vectors)
            print(f"Inserted {len(vectors)} vectors into Pinecone.")