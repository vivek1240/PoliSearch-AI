from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os
import hashlib
from elasticsearch import Elasticsearch
from llama_index.vector_stores.elasticsearch import ElasticsearchStore, AsyncDenseVectorStrategy
from llama_index.core import StorageContext, VectorStoreIndex
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
from llama_parse import LlamaParse
from llama_index.core.schema import TextNode
import logging
from typing import List
from rag_eval import evaluate_faithfulness, evaluate_answer_relevance, evaluate_context_relevance


# Set up API keys (environment variables)
def set_api_keys(config):
    """
    Set the API keys for OpenAI and Llama Cloud from the provided configuration.

    Args:
        config (dict): Dictionary containing the API keys for OpenAI and Llama Cloud.
    """
    logging.info("Setting up API keys.")
    os.environ["OPENAI_API_KEY"] = config["openai_api_key"]
    os.environ["LLAMA_CLOUD_API_KEY"] = config["llama_cloud_api_key"]
    logging.info("API keys set.")


# Set up Elasticsearch connection
def connect_elasticsearch(es_url):
    """
    Connect to an Elasticsearch instance.

    Args:
        es_url (str): The URL of the Elasticsearch instance.

    Returns:
        Elasticsearch: Elasticsearch client instance.

    Raises:
        Exception: If connection to Elasticsearch fails.
    """
    logging.info(f"Connecting to Elasticsearch at {es_url}.")
    try:
        es = Elasticsearch(es_url)
        logging.info("Connected to Elasticsearch.")
        return es
    except Exception as e:
        logging.error(f"Failed to connect to Elasticsearch: {str(e)}")
        raise e

# Create Elasticsearch index if it doesn't exist
def create_index_if_not_exists(es, index_name):
    """
    Create an Elasticsearch index if it doesn't exist.

    Args:
        es (Elasticsearch): Elasticsearch client instance.
        index_name (str): Name of the index to be created.

    Returns:
        None
    """
    logging.info(f"Checking if Elasticsearch index '{index_name}' exists.")
    if not es.indices.exists(index=index_name):
        mappings = {
            "mappings": {
                "properties": {
                    "hash": {"type": "keyword"},
                    "title": {"type": "text"},
                    "source": {"type": "text"}
                }
            }
        }
        es.indices.create(index=index_name, body=mappings)
        logging.info(f"Index '{index_name}' created.")
        print(f"Index '{index_name}' created.")
    else:
        print(f"Index '{index_name}' already exists.")
        logging.info(f"Index '{index_name}' already exists.")

# Get PDF files from the directory
def get_data_files(data_dir):
    """
    Retrieve all PDF files from a specified directory.

    Args:
        data_dir (str): Directory containing the PDF files.

    Returns:
        list: List of paths to PDF files.
    """
    logging.info(f"Retrieving PDF files from directory: {data_dir}.")
    return [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith(".pdf")]

#################### Ingestion ######################
def get_file_hash(file_path):
    """
    Calculate the SHA-256 hash of a file.

    Args:
        file_path (str): Path to the file.

    Returns:
        str: SHA-256 hash of the file.
    """
    logging.info(f"Calculating hash for file: {file_path}.")
    hash_func = hashlib.sha256()
    with open(file_path, 'rb') as f:
        while chunk := f.read(8192):
            hash_func.update(chunk)
    return hash_func.hexdigest()

# Check if a file with the given hash exists in Elasticsearch
def file_exists(es, file_hash, index_name):
    """
    Check if a file with a specific hash exists in an Elasticsearch index.

    Args:
        es (Elasticsearch): Elasticsearch client instance.
        file_hash (str): SHA-256 hash of the file.
        index_name (str): Name of the Elasticsearch index.

    Returns:
        bool: True if the file exists, False otherwise.
    """
    logging.info(f"Checking if file with hash '{file_hash}' exists in index '{index_name}'.")
    query = {"query": {"term": {"hash.keyword": file_hash}}}
    response = es.search(index=index_name, body=query)
    return response['hits']['total']['value'] > 0

# Parse PDFs using LlamaParse
def parse_pdf(file_path):
    """
    Parse a PDF file using LlamaParse.

    Args:
        file_path (str): Path to the PDF file.

    Returns:
        list: List of parsed documents in markdown format.
    """
    logging.info(f"Parsing PDF file: {file_path}.")
    parser = LlamaParse(
        result_type="markdown",
        parsing_instruction="These are policy documents.",
        use_vendor_multimodal_model=True,
        vendor_multimodal_model_name="openai-gpt4o",
    )
    return parser.load_data([file_path], extra_info={"source": "policy_documents"})

# Ingest documents into the vector store
def ingest_documents(es, index_name, files):
    """
    Ingest parsed PDF documents into Elasticsearch and Llama index.

    Args:
        es (Elasticsearch): Elasticsearch client instance.
        index_name (str): Name of the Elasticsearch index.
        files (list): List of file paths to be ingested.

    Returns:
        list: List of TextNode objects representing the ingested documents.
    """
    logging.info("Ingesting documents into Elasticsearch and Llama index.")
    nodes = []
    for i, file_path in enumerate(files):
        file_name = os.path.basename(file_path)
        file_hash = get_file_hash(file_path)

        if file_exists(es, file_hash, index_name):
            print(f"File '{file_name}' is already indexed. Skipping...")
            continue

        documents = parse_pdf(file_path)
        for doc in documents:
            node = TextNode(
                text=doc.text,
                metadata={
                    "source": "policy_document",
                    "title": file_name,
                    "hash": file_hash
                }
            )
            nodes.append(node)
    return nodes

########################################################################
# Set up vector store for retrieval
def setup_vector_store(es_url, index_name):
    """
    Set up an Elasticsearch vector store for dense retrieval.

    Args:
        es_url (str): The URL of the Elasticsearch instance.
        index_name (str): Name of the index for the vector store.

    Returns:
        ElasticsearchStore: Configured Elasticsearch vector store.
    
    Raises:
        Exception: If there is an error during vector store setup.
    """
    logging.info("Setting up Elasticsearch vector store.")
    try:
        vector_store = ElasticsearchStore(
            index_name=index_name,
            es_url=es_url,
            retrieval_strategy=AsyncDenseVectorStrategy(),
        )
        logging.info("Elasticsearch vector store set up successfully.")
        return vector_store
    except Exception as e:
        logging.error(f"Error setting up vector store: {str(e)}")
        raise e

# Set up Llama index for embeddings
def setup_llama_index(nodes, vector_store):
    """
    Set up the LlamaIndex for storing and retrieving document embeddings.

    Args:
        nodes (list): List of TextNode objects representing the documents.
        vector_store (ElasticsearchStore): Configured Elasticsearch vector store.

    Returns:
        VectorStoreIndex: Llama index configured with the provided documents and vector store.

    Raises:
        Exception: If there is an error during Llama index setup.
    """
    logging.info("Setting up Llama index.")
    try:
        embed_model = OpenAIEmbedding(model="text-embedding-ada-002")
        llm = OpenAI("gpt-4o")
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        index = VectorStoreIndex(nodes, storage_context=storage_context)
        logging.info("Llama index set up successfully.")
        return index
    except Exception as e:
        logging.error(f"Error setting up Llama index: {str(e)}")
        raise e


# Save the index for future use
def save_index(index):
    """
    Save the Llama index to persistent storage for future use.

    Args:
        index (VectorStoreIndex): The Llama index to be saved.
    """
    logging.info("Saving the index for future use.")
    index.storage_context.persist()

# Search the indexed documents and return results along with the answer
def search_index(index, query):
    """
    Search the Llama index for a given query.

    Args:
        index (VectorStoreIndex): The Llama index to search.
        query (str): Query string to search for.

    Returns:
        tuple: A tuple containing the search response and relevant document chunks.
    """
    
    # Retrieve relevant documents
    logging.info(f"Searching index for query: '{query}'.")
    retriever = index.as_retriever(top_k=5)
    results = retriever.retrieve(query)
    
    # Now use the query engine to generate the answer from the chunks
    query_engine = index.as_query_engine()
    
    # Generate the response (answer)
    response = query_engine.query(query)
    
    # Collect the chunks and the answer
    num_of_chunks = len(results)
    answer = str(response)
    retrieved_chunks = [result.get_text() for result in results]
    # retrieved_chunks = [[result.get_text()] for result in results]

    logging.info(f"Search complete. Found {len(retrieved_chunks)} relevant chunks.")
    return answer, retrieved_chunks