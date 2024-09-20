# from fastapi import FastAPI, HTTPException
# from pydantic import BaseModel
# import uvicorn
# import nest_asyncio
# import os
# import hashlib
# from elasticsearch import Elasticsearch
# from llama_index.vector_stores.elasticsearch import ElasticsearchStore, AsyncDenseVectorStrategy
# from llama_index.core import StorageContext, VectorStoreIndex
# from llama_index.embeddings.openai import OpenAIEmbedding
# from llama_index.llms.openai import OpenAI
# from llama_parse import LlamaParse
# from llama_index.core.schema import TextNode
# import sys

# # Set up logging configuration
# import logging
# from pydantic import BaseModel
# from typing import List
# from rag_eval import evaluate_faithfulness, evaluate_answer_relevance, evaluate_context_relevance, evaluate_rag_system
# from utils import set_api_keys, connect_elasticsearch, create_index_if_not_exists, get_data_files, ingest_documents, setup_vector_store, setup_llama_index, save_index, search_index

import hashlib
import logging
import os
import sys

import nest_asyncio
import uvicorn
from elasticsearch import Elasticsearch
from fastapi import FastAPI, HTTPException
from llama_index.core import StorageContext, VectorStoreIndex
from llama_index.core.schema import TextNode
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
from llama_index.vector_stores.elasticsearch import AsyncDenseVectorStrategy, ElasticsearchStore
from llama_parse import LlamaParse
from pydantic import BaseModel
from rag_eval import (evaluate_answer_relevance, evaluate_context_relevance,
                      evaluate_faithfulness, evaluate_rag_system)
from typing import List
from utils import (connect_elasticsearch, create_index_if_not_exists,
                   get_data_files, ingest_documents, save_index, search_index,
                   set_api_keys, setup_llama_index, setup_vector_store)


# Set up logging configuration to log both to a file and the console
logging.basicConfig(
    level=logging.INFO,  # Set the logging level
    format='%(asctime)s - %(levelname)s - %(message)s',  # Log format
    handlers=[
        logging.FileHandler("evaluation_system.log"),  # Log to a file named evaluation_system.log
        logging.StreamHandler()  # Also log to console
    ]
)
nest_asyncio.apply()

# FastAPI app instance
app = FastAPI()

# Configuration
CONFIG = {
    "openai_api_key": os.getenv("OPENAI_API_KEY"),
    "llama_cloud_api_key": os.getenv("LLAMA_CLOUD_API_KEY"),
    "es_url": os.getenv("ES_URL", "http://localhost:9200"),
    "es_url": os.getenv("ES_URL", "http://localhost:9200"),
    "index_name": "policy_documents_dense",
    "data_dir": "/app/data"  # Inside the container

}

# Pydantic model for question input
class QuestionInput(BaseModel):
    question: str

########################### Ask Question  ###########################
# Endpoint to ask a question and get an answer from the ingested documents
@app.post("/ask_question")
async def ask_question(input_data: QuestionInput):
    logging.info(f"Received query: {input_data.question}.")
    query = input_data.question
    
    # Load the saved index (you can modify this if needed)
    try:
        vector_store = setup_vector_store(CONFIG['es_url'], CONFIG['index_name'])
        index = setup_llama_index([], vector_store)  # Reinitialize the index with stored vector store
    except Exception as e:
        logging.error(f"Error initializing vector store or index: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error initializing index: {e}")

    # Search the index and get the answer and relevant chunks
    try:
        answer, chunks = search_index(index, query)
        return {
            "question": query,
            "answer": answer,
            "relevant_chunks": chunks
        }
    except Exception as e:
        logging.error(f"Error searching index: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error searching index: {e}")

########################### Eval RAG  ###########################
# Pydantic model for evaluating questions and golden answers
class EvaluationInput(BaseModel):
    questions: List[str]
    golden_answers: List[str]

@app.post("/evaluate_rag")
async def evaluate_rag(input_data: EvaluationInput):
    try:
        # Extract the questions and golden answers
        questions = input_data.questions
        golden_answers = input_data.golden_answers
        
        # Load the saved index (you can modify this if needed)
        vector_store = setup_vector_store(CONFIG['es_url'], CONFIG['index_name'])
        index = setup_llama_index([], vector_store)
        
        # Initialize empty lists to store answers and contexts
        answers = []
        contexts = []
        
        # Retrieve the answers and context chunks for each question
        for question in questions:
            answer, chunks = search_index(index, question)
            answers.append(answer)
            contexts.append(chunks)
        
        # Evaluate the RAG pipeline using the retrieved answers and contexts
        evaluation_scores = evaluate_rag_system(questions, answers, contexts, golden_answers)
        
        # Convert the scores to a list of dictionaries
        evaluation_scores_dict = evaluation_scores.to_dict(orient='records')

        return {
            "evaluation_scores": evaluation_scores_dict
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during evaluation: {e}")

def main():
    logging.info("Starting the main document ingestion process.")
    set_api_keys(CONFIG)
    es = connect_elasticsearch(CONFIG['es_url'])
    create_index_if_not_exists(es, CONFIG['index_name'])

    # Retrieve PDF files
    files = get_data_files(CONFIG['data_dir'])

    # Ingest documents into the index
    nodes = ingest_documents(es, CONFIG['index_name'], files)

    if nodes:
        try:
            vector_store = setup_vector_store(CONFIG['es_url'], CONFIG['index_name'])
            index = setup_llama_index(nodes, vector_store)
            save_index(index)
        except Exception as e:
            logging.error(f"Error during vector store or index setup: {str(e)}")
    else:
        logging.info("No new documents to ingest.")

if __name__ == "__main__":
    # Check if the script is being run with the --ingest argument
    if len(sys.argv) > 1 and sys.argv[1] == "--ingest":
        main()  # Run document ingestion only
    else:
        # Otherwise, start the FastAPI app
        logging.info("Starting FastAPI app.")
        uvicorn.run(app, host="0.0.0.0", port=8000)