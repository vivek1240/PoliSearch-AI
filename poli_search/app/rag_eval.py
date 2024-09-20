import openai
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
from openai import Client
client = Client()
client.chat.completions.create

openai.api_key = "Your_API_key_here"


# Function to evaluate faithfulness of the answer w.r.t. context
def evaluate_faithfulness(question, answer, context):
    """
    Evaluates the faithfulness of the generated answer with respect to the given context.
    """
    # Extract statements from the answer
    extract_prompt = f"Given a question and answer, create one or more statements from each sentence in the given answer. question: {question} answer: {answer}"
    extract_response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": extract_prompt}]
    )
    extract_response = extract_response.model_dump()    # <--- convert to dictionary
    statements = extract_response['choices'][0]['message']['content'].split('\n')


    # Verify each statement against the context
    statements_formatted = "\n".join(f"statement: {statement}" for statement in statements)
    verify_prompt = f"Given the context: {context}, check whether the statements are supported (Yes/No) and provide a verdict for each statement."

    verify_response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": verify_prompt}]
    )
    
    verify_response = verify_response.model_dump()    # <--- convert to dictionary
    supported_statements = verify_response['choices'][0]['message']['content'].count('Yes')
    total_statements = len(statements)
    # faithfulness_score = supported_statements / total_statements if total_statements else 0
    faithfulness_score = min(supported_statements / total_statements if total_statements else 0, 1)
    return faithfulness_score

# Function to evaluate relevance of the generated answer to the question
def evaluate_answer_relevance(question, answer):
    """
    Evaluates the relevance of the generated answer to the input question.
    """
    # Generate questions from the answer
    generate_prompt = f"Generate a question for the given answer: {answer}"
    generate_response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": generate_prompt}]
    )
    generate_response = generate_response.model_dump()    # <--- convert to dictionary
    generated_questions = generate_response['choices'][0]['message']['content'].split('\n')

    # Calculate cosine similarity between the original and generated questions
    question_embedding = client.embeddings.create(input=question, model='text-embedding-ada-002').data[0].embedding
    question_embedding = np.array(question_embedding).reshape(1, -1)  # Reshape to 2D

    similarities = []
    
    for gen_question in generated_questions:
        gen_question_embedding = client.embeddings.create(input=gen_question, model='text-embedding-ada-002').data[0].embedding
        gen_question_embedding = np.array(gen_question_embedding).reshape(1, -1)  # Reshape to 2D
        similarity = cosine_similarity(question_embedding, gen_question_embedding)
        similarities.append(similarity[0][0])
    
    answer_relevance_score = sum(similarities) / len(similarities) if similarities else 0
    return answer_relevance_score


# Function to evaluate the relevance of the context to the input question
def evaluate_context_relevance(question, context):
    """
    Evaluates the relevance of the context to the input question.
    """
    # Ensure context is a string
    if isinstance(context, list):
        context = "\n".join(context)  # Join list into a single string

    extract_prompt = f"Please extract relevant sentences from the provided context to answer the question. If no relevant sentences are found, return 'Insufficient Information'. question: {question} context: {context}"
    extract_response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": extract_prompt}]
    )
    extract_response = extract_response.model_dump()    # <--- convert to dictionary
    extracted_sentences = extract_response['choices'][0]['message']['content'].split('\n')
    total_sentences = context.split('\n')
    
    context_relevance_score = len(extracted_sentences) / len(total_sentences) if total_sentences else 0
    return context_relevance_score

# Overall evaluation function for RAG system
def evaluate_rag_system(questions, answers, contexts, golden_answers):
    """
    Evaluates the RAG system using the provided dataset.
    """
    scores = []
    for i in range(len(questions)):
        question = questions[i]
        answer = answers[i]
        context = contexts[i]
        golden_answer = golden_answers[i]

        # Evaluating metrics
        faithfulness_score = evaluate_faithfulness(question, answer, context)
        answer_relevance_score = evaluate_answer_relevance(question, answer)
        context_relevance_score = evaluate_context_relevance(question, context)

        scores.append({
            "faithfulness": faithfulness_score,
            "answer_relevance": answer_relevance_score,
            "context_relevance": context_relevance_score
        })
    
    return pd.DataFrame(scores)