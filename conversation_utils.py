# conversation_utils.py

import boto3
import json
import logging
import os
import re
from datetime import datetime
import tiktoken
from config import Config

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# S3 Setup
s3_client = boto3.client('s3')
conversation_history_bucket = Config.CONVERSATION_HISTORY_BUCKET

# Bedrock Setup
bedrock = boto3.client('bedrock-runtime', region_name=Config.AWS_REGION)

# Token counter setup
tokenizer = tiktoken.get_encoding(Config.TOKENIZER_MODEL)

def count_tokens(text: str) -> int:
    """Count the number of tokens in a text string"""
    return len(tokenizer.encode(text))

def clear_conversation_history(username):
    """Clear conversation history for a user"""
    try:
        empty_history = []
        s3_client.put_object(
            Bucket=conversation_history_bucket,
            Key=f"{username}/conversation_history.json",
            Body=json.dumps(empty_history),
            ContentType='application/json'
        )
        logger.info(f"Conversation history cleared for user: {username}")
    except Exception as e:
        logger.error(f"Error clearing conversation history for user {username}: {e}")

def load_conversation_history(username):
    """Load conversation history for a user"""
    try:
        s3_object = s3_client.get_object(
            Bucket=conversation_history_bucket, 
            Key=f"{username}/conversation_history.json"
        )
        history = json.loads(s3_object['Body'].read().decode('utf-8'))
        return history if isinstance(history, list) else []
    except s3_client.exceptions.NoSuchKey:
        return []
    except Exception as e:
        logger.error(f"Error loading conversation history: {e}")
        return []

def save_conversation_history(username, history):
    """Save conversation history for a user"""
    try:
        s3_client.put_object(
            Bucket=conversation_history_bucket,
            Key=f"{username}/conversation_history.json",
            Body=json.dumps(history, indent=2),
            ContentType='application/json'
        )
    except Exception as e:
        logger.error(f"Error saving conversation history: {e}")

def update_token_count(username, user_query, response):
    """Update token count for a user"""
    try:
        user_tokens = count_tokens(user_query)
        bot_tokens = count_tokens(response)
        total_tokens = user_tokens + bot_tokens
        
        try:
            token_count = s3_client.get_object(
                Bucket=conversation_history_bucket,
                Key=f"{username}/token_count.json"
            )
            current_count = json.loads(token_count['Body'].read().decode('utf-8'))['count']
        except:
            current_count = 0
        
        new_count = current_count + total_tokens
        
        s3_client.put_object(
            Bucket=conversation_history_bucket,
            Key=f"{username}/token_count.json",
            Body=json.dumps({'count': new_count}),
            ContentType='application/json'
        )
        
        return new_count
    except Exception as e:
        logger.error(f"Error updating token count: {e}")
        return 0

def recursive_rag(query, vector_store, conversation_history, iterations=4, original_query=None):
    """Perform recursive RAG"""
    if iterations <= 0:
        return None

    if original_query is None:
        original_query = query

    results = vector_store.similarity_search(query, k=Config.SEARCH_SIMILARITY)

    if not results:
        logger.info("No results found for the query.")
        return None
    
    # Process Retrieved Chunks
    seen_content = set()
    deduplicated_chunks = []
    
    for chunk in results:
        chunk_content = chunk.page_content.strip()
        if chunk_content not in seen_content:
            deduplicated_chunks.append(chunk)
            seen_content.add(chunk_content)

    full_content = []
    for chunk in deduplicated_chunks:
        full_content.extend([
            f"File: {os.path.basename(chunk.metadata.get('file_path', 'Unknown'))}",
            f"Chunk ID : {chunk.metadata.values}",
            chunk.page_content.strip(),
            ""
        ])
    full_content = "\n".join(full_content)
    full_content = re.sub(r'\n{3,}', '\n\n', full_content).strip()

    prompt_info = (
        "You are a bot designed to assist engineers working on the Amazon Redshift database. "
        "Your responses should appear as if you possess all necessary knowledge by default, "
        "without referencing or revealing any internal processes, including code snippets.\n\n"
        "Guidelines:\n"
        "1. Do not mention or imply the use of code snippets, conversation history, or augmentation in your responses.\n"
        "2. If the information needed to answer is unavailable, state that you are unable to answer, without referencing the reason.\n"
        "3. Keep your answers short, precise, and direct. Avoid introductory phrases like 'Based on the code' or 'From the query'.\n"
        "4. Use conversation history to maintain context and iteratively improve code if possible. If no improvements are needed, return the previous iteration without commenting on it.\n\n"
        "5. Always maintain a professional and concise tone."
    )

    messages = conversation_history + [
        {
            "role": "user",
            "content": f"Original query: {original_query}\n\nFollow-up query: {query}\n\n"
                      f"Please answer the follow-up query while keeping the original query in context. Here are some code code snippets which may or may not be relevant to the query\n{full_content}\n\n Prompt info that should be followed VERY closely: {prompt_info}"
        }
    ]

    kwargs = {
        "modelId": Config.BEDROCK_MODEL_ID,
        "contentType": "application/json",
        "accept": "application/json",
        "body": json.dumps({
            "anthropic_version": Config.ANTHROPIC_VERSION,
            "max_tokens": Config.MODEL_MAX_TOKENS,
            "top_k": Config.MODEL_TOP_K,
            "stop_sequences": [],
            "temperature": Config.MODEL_TEMPERATURE,
            "top_p": Config.MODEL_TOP_P,
            "messages": messages
        })
    }

    response = bedrock.invoke_model(**kwargs)
    response_content = json.loads(response['body'].read())['content'][0]['text']

    if iterations > 1:
        conversation_history.extend([
            {"role": "user", "content": f"Original query: {original_query}\nFollow-up query: {query}"},
            {"role": "assistant", "content": response_content},
        ])
    else:
        conversation_history.extend([
            {"role": "user", "content": f"Original query: {original_query}"},
            {"role": "assistant", "content": response_content},
        ])

    next_response = recursive_rag(response_content, vector_store, conversation_history, iterations - 1, original_query)
    return next_response if next_response else response_content

def format_code_blocks(response):
    """Format code blocks in the response"""
    if isinstance(response, str):
        import html
        opening_blocks = response.count('```')
        
        if opening_blocks % 2 != 0:
            response += '\n```'

        code_pattern = re.compile(r'```(cpp|c\+\+|python|sql|javascript|bash)?(.*?)```', re.DOTALL)
        matches = code_pattern.findall(response)
        
        for language, match in matches:
            escaped_code = html.escape(match.strip())
            if language in ['c++', 'cpp']:
                language = 'cpp'
            wrapped_code = f'<pre><code class="language-{language or "plaintext"}">{escaped_code}</code></pre>'
            response = response.replace(f"```{language or ''}{match}```", wrapped_code)
    
    return response

def handle_feedback(feedback_data, username):
    """Handle user feedback"""
    try:
        timestamp = datetime.now().isoformat()
        
        feedback_object = {
            'timestamp': timestamp,
            'username': username,
            'user_message': feedback_data.get('user_message'),
            'bot_message': feedback_data.get('bot_message'),
            'feedback_type': feedback_data.get('feedback_type'),
            'negative_feedback_reason': feedback_data.get('negative_feedback_reason', None)
        }

        file_key = f"feedback/{'positive_feedback.json' if feedback_data['feedback_type'] == 'positive' else 'negative_feedback.json'}"
        
        try:
            existing_feedback = s3_client.get_object(
                Bucket=conversation_history_bucket,
                Key=file_key
            )
            feedback_list = json.loads(existing_feedback['Body'].read().decode('utf-8'))
        except s3_client.exceptions.NoSuchKey:
            feedback_list = []

        feedback_list.append(feedback_object)

        s3_client.put_object(
            Bucket=conversation_history_bucket,
            Key=file_key,
            Body=json.dumps(feedback_list, indent=2),
            ContentType='application/json'
        )

        return True
    except Exception as e:
        logger.error(f"Error handling feedback: {e}")
        return False
