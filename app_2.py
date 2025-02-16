from flask import Flask, request, jsonify, render_template, redirect, session, url_for
from flask_session import Session
import os
import json
import logging
from collections import defaultdict
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import boto3
import re
from functools import wraps
from datetime import timedelta
import jwt as pyjwt
import datetime
from html import escape
import html
from datetime import datetime
import tiktoken
from config import Config
from auth_utils import login_required, get_user_id, handle_login, handle_idp_response, handle_logout
from conversation_utils import (
    recursive_rag, load_conversation_history, save_conversation_history,
    clear_conversation_history, update_token_count, format_code_blocks,
    handle_feedback
)

SEARCH_SIMILARITY = 15

tokenizer = tiktoken.get_encoding(Config.TOKENIZER_MODEL)  # For getting Token Count

#S3 Setup
s3_client = boto3.client('s3')
conversation_history = Config.CONVERSATION_HISTORY_BUCKET
client = boto3.client('s3')
conversation_history_bucket = Config.CONVERSATION_HISTORY_BUCKET


embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-mpnet-base-v2",
    model_kwargs={'device': "cpu"},
    encode_kwargs={"batch_size": 131072}
)

#Logging Setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()
bedrock = boto3.client('bedrock-runtime', region_name=Config.AWS_REGION)


#Flask App Setup
app = Flask(__name__, static_folder='static')
app.secret_key = Config.SECRET_KEY
app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(minutes=Config.PERMANENT_SESSION_LIFETIME)
app.config['SESSION_TYPE'] = Config.SESSION_TYPE
Session(app)

#Gets User ID
def get_user_id():
    session['username'] = pyjwt.decode(session['user'].get('id_token'), options={"verify_signature": False}).get('sub')
    return pyjwt.decode(session['user'].get('id_token'), options={"verify_signature": False}).get('sub')


#Clear Conversation History on Clear Chat Button Press
def clear_conversation_history(username):
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

#Conversation Token Counter
def count_tokens(text: str) -> int:
    return len(tokenizer.encode(text))

#Load Conversation History
def load_conversation_history(username):
    try:
        s3_object = s3_client.get_object(Bucket=conversation_history_bucket, Key=f"{username}/conversation_history.json")
        history = json.loads(s3_object['Body'].read().decode('utf-8'))
        return history if isinstance(history, list) else []
    except s3_client.exceptions.NoSuchKey:
        return []
    except Exception as e:
        logger.error(f"Error loading conversation history: {e}")
        return []

#Save Conversation History
def save_conversation_history(username, history):
    try:
        s3_client.put_object(
            Bucket=conversation_history_bucket,
            Key=f"{username}/conversation_history.json",
            Body=json.dumps(history, indent=2),
            ContentType='application/json'
        )
    except Exception as e:
        logger.error(f"Error saving conversation history: {e}")

# Recursive RAG
def recursive_rag(query, vector_store, conversation_history, iterations=4, original_query=None):
    if iterations <= 0:
        return None

    if original_query is None:
        original_query = query

    #Similarity Search
    results = vector_store.similarity_search(query, k=Config.SEARCH_SIMILARITY)

    if not results:
        logger.info("No results found for the query.")
        return None
    
    #Process Retrieved Chunks
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

    #Prompt Information for the Model
    prompt_info = (
        "You are a bot designed to assist engineers working on the Amazon Redshift database. Your responses should appear as if you possess all necessary knowledge by default, without referencing or revealing any internal processes, including code snippets.\n\n"
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

    
    save_conversation_history(session.get('username'), conversation_history)
    logger.info("\nIteration finished\n")
    logger.info(f"conversation history is {load_conversation_history(get_user_id())}\n")
    next_response = recursive_rag(response_content, vector_store, conversation_history, iterations - 1, original_query)
    return next_response if next_response else response_content

index_path = "/home/ec2-user/internProject/app/faiss_index_final_improved_3"
vector_store = FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)
logger.info("FAISS index loaded successfully.")

def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user' not in session or not session.get('user'):
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

@app.after_request
def add_cache_control_headers(response):
    response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, post-check=0, pre-check=0, max-age=0'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '0'
    return response

#Home Page
@app.route("/")
@login_required
def home():
    if 'username' not in session and 'user' in session:
        decoded_token = pyjwt.decode(session['user'].get('id_token'), options={"verify_signature": False})
        session['username'] = decoded_token.get('sub')
    
    username = session.get('username')
    conversation_history = load_conversation_history(username)
    session['previous_conversation_history'] = conversation_history
    logger.info(f"previous conversation histories are {conversation_history} and user id is {username}")
    clear_conversation_history(username)
    return render_template("index.html", pyjwt=pyjwt)



#Chat Function
@app.route("/chat", methods=["POST"])
@login_required
def chat():
    try:
        user_query = request.json.get("query", "")
        username = session.get('username')
        
        conversation_history = load_conversation_history(username)
        response = recursive_rag(user_query, vector_store, conversation_history, 4, user_query)
        
        conversation_history = load_conversation_history(username)

        if len(conversation_history) >= 8:
            preserved_history = conversation_history[:-8]
            last_pair = conversation_history[-2:]
            conversation_history = preserved_history + last_pair
            save_conversation_history(username, conversation_history)

        response = format_code_blocks(response)
        new_count = update_token_count(username, user_query, response)

        return jsonify({
            "response": response,
            "token_count": new_count
        })

    except Exception as e:
        logger.error(f"Error during chat processing: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/submit-feedback', methods=['POST'])
@login_required
def submit_feedback():
    try:
        feedback_data = request.json
        username = session.get('username')
        
        if handle_feedback(feedback_data, username):
            return jsonify({'status': 'success'}), 200
        else:
            return jsonify({'error': 'Failed to save feedback'}), 500

    except Exception as e:
        logger.error(f"Error saving feedback: {e}")
        return jsonify({'error': str(e)}), 500

#Health Check
@app.route('/heartbeat')
def heartbeat():
    return 'OK', 200

#Clear Chat Function
@app.route('/clear-chat', methods=['POST'])
@login_required
def clear_chat():
    try:
        username = session.get('username')
        clear_conversation_history(username)
        
        s3_client.put_object(
            Bucket=conversation_history_bucket,
            Key=f"{username}/token_count.json",
            Body=json.dumps({'count': 0}),
            ContentType='application/json'
        )
        
        return jsonify({"message": "Chat cleared successfully"}), 200
    except Exception as e:
        logger.error(f"Error clearing chat: {e}")
        return jsonify({"error": str(e)}), 500


#Login Handling
@app.route('/login')
def login():
    return handle_login()

#Upload Diff File
@app.route('/upload-diff', methods=['POST'])
@login_required
def upload_diff():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
            
        if not file.filename.endswith(('.diff', '.patch')):
            return jsonify({'error': 'Invalid file type. Only .diff or .patch files are allowed'}), 400

        diff_content = file.read().decode('utf-8')
        
        analysis_result = process_diff_content(diff_content)
        formatted_analysis = format_security_analysis(analysis_result)
        
        return jsonify({
            'message': 'File processed successfully',
            'filename': file.filename,
            'analysis': formatted_analysis
        }), 200

    except Exception as e:
        logger.error(f"Error processing diff file: {e}")
        return jsonify({'error': str(e)}), 500

#Process Diff File for CR
def process_diff_content(diff_content):
    """
    Process the diff content using Claude LLM with RAG to analyze security concerns and code review validity
    """
    try:
        lines = diff_content.split('\n')
        files_changed = len([l for l in lines if l.startswith('diff --git')])
        additions = len([l for l in lines if l.startswith('+')])
        deletions = len([l for l in lines if l.startswith('-')])
        
        additions -= len([l for l in lines if l.startswith('+++')])
        deletions -= len([l for l in lines if l.startswith('---')])

        results = vector_store.similarity_search(diff_content, k=Config.SEARCH_SIMILARITY)
        
        file_groups = defaultdict(list)
        for chunk in results:
            file_path = chunk.metadata.get('file_path', '')
            file_name = os.path.basename(file_path)
            file_groups[file_name].append(chunk)

        sorted_deduplicated_chunks = []
        for file_name, chunks in file_groups.items():
            sorted_file_chunks = sorted(chunks, key=lambda x: int(x.metadata.get("id", "9999")))
            deduplicated_file_chunks = []
            seen_content = set()
            for chunk in sorted_file_chunks:
                chunk_content = chunk.page_content.strip()
                if chunk_content not in seen_content:
                    deduplicated_file_chunks.append(chunk)
                    seen_content.add(chunk_content)
            sorted_deduplicated_chunks.extend(deduplicated_file_chunks)

        similar_content = []
        for chunk in sorted_deduplicated_chunks:
            similar_content.extend([
                f"Similar File: {os.path.basename(chunk.metadata.get('file_path', 'Unknown'))}",
                chunk.page_content.strip(),
                ""
            ])
        similar_content = "\n".join(similar_content)
        similar_content = re.sub(r'\n{3,}', '\n\n', similar_content).strip()

        #Prompt Info to Analyze Diff File

        messages = [
            {
                "role": "user",
                "content": Config.SECURITY_PROMPT.format(
                    diff_content=diff_content,
                    similar_content=similar_content
                )
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


        bedrock = boto3.client('bedrock-runtime')
        response = bedrock.invoke_model(**kwargs)
        analysis_response = json.loads(response['body'].read())['content'][0]['text']

        return {
            'basic_stats': {
                'files_changed': files_changed,
                'additions': additions,
                'deletions': deletions,
                'total_changes': additions + deletions
            },
            'security_analysis': analysis_response,
            'timestamp': datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"Error in diff processing: {e}")
        raise


def format_security_analysis(analysis_result):
    """
    Helper function to format the security analysis results for display
    """
    return f"""Time: {analysis_result['timestamp']}
    Security Analysis:
    {analysis_result['security_analysis']}"""

@app.route('/idpresponse')
def idp_response():
    code = request.args.get('code')
    return handle_idp_response(code)

@app.route('/logout', methods=['GET', 'POST'])
def logout():
    return handle_logout()

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=Config.PORT)
