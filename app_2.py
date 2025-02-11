from flask import Flask, request, jsonify, render_template, redirect, session, url_for
from flask_session import Session
import os
import json
import time
import logging
from collections import defaultdict
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import boto3
import re
import urllib.parse
import requests
from functools import wraps
from datetime import timedelta
import jwt as pyjwt
import datetime
from html import escape
import html
import uuid
from datetime import datetime

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()
bedrock = boto3.client('bedrock-runtime', region_name='us-west-2')


app = Flask(__name__, static_folder='static')
app.secret_key = 'deepak'
app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(minutes=30)
app.config['SESSION_TYPE'] = 'filesystem'
Session(app)

AMAZON_FEDERATE_URL = 'https://idp-integ.federate.amazon.com'
REDIRECT_URI = 'http://elb-1340525831.us-west-2.elb.amazonaws.com/idpresponse'
CLIENT_ID = 'deepak_chatbot'
CLIENT_SECRET = 'lB7NfuttFSgAQjkOa2Jv9P0jMpObJRuZkmN98z70SaSN'

s3_client = boto3.client('s3')
conversation_history = 'conversation-history-deepak' 

client = boto3.client('s3')

def get_user_id():
    session['username'] = pyjwt.decode(session['user'].get('id_token'), options={"verify_signature": False}).get('sub')
    return pyjwt.decode(session['user'].get('id_token'), options={"verify_signature": False}).get('sub')

conversation_history_bucket = 'conversation-history-deepak'

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

import tiktoken
tokenizer = tiktoken.get_encoding("cl100k_base")

def count_tokens(text: str) -> int:
    return len(tokenizer.encode(text))


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

def append_conversation_history(username, new_messages):
    try:
        try:
            existing_history = load_conversation_history(username)
        except:
            existing_history = []

        updated_history = existing_history + new_messages

        max_history_length = 100
        if len(updated_history) > max_history_length:
            updated_history = updated_history[-max_history_length:]

        s3_client.put_object(
            Bucket=conversation_history_bucket,
            Key=f"{username}/conversation_history.json",
            Body=json.dumps(updated_history, indent=2),
            ContentType='application/json'
        )
        logger.info(f"Conversation history updated for user: {username}")
    except Exception as e:
        logger.error(f"Error saving conversation history for user {username}: {e}")


# Recursive RAG
def recursive_rag(query, vector_store, conversation_history, iterations=4, original_query=None):
    if iterations <= 0:
        return None

    if original_query is None:
        original_query = query

    results = vector_store.similarity_search(query, k=15)
    if not results:
        logger.info("No results found for the query.")
        return None
    
    seen_content = set()
    deduplicated_chunks = []
    
    for chunk in results:
        chunk_content = chunk.page_content.strip()
        if chunk_content not in seen_content:
            deduplicated_chunks.append(chunk)
            seen_content.add(chunk_content)

    full_content = ""
    for chunk in deduplicated_chunks:
        full_content += f"File: {os.path.basename(chunk.metadata.get('file_path', 'Unknown'))}\n"
        full_content += f"Chunk ID : {chunk.metadata.values}\n"
        full_content += chunk.page_content.strip() + "\n\n"
    full_content = re.sub(r'\n{3,}', '\n\n', full_content).strip()


    prompt_info = (
    "You are a bot designed to assist engineers working on the Amazon Redshift database. Your responses should appear as if you possess all necessary knowledge by default, without referencing or revealing any internal processes, including code snippets.\n\n"
    "Guidelines:\n"
    "1. Do not mention or imply the use of code snippets, conversation history, or augmentation in your responses.\n"
    "2. If the information needed to answer is unavailable, state that you are unable to answer, without referencing the reason.\n"
    "3. Keep your answers short, precise, and direct. Avoid introductory phrases like 'Based on the code' or 'From the query'.\n"
    "4. Use conversation history to maintain context and iteratively improve code if possible. If no improvements are needed, return the previous iteration without commenting on it.\n\n"
    "Always maintain a professional and concise tone."
)


    messages = conversation_history + [
        {
            "role": "user",
            "content": f"Original query: {original_query}\n\nFollow-up query: {query}\n\n"
                       f"nPlease answer the follow-up query while keeping the original query in context. Here are some code code snippets which may or may not be relevant to the query\n{full_content}\n\n Prompt info that should be followed VERY closely: {prompt_info}"
        }
    ]

    kwargs = {
        "modelId": "anthropic.claude-3-5-sonnet-20241022-v2:0",
        "contentType": "application/json",
        "accept": "application/json",
        "body": json.dumps({
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 550,
            "top_k": 10,
            "stop_sequences": [],
            "temperature": 0.3,
            "top_p": 0.95,
            "messages": messages
        })
    }

    #time.sleep(1)
    #logger.info(f"\nMessages is {messages} and its size is {len(messages)}\n")
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

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-mpnet-base-v2",
    model_kwargs={'device': "cpu"},
    encode_kwargs={"batch_size": 131072}
)
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




@app.route("/chat", methods=["POST"])
@login_required
def chat():
    try:
        user_data = session['user']
        decoded_token = pyjwt.decode(user_data['access_token'], options={"verify_signature": False})

        json_output = json.dumps(decoded_token, indent=2)
        logger.info("Access token is json_output")

        decoded_token = pyjwt.decode(user_data['id_token'], options={"verify_signature": False})

        json_output = json.dumps(decoded_token, indent=2)

        user_query = request.json.get("query", "")
        username = session.get('username')
        
        conversation_history = load_conversation_history(username)
        response = recursive_rag(user_query, vector_store, conversation_history, 4, user_query)
        logger.info(f"response is {response}")

        conversation_history = load_conversation_history(username)

        if len(conversation_history) >= 8:
            preserved_history = conversation_history[:-8]
            last_pair = conversation_history[-2:]
            conversation_history = preserved_history + last_pair
            save_conversation_history(username, conversation_history)


        if isinstance(response, list):
            response = response[-1:] 


        if isinstance(response, str):
            import re
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


            
        logger.info(f"formatted response is {response}")

        user_tokens = count_tokens(user_query)
        bot_tokens = count_tokens(response)
        total_tokens = user_tokens + bot_tokens
        
        # Get current token count from S3 or create new
        try:
            token_count = s3_client.get_object(
                Bucket=conversation_history_bucket,
                Key=f"{username}/token_count.json"
            )
            current_count = json.loads(token_count['Body'].read().decode('utf-8'))['count']
        except:
            current_count = 0
        
        # Update total token count
        new_count = current_count + total_tokens
        
        # Save the new count to S3
        s3_client.put_object(
            Bucket=conversation_history_bucket,
            Key=f"{username}/token_count.json",
            Body=json.dumps({'count': new_count}),
            ContentType='application/json'
        )
        logger.info(f"Sending response with token count: {new_count}")  
        return jsonify({
            "response": response,
            "token_count": new_count  # Include the token count in the response
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
        timestamp = datetime.now().isoformat()
        
        # Create the feedback object
        feedback_object = {
            'timestamp': timestamp,
            'username': username,
            'user_message': feedback_data.get('user_message'),
            'bot_message': feedback_data.get('bot_message'),
            'feedback_type': feedback_data.get('feedback_type'),
            'negative_feedback_reason': feedback_data.get('negative_feedback_reason', None)
        }

        # Determine which file to write to based on feedback type
        file_key = f"feedback/{'positive_feedback.json' if feedback_data['feedback_type'] == 'positive' else 'negative_feedback.json'}"
        
        try:
            # Try to get existing feedback
            existing_feedback = s3_client.get_object(
                Bucket=conversation_history_bucket,
                Key=file_key
            )
            feedback_list = json.loads(existing_feedback['Body'].read().decode('utf-8'))
        except s3_client.exceptions.NoSuchKey:
            feedback_list = []

        # Add new feedback
        feedback_list.append(feedback_object)

        # Save updated feedback
        s3_client.put_object(
            Bucket=conversation_history_bucket,
            Key=file_key,
            Body=json.dumps(feedback_list, indent=2),
            ContentType='application/json'
        )

        return jsonify({'status': 'success'}), 200

    except Exception as e:
        logger.error(f"Error saving feedback: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/heartbeat')
def heartbeat():
    return 'OK', 200

@app.route('/clear-chat', methods=['POST'])
@login_required
def clear_chat():
    try:
        username = session.get('username')
        clear_conversation_history(username)
        
        # Reset token count
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



@app.route('/login')
def login():
    if 'user' in session:
        return redirect(url_for('home'))

    query_params = {
        'client_id': CLIENT_ID,
        'redirect_uri': REDIRECT_URI,
        'scope': 'openid',
        'response_mode': 'query',
        'response_type': 'code'
    }
    
    url = f'{AMAZON_FEDERATE_URL}/api/oauth2/v1/authorize'
    return redirect(f'{url}?{requests.compat.urlencode(query_params)}')

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

        results = vector_store.similarity_search(diff_content, k=17)
        
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

        similar_content = ""
        for chunk in sorted_deduplicated_chunks:
            similar_content += f"Similar File: {os.path.basename(chunk.metadata.get('file_path', 'Unknown'))}\n"
            similar_content += chunk.page_content.strip() + "\n\n"
        similar_content = re.sub(r'\n{3,}', '\n\n', similar_content).strip()

        security_prompt = """
            You are a senior security engineer reviewing code changes across multiple languages (Python, JavaScript, C++). Analyze the following for potential security issues and code quality concerns.

            Focus on these key areas:
            1. Memory & Resource Management:
            - Buffer overflows and underflows
            - Use-after-free vulnerabilities
            - Null pointer dereferences
            - Memory leaks
            - Out-of-bounds read/write
            - Resource cleanup
            - Integer overflow/underflow
            - Thread safety violations

            2. Input Validation & Sanitization:
            - SQL/NoSQL injection
            - OS command injection
            - Path traversal
            - Buffer boundary checks
            - Input size validation
            - Type safety
            - Pointer validation
            - Array bounds checking

            3. Authentication & Authorization:
            - Missing authorization checks
            - Improper authentication
            - Session management
            - Privilege escalation
            - setuid/setgid ordering
            - Permission validation
            - Access control implementation

            4. Cryptography & Data Protection:
            - Weak cryptographic implementations
            - Insecure random number generation
            - Certificate validation
            - Sensitive data exposure
            - Clear text credentials
            - Secure storage patterns
            - Encryption consistency

            5. Memory Safety (C++ Specific):
            - Pointer arithmetic
            - sizeof usage
            - Stack address returns
            - Multiple lock handling
            - Pointer scaling
            - Buffer access safety
            - Memory allocation

            6. Client-Side Security (JavaScript Specific):
            - Cross-site scripting (XSS)
            - DOM manipulation
            - eval() usage
            - Async operation safety
            - CSRF protection
            - Frame security
            - Browser API safety

            7. System & File Operations:
            - File permission handling
            - Temporary file security
            - Directory traversal
            - File extension validation
            - Concurrent file access
            - Resource locking
            - File system race conditions

            8. Error Handling & Logging:
            - Exception management
            - Error propagation
            - Stack trace exposure
            - Logging of sensitive data
            - Null checks
            - Return value validation
            - Switch statement completeness

            9. Language-Specific Patterns:
            C++:
            - RAII compliance
            - Smart pointer usage
            - STL container safety
            - Const correctness
            JavaScript:
            - Promise handling
            - Prototype safety
            - Event loop considerations
            Python:
            - Deserialization safety
            - Package import security
            - Context manager usage

            For each issue found:
            - Identify the specific location and language context
            - Explain the potential risk and exploit scenario
            - Provide a recommended fix with language-specific best practices
            - Rate the severity (Critical/High/Medium/Low)
            - Include relevant secure coding patterns
            - Reference similar patterns in the existing codebase

            Diff content to analyze:
            {diff_content}

            Similar code files for context:
            {similar_content}

            Begin your analysis with memory safety and critical security issues first, followed by language-specific concerns. Do not include any preamble or conclusion.
            """

        messages = [
            {
                "role": "user",
                "content": security_prompt.format(
                    diff_content=diff_content,
                    similar_content=similar_content
                )
            }
        ]

        kwargs = {
            "modelId": "anthropic.claude-3-5-sonnet-20241022-v2:0",
            "contentType": "application/json",
            "accept": "application/json",
            "body": json.dumps({
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": 1000,
                "top_k": 10,
                "stop_sequences": [],
                "temperature": 0.3,
                "top_p": 0.95,
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
    if code:
        payload = {
            'code': code,
            'client_id': CLIENT_ID,
            'client_secret': CLIENT_SECRET,
            'grant_type': 'authorization_code',
            'redirect_uri': REDIRECT_URI
        }
        headers = {
            'Content-Type': 'application/json'
        }

        response = requests.post(f"{AMAZON_FEDERATE_URL}/api/oauth2/v1/token", json=payload, headers=headers)
        
        if response.status_code == 200:
            token_data = response.json()
            id_token = token_data.get('id_token')
            access_token = token_data.get('access_token')
            
            if id_token:
                decoded_id_token = pyjwt.decode(id_token, options={"verify_signature": False})
                session['username'] = decoded_id_token.get('sub')
            else:
                logger.warning("No id_token found in the response")

            if access_token:
                decoded_access_token = pyjwt.decode(access_token, options={"verify_signature": False})
            else:
                logger.warning("No access_token found in the response")

            session['user'] = token_data
            
            return redirect(url_for('home'))
        else:
            logger.error(f"Token endpoint returned an error: {response.text}")
            return jsonify({'error': 'Failed to get token', 'status_code': response.status_code}), 400
    else:
        return jsonify({'error': 'No authorization code received'}), 400

@app.route('/logout', methods=['GET', 'POST'])
def logout():
    session.clear()
    
    response = redirect(url_for('login'))
    
    response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, post-check=0, pre-check=0, max-age=0'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '0'
    
    return response


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=50000)
