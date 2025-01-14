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

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()


app = Flask(__name__)
app.secret_key = 'deepak'
app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(minutes=30)
app.config['SESSION_TYPE'] = 'filesystem'
Session(app)

# Constants for Amazon Federate OAuth2
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
        # Create an empty list to represent a cleared conversation history
        empty_history = []
        
        # Save the empty list to the user's conversation history file in S3
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
        # Try to load existing history
        try:
            existing_history = load_conversation_history(username)
        except:
            existing_history = []

        # Append new messages to existing history
        updated_history = existing_history + new_messages

        # Trim the history if it exceeds a certain length (e.g., 100 messages)
        max_history_length = 100
        if len(updated_history) > max_history_length:
            updated_history = updated_history[-max_history_length:]

        # Save the updated history back to S3
        s3_client.put_object(
            Bucket=conversation_history_bucket,
            Key=f"{username}/conversation_history.json",
            Body=json.dumps(updated_history, indent=2),
            ContentType='application/json'
        )
        logger.info(f"Conversation history updated for user: {username}")
    except Exception as e:
        logger.error(f"Error saving conversation history for user {username}: {e}")

# Recursive RAG function
def recursive_rag(query, vector_store, conversation_history, iterations=4, original_query=None):
    if iterations <= 0:
        return None

    if original_query is None:
        original_query = query

    results = vector_store.similarity_search(query, k=35)
    if not results:
        logger.info("No results found for the query.")
        return None

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

    full_content = ""
    for chunk in sorted_deduplicated_chunks:
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

    bedrock = boto3.client('bedrock-runtime')
    time.sleep(15)
    response = bedrock.invoke_model(**kwargs)
    response_content = json.loads(response['body'].read())['content'][0]['text']

    # Update conversation history
    if iterations > 1:
    # For all iterations except the last, include both original and follow-up queries
        conversation_history.extend([
        {"role": "user", "content": f"Original query: {original_query}\nFollow-up query: {query}"},
        {"role": "assistant", "content": response_content},
    ])
    else:
    # For the last iteration, only include the original query
        conversation_history.extend([
        {"role": "user", "content": f"Original query: {original_query}"},
        {"role": "assistant", "content": response_content},
    ])

    
    save_conversation_history(session.get('username'), conversation_history)
    logger.info("\nIteration finished\n")
    logger.info(f"conversation history is {load_conversation_history(get_user_id())}\n")
    next_response = recursive_rag(response_content, vector_store, conversation_history, iterations - 1, original_query)
    return next_response if next_response else response_content

# Load FAISS vector store and embeddings
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

# Flask routes
@app.route("/")
@login_required
def home():
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

        # Convert the decoded token to a JSON string with indentation
        json_output = json.dumps(decoded_token, indent=2)
        logger.info("Access token is json_output")

        decoded_token = pyjwt.decode(user_data['id_token'], options={"verify_signature": False})

        # Convert the decoded token to a JSON string with indentation
        json_output = json.dumps(decoded_token, indent=2)

        user_query = request.json.get("query", "")
        username = session.get('username')
        
        # Load the conversation history
        conversation_history = load_conversation_history(username)
        # Pass the full conversation history to recursive_rag
        response = recursive_rag(user_query, vector_store, conversation_history, 4, user_query)
        logger.info(f"response is {response}")

        conversation_history = load_conversation_history(username)

        if len(conversation_history) >= 8:
    # Keep everything except the last 6 messages
            preserved_history = conversation_history[:-8]
    # Keep only the last pair (last 2 messages)
            last_pair = conversation_history[-2:]
    # Combine the preserved history with the last pair
            conversation_history = preserved_history + last_pair
            save_conversation_history(username, conversation_history)


        # Modify the response to keep only the last iteration
        if isinstance(response, list):
            response = response[-1:]  # Keep only the last item in the list

       # Use a more specific regex to capture code blocks and their language
        formatted_response = re.sub(
    r'```(\w+)?\s*([\s\S]*?)```',
    lambda m: f'''
        <div class="code-block">
            <div class="code-header">{(m.group(1) or "Code").capitalize()}</div>
            <pre><code class="language-{m.group(1) or "plaintext"}">{html.escape(m.group(2))}</code></pre>
        </div>''',
    response
)


        # Preserve newlines outside of code blocks
        formatted_response = re.sub(
            r'(?<!<pre><code)(\n)(?!</code></pre>)',
            '<br>',
            formatted_response
        )

        # Preserve indentation with non-breaking spaces
        formatted_response = re.sub(r'^( +)', lambda m: '&nbsp;' * len(m.group(1)), formatted_response, flags=re.MULTILINE)

        logger.info(f"formatted_response is {formatted_response}")

        
        return jsonify({
            "response": formatted_response})

    except Exception as e:
        logger.error(f"Error during chat processing: {e}")
        return jsonify({"error": str(e)}), 500




@app.route('/heartbeat')
def heartbeat():
    return 'OK', 200

@app.route('/clear-chat', methods=['POST'])
@login_required
def clear_chat():
    try:
        username = session.get('username')
        clear_conversation_history(username)
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
            else:
                logger.warning("No id_token found in the response")

            if access_token:
                decoded_access_token = pyjwt.decode(access_token, options={"verify_signature": False})
            else:
                logger.warning("No access_token found in the response")

            # Store the full token_data in session, or choose which parts you want to store
            session['user'] = token_data
            
            return redirect(url_for('home'))
        else:
            logger.error(f"Token endpoint returned an error: {response.text}")
            return jsonify({'error': 'Failed to get token', 'status_code': response.status_code}), 400
    else:
        return jsonify({'error': 'No authorization code received'}), 400

@app.route('/logout', methods=['GET', 'POST'])
def logout():
    # Clear the user session
    session.clear()
    
    # Redirect to the login page
    response = redirect(url_for('login'))
    
    # Add cache-control headers to prevent cached pages
    response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, post-check=0, pre-check=0, max-age=0'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '0'
    
    # Ensure the response is returned without any cached session data
    return response


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=50000)
