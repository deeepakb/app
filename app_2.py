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

def load_conversation_history(username):
    try:
        response = s3_client.get_object(Bucket=conversation_history_bucket, Key=f"{username}/history.json")
        history = json.loads(response['Body'].read().decode('utf-8'))
        return history if isinstance(history, list) else []
    except s3_client.exceptions.NoSuchKey:
        return []
    except Exception as e:
        logger.error(f"Error loading conversation history: {e}")
        return []

def save_conversation_history(username, history):
    try:
        conversation_id = str(uuid.uuid4())
        timestamp = datetime.utcnow().isoformat()
        
        # Add conversation_id and timestamp to the history
        for entry in history:
            entry['conversation_id'] = conversation_id
            entry['timestamp'] = timestamp

        s3_client.put_object(
            Bucket=conversation_history_bucket,
            Key=f"{username}/history.json",
            Body=json.dumps(history, indent=2)
        )
    except Exception as e:
        logger.error(f"Error saving conversation history: {e}")


# Recursive RAG function
def recursive_rag(query, vector_store, conversation_history, iterations=3, original_query=None):
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
    "You are a bot primarily aimed at helping engineers working on the Amazon Redshift database due to its large size (1.5GB).\n"
    "You are being augmented with code snippets from the Redshift codebase, however, do NOT reference the fact you have been augmented with snippets in your responses. The user does not know you are being given code snippets. Respond as if you have all the knowledge by default.\n"
    "If you believe the code snippets that you need to answer were not supplied, simply state you're unable to answer the question. Don't reference the fact that the code snippets aren't relevant. NEVER MENTION THE CODE SNIPPETS.\n"
    "Keep your answer short and precise. NEVER MENTION THE CODE SNIPPETS.\n"
    "Conversation history is added for context; improve code iteratively if possible. If no improvements are possible, just straight up return the previous iteration with no change and dont mention no improvements are needed, just return it. NEVER MENTION THE CODE SNIPPETS. "
    "Use the conversation history to understand the context and answer based on that. NEVER MENTION THE CODE SNIPPETS.\n"
    "Provide direct answers to the user's queries. Avoid introductory phrases like 'Based on the code...' or 'Based on the query...' NEVER MENTION THE CODE SNIPPETS."
    )

    # Remove previous iterations for the current query, keeping only the last one
    conversation_history = [
        entry for entry in conversation_history if "Original query:" not in entry.get("content", "")
    ]
    #logger.info(pyjwt.decode(session['user'].get('id_token'))['sub'])
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
    conversation_history.extend([
        {"role": "user", "content": f"Original query: {original_query}\nFollow-up query: {query}"},
        {"role": "assistant", "content": response_content},
        #{"time" : datetime.datetime.now()}
    ])
    
    save_conversation_history(conversation_history)
    logger.info("\nIteration finished\n")
    logger.info(f"conversation history is {conversation_history} \n")
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

        # Pass the full conversation history to recursive_rag
        response = recursive_rag(user_query, vector_store, conversation_history, 4, user_query)
        logger.info(f"response is {response}")

        conversation_history = load_conversation_history(username)

        # Remove the previous 3 elements, keeping everything before that and the last one
        if len(conversation_history) > 4:
            conversation_history = conversation_history[:-4] + [conversation_history[-1]]
            save_conversation_history(username, conversation_history)

        # Modify the response to keep only the last iteration
        if isinstance(response, list):
            response = response[-1:]  # Keep only the last item in the list

        # Use a more specific regex to capture code blocks and their language
        formatted_response = re.sub(
            r'```(\w+)?\s*([\s\S]*?)```',
            lambda m: f'<div class="code-block"><pre><code class="language-{m.group(1) or ""}">{html.escape(m.group(2))}</code></pre></div>',
            response
        )

        # Replace newlines with <br> for regular text, but not within code blocks
        formatted_response = re.sub(
            r'(?<!>)(\n)(?!<)',
            '<br>',
            formatted_response
        )

        # Collapse consecutive <br> tags into a single <br> to avoid too many line breaks
        formatted_response = re.sub(r"(<br>\s*)+", "<br>", formatted_response)
        formatted_response = re.sub(r'^( +)', lambda m: '&nbsp;' * len(m.group(1)), formatted_response, flags=re.MULTILINE)

        return jsonify({"response": formatted_response})
    except Exception as e:
        logger.error(f"Error during chat processing: {e}")
        return jsonify({"error": str(e)}), 500




@app.route('/heartbeat')
def heartbeat():
    return 'OK', 200


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
    logger.info(f"Full incoming URL: {request.url}")
    logger.info(f"Query parameters: {request.args}")

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
                logger.info(f"Decoded ID Token: {decoded_id_token}")
            else:
                logger.warning("No id_token found in the response")

            if access_token:
                decoded_access_token = pyjwt.decode(access_token, options={"verify_signature": False})
                logger.info(f"Decoded Access Token: {decoded_access_token}")
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
