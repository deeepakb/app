# auth_utils.py

from functools import wraps
from flask import session, redirect, url_for, jsonify, request
import jwt as pyjwt
import requests
from config import Config

def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user' not in session or not session.get('user'):
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

def get_user_id():
    """Get user ID from session token"""
    session['username'] = pyjwt.decode(session['user'].get('id_token'), options={"verify_signature": False}).get('sub')
    return pyjwt.decode(session['user'].get('id_token'), options={"verify_signature": False}).get('sub')

def handle_login():
    """Handle the login redirect"""
    if 'user' in session:
        return redirect(url_for('home'))

    query_params = {
        'client_id': Config.CLIENT_ID,
        'redirect_uri': Config.REDIRECT_URI,
        'scope': 'openid',
        'response_mode': 'query',
        'response_type': 'code'
    }
    
    url = f'{Config.AMAZON_FEDERATE_URL}/api/oauth2/v1/authorize'
    return redirect(f'{url}?{requests.compat.urlencode(query_params)}')

def handle_idp_response(code):
    """Handle the IDP response and token exchange"""
    try:
        if code:
            payload = {
                'code': code,
                'client_id': Config.CLIENT_ID,
                'client_secret': Config.CLIENT_SECRET,
                'grant_type': 'authorization_code',
                'redirect_uri': Config.REDIRECT_URI
            }
            headers = {
                'Content-Type': 'application/json'
            }

            response = requests.post(
                f"{Config.AMAZON_FEDERATE_URL}/api/oauth2/v1/token", 
                json=payload, 
                headers=headers
            )
            
            if response.status_code == 200:
                token_data = response.json()
                id_token = token_data.get('id_token')
                
                if id_token:
                    decoded_id_token = pyjwt.decode(id_token, options={"verify_signature": False})
                    session['username'] = decoded_id_token.get('sub')
                
                session['user'] = token_data
                return redirect(url_for('home'))
            else:
                return jsonify({
                    'error': 'Failed to get token', 
                    'status_code': response.status_code
                }), 400
        else:
            return jsonify({'error': 'No authorization code received'}), 400
            
    except Exception as e:
        return jsonify({'error': f'Authentication error: {str(e)}'}), 500

def handle_logout():
    """Handle user logout"""
    session.clear()
    response = redirect(url_for('login'))
    
    # Set cache control headers
    response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, post-check=0, pre-check=0, max-age=0'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '0'
    
    return response
