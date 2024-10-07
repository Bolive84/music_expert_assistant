from flask import Flask, render_template, jsonify, request, session, Response, stream_with_context
from flask_session import Session
import os
import requests
from openai import OpenAI
from flask_cors import CORS
import os
from dotenv import load_dotenv

load_dotenv() 

app = Flask(__name__)
CORS(app)

OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')
TICKETMASTER_API_KEY = os.environ.get('TICKETMASTER_API_KEY')

# Configure Flask-Session
app.config['SECRET_KEY'] = os.urandom(24)
app.config['SESSION_TYPE'] = 'filesystem'
app.config['SESSION_PERMANENT'] = False
app.config['SESSION_USE_SIGNER'] = True
Session(app)

client = OpenAI(api_key=OPENAI_API_KEY)

def get_location_from_ip():
    # Try to get the client's IP address
    if request.headers.get('X-Forwarded-For'):
        client_ip = request.headers.get('X-Forwarded-For').split(',')[0]
    else:
        client_ip = request.remote_addr

    # Use the client's IP to get location data
    response = requests.get(f"https://ipinfo.io/{client_ip}/json")
    if response.status_code == 200:
        data = response.json()
        location = data.get('loc', '').split(',')
        if len(location) == 2:
            latitude, longitude = location[0], location[1]
            return latitude, longitude
    
    # Return None if unable to get location
    return None, None

# Example usage in a Flask route
@app.route('/get_user_location')
def get_user_location():
    latitude, longitude = get_location_from_ip()
    if latitude and longitude:
        return f"User location: Latitude {latitude}, Longitude {longitude}"
    else:
        return "Unable to determine user location"

@app.route('/')
def index():
    api_key = TICKETMASTER_API_KEY
    latitude, longitude = get_location_from_ip()
    return render_template('index.html', api_key=api_key, latitude=latitude, longitude=longitude)

@app.route('/get_location')
def get_location():
    latitude, longitude = get_location_from_ip()
    return jsonify({'latitude': latitude, 'longitude': longitude})

def identify_artist_name(message):
    ner_messages = [
        {"role": "system", "content": "You are an expert in named entity recognition. Your task is to identify any music artist names mentioned in the given text. If an artist name is found, respond with ONLY the artist name. If no artist name is found, respond with 'No artist found'."},
        {"role": "user", "content": f"Identify any music artist names in this text: {message}"}
    ]
    
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=ner_messages,
            temperature=0.3,
            max_tokens=50
        )
        artist_name = response.choices[0].message.content.strip()
        return None if artist_name == "No artist found" else artist_name
    except Exception as e:
        print(f"Error in artist name identification: {e}")
        return None

def chat_gpt_helper(messages):
    try:
        full_response = ''
        for chunk in client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            temperature=0.0,
            top_p=1.0,
            frequency_penalty=0.0,
            presence_penalty=0.0,
            stream=True,
        ):
            content = chunk.choices[0].delta.content
            if content is not None:
                print(content, end='')
                full_response += content
                encoded_content = content.replace('\n', '\\n').replace('\r', '\\r')
                yield f'data: {encoded_content}\n\n'
        
        session['conversation_history'].append({"role": "assistant", "content": full_response})
        session.modified = True
    except Exception as e:
        print(e)
        yield f'data: Error: {str(e)}\n\n'

@app.route('/ask', methods=['POST'])
def ask():
    data = request.json
    user_message = data.get('message')
    user_address = data.get('userAddress')
    city = data.get('city')

    if 'conversation_history' not in session:
        session['conversation_history'] = []
    
    if 'current_artist' not in session:
        session['current_artist'] = None

    # Identify artist name using ChatGPT
    artist_name = identify_artist_name(user_message)
    if artist_name:
        session['current_artist'] = artist_name

    # Create a clear location statement
    location_info = f"The user is located at {user_address}, {city}." if user_address and city else "The user's location is not provided."
    
    # Include the current artist and location in the message
    if session['current_artist']:
        full_message = f"{location_info} Regarding {session['current_artist']}: {user_message}"
    else:
        full_message = f"{location_info} User's question: {user_message}"

    session['conversation_history'].append({"role": "user", "content": full_message})

    # Create a system message that clearly states the location
    location_reminder = {
        "role": "system",
        "content": f"{location_info} Always consider this location information when answering questions or providing recommendations."
    }

    # Add a system message to reinforce focus on the current artist and include location
    current_context_message = {
        "role": "system",
        "content": f"The current artist being discussed is {session['current_artist']}. " +
                   "Focus all responses on this artist unless explicitly asked about another."
    }
    messages = [
        {"role": "system", "content": """You are a knowledgeable music expert assistant specializing in providing comprehensive information about music artists across various genres and eras. Your expertise includes: biographical details of artists, discographies and album information, musical style and influences, awards and achievements. You also have information about the user's location to provide relevant event recommendations. 

        Critical Instructions:
        - Always refer to the most recently mentioned artist in the conversation.
        - If multiple artists are mentioned in a single query, prioritize the last one mentioned.
        - Always start your response by directly answering the user's question about the most recent artist.
        - If the user hasn't asked about a specific artist, do not assume or introduce one.

        Context Management:
        - Keep track of the most recently discussed artist and use that as the primary context for follow-up questions.
        - If a new artist is mentioned, immediately switch your focus to that artist.
        - Only refer back to previously mentioned artists if explicitly asked or if making a direct comparison requested by the user.

        Important guidelines:
        - Provide concise, relevant answers without unnecessary repetition.
        - Never repeat information you've mentioned before unless specifically asked.
        - Focus solely on new information or details that directly answer the user's question.
        - If a question has been answered before, briefly acknowledge this and only add new insights if available.

        Format your responses for readability:
        - Use paragraphs to separate different topics or ideas.
        - Use bullet points for lists.
        - Use indentation where appropriate (e.g., for sub-points).
        - Insert line breaks to improve readability.
        - For emphasis, use single quotes. For example: 'Love Story'.
        - For song titles or album names, use single quotes. For example: 'Folklore' album.
        
        Remember: Your primary goal is to directly and concisely answer the user's specific question about the most recently mentioned artist without providing unrequested background information."""},
        location_reminder,
        current_context_message,
        *session['conversation_history']
    ]
    
    # - DO NOT provide any introductory or general information about an artist unless explicitly asked.

    return Response(stream_with_context(chat_gpt_helper(messages)),
                    mimetype='text/event-stream')
    
@app.route('/clear_chat', methods=['POST'])
def clear_chat():
    if 'conversation_history' in session:
        session['conversation_history'] = []
        session.modified = True
    return jsonify({'status': 'success'})

@app.route('/clear_session', methods=['POST'])
def clear_session():
    # Clear the entire session
    session.clear()
    return jsonify({"message": "Session cleared"}), 200

if __name__ == '__main__':
    app.run(port=8080, debug=True, threaded=True)