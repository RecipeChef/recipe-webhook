# app.py
import os
from flask import Flask, request, jsonify
from google.cloud import dialogflow_v2 as dialogflow
from google.oauth2 import service_account

app = Flask(__name__)

# Step 1: Load Dialogflow Credentials
CREDENTIALS_PATH = "/etc/secrets/dialogflow_key.json"  # This matches what you did in Render
PROJECT_ID = "recipechef-noml"  # <<< CHANGE THIS TO YOUR OWN project id

credentials = service_account.Credentials.from_service_account_file(CREDENTIALS_PATH)
session_client = dialogflow.SessionsClient(credentials=credentials)

# Step 2: Setup Flask Endpoint to talk to Dialogflow
@app.route('/chat', methods=['POST'])
def chat():
    try:
        user_message = request.json.get('message')
        if not user_message:
            return jsonify({"error": "No message provided"}), 400

        # Create a session
        session_id = "some-unique-session-id"  # You can generate a unique one if needed
        session = session_client.session_path(PROJECT_ID, session_id)

        # Prepare the text input
        text_input = dialogflow.TextInput(text=user_message, language_code="en")
        query_input = dialogflow.QueryInput(text=text_input)

        # Send request to Dialogflow
        response = session_client.detect_intent(session=session, query_input=query_input)

        # Get the chatbot reply
        fulfillment_text = response.query_result.fulfillment_text

        return jsonify({'reply': fulfillment_text})
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Step 3: Start the Flask server (only locally, Render will manage otherwise)
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
