from flask import Flask, request, jsonify
from PIL import Image
import base64
import io
import logging
import requests
from google.cloud import dialogflow_v2 as dialogflow
from google.oauth2 import service_account
from clarifai_grpc.channel.clarifai_channel import ClarifaiChannel
from clarifai_grpc.grpc.api import service_pb2, service_pb2_grpc
from clarifai_grpc.grpc.api import resources_pb2
from clarifai_grpc.grpc.api.status import status_code_pb2

# App setup
app = Flask(__name__)
logging.basicConfig(level=logging.INFO)

# === 🔐 Dialogflow Setup ===
DIALOGFLOW_PROJECT_ID = "recipechef-noml"  # <== CHANGE THIS!
DIALOGFLOW_CREDENTIALS = service_account.Credentials.from_service_account_file(
    "/etc/secrets/dialogflow_key.json"
)
dialogflow_session_client = dialogflow.SessionsClient(credentials=DIALOGFLOW_CREDENTIALS)

# === 🔐 Clarifai Setup ===
CLARIFAI_API_KEY = "4a4ea9088cfa42c29e63f7b6806ad272"  # <== CHANGE THIS!
clarifai_channel = ClarifaiChannel.get_grpc_channel()
clarifai_stub = service_pb2_grpc.V2Stub(clarifai_channel)
clarifai_metadata = (("authorization", f"Key {CLARIFAI_API_KEY}"),)

# === 🔐 Spoonacular Setup ===
SPOONACULAR_API_KEY = "your-spoonacular-api-key"  # <== CHANGE THIS!

# === /chat ===
@app.route('/chat', methods=['POST'])
def chat():
    user_message = request.json.get('message')
    session_id = "user-session-id"

    session = dialogflow_session_client.session_path(DIALOGFLOW_PROJECT_ID, session_id)
    text_input = dialogflow.TextInput(text=user_message, language_code="en")
    query_input = dialogflow.QueryInput(text=text_input)
    response = dialogflow_session_client.detect_intent(session=session, query_input=query_input)

    return jsonify({'reply': response.query_result.fulfillment_text})


# === /analyze-image ===
@app.route('/analyze-image', methods=['POST'])
def analyze_image():
    try:
        # Unwanted words to filter out from Clarifai predictions
        UNWANTED_WORDS = {"aliment", "micronutrient", "pasture", "comestible"}

        # 1. Receive image file from Flutter
        image_file = request.files['file']
        image = Image.open(image_file.stream)

        # 2. Resize to 300x300
        resized = image.resize((300, 300))

        # 3. Convert to base64
        buffered = io.BytesIO()
        resized.save(buffered, format="JPEG")
        image_bytes = buffered.getvalue()
        image_base64 = base64.b64encode(image_bytes)

        # 4. Send to Clarifai
        request_clarifai = service_pb2.PostModelOutputsRequest(
            model_id="food-item-v1-recognition",
            inputs=[
                resources_pb2.Input(
                    data=resources_pb2.Data(
                        image=resources_pb2.Image(base64=image_base64)
                    )
                )
            ]
        )
        response = clarifai_stub.PostModelOutputs(request_clarifai, metadata=clarifai_metadata)

        ingredients = []
        if response.status.code == status_code_pb2.SUCCESS:
            for concept in response.outputs[0].data.concepts:
                if concept.value > 0.5 and concept.name not in UNWANTED_WORDS:
                    ingredients.append(concept.name)

        return jsonify({"ingredients": ingredients})
    except Exception as e:
        logging.exception("Clarifai image analysis failed")
        return jsonify({"error": str(e)}), 500


# === /recipe-suggestions ===
@app.route('/recipe-suggestions', methods=['POST'])
def recipe_suggestions():
    try:
        ingredients = request.json.get('ingredients', [])
        ingredients_str = ",".join(ingredients)

        url = f"https://api.spoonacular.com/recipes/findByIngredients"
        params = {
            "ingredients": ingredients_str,
            "number": 5,
            "ranking": 1,
            "ignorePantry": True,
            "apiKey": SPOONACULAR_API_KEY
        }

        response = requests.get(url, params=params)
        recipes_data = response.json()

        recipes = []
        for r in recipes_data:
            recipes.append({
                "title": r.get("title"),
                "id": r.get("id"),
                "image": r.get("image"),
                "usedIngredients": [i["name"] for i in r.get("usedIngredients", [])],
                "missedIngredients": [i["name"] for i in r.get("missedIngredients", [])]
            })

        return jsonify({"recipes": recipes})
    except Exception as e:
        logging.exception("Recipe fetch failed")
        return jsonify({"error": str(e)}), 500


# === Run Flask server ===
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
