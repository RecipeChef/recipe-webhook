from flask import Flask, request, jsonify #Yeni
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
SPOONACULAR_API_KEY = "b97364cb57314c0fb18b8d7e93d7e5fc"  # <== CHANGE THIS!
#SPOONACULAR_API_KEY = "d9aa75aad8ba43eebbe69f973e4433eb"
# === 🌟 In-memory user session state ===
USER_STATE = {}

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
        UNWANTED_WORDS = {"aliment", "micronutrient", "pasture", "comestible"}
        CONFIDENCE_THRESHOLD = 0.5

        # 1. Get image from Flutter
        image_file = request.files['file']
        image = Image.open(image_file.stream).convert("RGB")
        resized = image.resize((300, 300))
        logging.info(f"Image resized to: {resized.size}")

        # 2. Save to bytes directly (NO base64!)
        buffered = io.BytesIO()
        resized.save(buffered, format="JPEG")
        image_bytes = buffered.getvalue()

        # 3. Send raw image bytes to Clarifai
        request_clarifai = service_pb2.PostModelOutputsRequest(
            model_id="food-item-v1-recognition",
            inputs=[
                resources_pb2.Input(
                    data=resources_pb2.Data(
                        image=resources_pb2.Image(base64=image_bytes)
                    )
                )
            ]
        )

        response = clarifai_stub.PostModelOutputs(request_clarifai, metadata=clarifai_metadata)

        if response.status.code != status_code_pb2.SUCCESS:
            logging.error(f"Clarifai model error: {response.status.description}")
            return jsonify({"error": "Clarifai model failed"}), 500

        # 4. Parse Clarifai response
        ingredients = []
        logging.info("Clarifai results:")
        for concept in response.outputs[0].data.concepts:
            logging.info(f"- {concept.name} ({concept.value:.2f})")
            if concept.value > CONFIDENCE_THRESHOLD and concept.name not in UNWANTED_WORDS:
                ingredients.append(concept.name)

        # Save ingredients in user state
        USER_STATE["user-session-id"] = {
            "ingredients": ingredients,
            "shown_recipe_ids": [],
            "chosen_recipe": None
        }

        return jsonify({"ingredients": ingredients})

    except Exception as e:
        logging.exception("Clarifai image analysis failed")
        return jsonify({"error": str(e)}), 500

# === /recipe-suggestions ===
@app.route('/recipe-suggestions', methods=['POST'])
def recipe_suggestions():
    try:
        data = request.json
        ingredients = data.get('ingredients', [])
        session_id = data.get('session_id', 'user-session-id')

        if not ingredients:
            return jsonify({"error": "No ingredients provided."}), 400

        if session_id not in USER_STATE:
            USER_STATE[session_id] = {"shown_recipe_ids": []}

        already_shown = set(USER_STATE[session_id].get("shown_recipe_ids", []))

        url = "https://api.spoonacular.com/recipes/findByIngredients"
        params = {
            "ingredients": ",".join(ingredients),
            "number": 10,
            "ranking": 1,
            "ignorePantry": True,
            "apiKey": SPOONACULAR_API_KEY
        }

        response = requests.get(url, params=params)
        recipes_data = response.json()

        new_recipes = []
        for recipe in recipes_data:
            if recipe["id"] not in already_shown:
                new_recipes.append({
                    "id": recipe["id"],
                    "title": recipe["title"],
                    "image": recipe["image"],
                    "usedIngredients": [i["name"] for i in recipe.get("usedIngredients", [])],
                    "missedIngredients": [i["name"] for i in recipe.get("missedIngredients", [])]
                })
                already_shown.add(recipe["id"])
            if len(new_recipes) == 5:
                break

        USER_STATE[session_id]["shown_recipe_ids"] = list(already_shown)
        return jsonify({"recipes": new_recipes})

    except Exception as e:
        logging.exception("Recipe fetch failed")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
