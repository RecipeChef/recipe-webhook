from flask import Flask, request, jsonify #Yeni çalışan 
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
DIALOGFLOW_PROJECT_ID = "recipechef-noml"
DIALOGFLOW_CREDENTIALS = service_account.Credentials.from_service_account_file(
    "/etc/secrets/dialogflow_key.json"
)
dialogflow_session_client = dialogflow.SessionsClient(credentials=DIALOGFLOW_CREDENTIALS)

# === 🔐 Clarifai Setup ===
CLARIFAI_API_KEY = "4a4ea9088cfa42c29e63f7b6806ad272"
clarifai_channel = ClarifaiChannel.get_grpc_channel()
clarifai_stub = service_pb2_grpc.V2Stub(clarifai_channel)
clarifai_metadata = (("authorization", f"Key {CLARIFAI_API_KEY}"),)

# === 🔐 Spoonacular Setup ===
SPOONACULAR_API_KEY = "b97364cb57314c0fb18b8d7e93d7e5fc"

# === 🌟 In-memory user session state ===
USER_STATE = {}

# === /chat ===
@app.route('/chat', methods=['POST'])
def chat():
    # user_message = request.json.get('message')
    # session_id = "user-session-id"

    # session = dialogflow_session_client.session_path(DIALOGFLOW_PROJECT_ID, session_id)
    # text_input = dialogflow.TextInput(text=user_message, language_code="en")
    # query_input = dialogflow.QueryInput(text=text_input)
    # response = dialogflow_session_client.detect_intent(session=session, query_input=query_input)

    # intent_name = response.query_result.intent.display_name

    body = request.json
    session_id = "user-session-id"

    intent_name = body['queryResult']['intent']['displayName']
    parameters = body['queryResult']['parameters']

    if intent_name == "MoreRecipesIntent":
        return handle_more_recipes(session_id)

    elif intent_name == "TextIngredientsIntent":
        parameters = response.query_result.parameters
        ingredients = []

        if "ingredients" in parameters:
            raw_ingredients = parameters["ingredients"]
            if hasattr(raw_ingredients, 'list_value'):
                ingredients = [v.string_value.lower() for v in raw_ingredients.list_value.values]
        logging.info(f"[chat] Extracted ingredients from intent: {ingredients}")
        USER_STATE[session_id] = {
            "ingredients": ingredients,
            "shown_recipe_ids": [],
            "request_count": 0
        }
        # Reuse /recipe-suggestions logic
    request_data = {"ingredients": ingredients, "session_id": session_id}
    with app.test_request_context('/recipe-suggestions', method='POST', json=request_data):
        return recipe_suggestions()
    else:
        return jsonify({'reply': body['queryResult']['fulfillmentText']})
    # return jsonify({'reply': response.query_result.fulfillment_text})
    

# === /analyze-image ===
@app.route('/analyze-image', methods=['POST'])
def analyze_image():
    try:
        UNWANTED_WORDS = {"aliment", "micronutrient", "pasture", "comestible"}
        CONFIDENCE_THRESHOLD = 0.6

        image_file = request.files['image'] #changed from file to image
        image = Image.open(image_file.stream).convert("RGB")
        resized = image.resize((300, 300))
        logging.info(f"Image resized to: {resized.size}")

        buffered = io.BytesIO()
        resized.save(buffered, format="JPEG")
        image_bytes = buffered.getvalue()

        request_clarifai = service_pb2.PostModelOutputsRequest(
            model_id="food-item-v1-recognition", #changed from food-item-v1-recognition to food-item-recognition
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

        ingredients = []
        logging.info("Clarifai results:")
        for concept in response.outputs[0].data.concepts:
            logging.info(f"- {concept.name} ({concept.value:.2f})")
            if concept.value > CONFIDENCE_THRESHOLD and concept.name not in UNWANTED_WORDS:
                ingredients.append(concept.name)

        USER_STATE["user-session-id"] = {
            "ingredients": ingredients,
            "shown_recipe_ids": [],
            "chosen_recipe": None #It will be removed later
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
        
        logging.info(f"[recipe-suggestions] Session: {session_id}") #added to see on render
        logging.info(f"[recipe-suggestions] Ingredients used: {ingredients}") #added to see on render


        if not ingredients:
            return jsonify({"error": "No ingredients provided."}), 400

        # if session_id not in USER_STATE:
        #     USER_STATE[session_id] = {"shown_recipe_ids": [],
        #                              "ingredients": ingredients, # added 04/05/2025
        #                              "request_count": 0} # added 04/05/2025

        if session_id not in USER_STATE: #added for to remove unwanted ingredients till
            USER_STATE[session_id] = {}
        USER_STATE[session_id]["ingredients"] = ingredients
        USER_STATE[session_id].setdefault("shown_recipe_ids", [])
        USER_STATE[session_id].setdefault("request_count", 0) #here

        already_shown = set(USER_STATE[session_id].get("shown_recipe_ids", []))
        logging.info(f"[recipe-suggestions] Already shown for {session_id}: {already_shown}")

        url = "https://api.spoonacular.com/recipes/findByIngredients"
        
        request_count = USER_STATE[session_id].get("request_count", 0) # added 04/05/2025
        ranking = 2 if request_count < 3 else 1 # added 04/05/2025
        logging.info(f"[recipe-suggestions] Ranking value: {ranking}") #added to see on render
        USER_STATE[session_id]["request_count"] = request_count + 1 # added 04/05/2025

        params = {
            "ingredients": ",".join(ingredients),
            "number": 60, #changed to 60 from 15
            "ranking": ranking, # added 04/05/2025
            "ignorePantry": True,
            "sort": "random", #"sort": "random",
            "apiKey": SPOONACULAR_API_KEY
        }

        new_recipes = []
        attempts = 0

        while len(new_recipes) < 10 and attempts < 5: #changed from 5 to 10
            response = requests.get(url, params=params)
            recipes_data = response.json()
            # recipes_data.sort(
            #     key=lambda r: (-len(r.get("usedIngredients", [])), len(r.get("missedIngredients", [])))
            # )

            for recipe in recipes_data:
                if recipe["id"] not in already_shown:
                    new_recipes.append({
                        "id": recipe["id"],
                        "title": recipe["title"],
                        "image": recipe["image"],
                        # "usedIngredients": [i["name"] for i in recipe.get("usedIngredients", [])],
                        "usedIngredients": [{"id": i["id"], "name": i["name"]} for i in recipe.get("usedIngredients", [])], #UsedIngredients id's will be sent to Flutter
                        # "missedIngredients": [i["name"] for i in recipe.get("missedIngredients", [])]
                        "missedIngredients": [{"id": i["id"], "name": i["name"]} for i in recipe.get("missedIngredients", [])] #MissedIngredients id's will be sent to Flutter
                    })
                    already_shown.add(recipe["id"])
                    if len(new_recipes) == 10: #changed from 5 to 10
                        break
            attempts += 1

        USER_STATE[session_id].setdefault("shown_recipe_ids", [])
        USER_STATE[session_id]["shown_recipe_ids"] += [r["id"] for r in new_recipes if r["id"] not in USER_STATE[session_id]["shown_recipe_ids"]]
        
        logging.info(f"[recipe-suggestions] Returned {len(new_recipes)} new recipes") #added to see on render
        logging.info(f"[recipe-suggestions] Recipe IDs: {[r['id'] for r in new_recipes]}") #added to see on render
        return jsonify({"recipes": new_recipes})

    except Exception as e:
        logging.exception("Recipe fetch failed")
        return jsonify({"error": str(e)}), 500

# === /handle-more-recipes ===
def handle_more_recipes(session_id):
    try:
        user_data = USER_STATE.get(session_id)
        logging.info(f"User state for {session_id}: {user_data}") #added for test
        if not user_data or not user_data.get("ingredients"):
            return jsonify({"reply": "Sorry, I couldn't find your ingredients. Please send a new image."})

        ingredients = user_data["ingredients"]
        logging.info(f"[handle_more_recipes] Session: {session_id}") #added to see on render
        logging.info(f"[handle_more_recipes] Ingredients used: {ingredients}") # added to see on render
        if not ingredients: # Added for making sure to see recipes with the right ingredient list till
            return jsonify({"reply": "Your ingredient list seems empty. Please send a new one."}) #here
            
        already_shown = set(user_data.get("shown_recipe_ids", []))
        logging.info(f"[handle_more_recipes] Already shown for {session_id}: {already_shown}")

        request_count = user_data.get("request_count", 0) # added 04/05/2025
        ranking = 2 if request_count < 3 else 1 # added 04/05/2025
        logging.info(f"[handle_more_recipes] Ranking value: {ranking}") #added to see on render
        user_data["request_count"] = request_count + 1 # added 04/05/2025

        url = "https://api.spoonacular.com/recipes/findByIngredients"
        params = {
            "ingredients": ",".join(ingredients),
            "number": 60, #changed to 50 from 15
            "ranking": ranking, # added 04/05/2025
            "ignorePantry": True,
            "sort": "random", # "sort": "random"
            "apiKey": SPOONACULAR_API_KEY
        }

        new_recipes = []
        attempts = 0

        while len(new_recipes) < 10 and attempts < 5: #changed from 5 to 10. Besides it attempts to find 5 times best-matching recipes
            response = requests.get(url, params=params)
            recipes_data = response.json()
            # recipes_data.sort(
            #     key=lambda r: (-len(r.get("usedIngredients", [])), len(r.get("missedIngredients", [])))
            # ) #Added for less missing ingredients and more used ingredients

            for recipe in recipes_data:
                if recipe["id"] not in already_shown:
                    new_recipes.append({
                        "id": recipe["id"],
                        "title": recipe["title"],
                        "image": recipe["image"],
                        # "usedIngredients": [i["name"] for i in recipe.get("usedIngredients", [])],
                        "usedIngredients": [{"id": i["id"], "name": i["name"]} for i in recipe.get("usedIngredients", [])],
                        # "missedIngredients": [i["name"] for i in recipe.get("missedIngredients", [])]
                        "missedIngredients": [{"id": i["id"], "name": i["name"]} for i in recipe.get("missedIngredients", [])]
                    })
                    already_shown.add(recipe["id"])
                    if len(new_recipes) == 10: #changed from 5 to 10
                        break
            attempts += 1

        USER_STATE[session_id].setdefault("shown_recipe_ids", [])
        USER_STATE[session_id]["shown_recipe_ids"] += [r["id"] for r in new_recipes if r["id"] not in USER_STATE[session_id]["shown_recipe_ids"]]
        recipe_ids = [r["id"] for r in new_recipes]
        if not new_recipes:
            return jsonify({"reply": "I’ve already shown you all the matching recipes. Try new ingredients!", "recipes": []})
            
        logging.info(f"[handle_more_recipes] Returned {len(new_recipes)} new recipes") # added to see on render
        logging.info(f"[handle_more_recipes] Recipe IDs: {[r['id'] for r in new_recipes]}") # added to see on render
        return jsonify({"reply": f"Here are more recipe suggestions! (IDs: {recipe_ids})", "recipes": new_recipes})

    except Exception as e:
        logging.exception("More recipe fetch failed")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
