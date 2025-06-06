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
import firebase_admin
from firebase_admin import credentials, firestore
import os
import google.generativeai as genai
import re

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
SPOONACULAR_API_KEY = "d9aa75aad8ba43eebbe69f973e4433eb"
#"b97364cb57314c0fb18b8d7e93d7e5fc"

# === 🌟 In-memory user session state ===
USER_STATE = {}

# === Firebase Setup ===
cred = credentials.Certificate("/etc/secrets/firebase_key.json")
firebase_admin.initialize_app(cred)
db = firestore.client()

# === 🔐 Gemini Setup ===
genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))
gemini_model = genai.GenerativeModel("models/gemini-2.0-flash")

@app.route('/chat', methods=['POST'])
def chat():
    user_message = request.json.get('message')

    if not user_message:
        return jsonify({"error": "Missing 'message' field in request"}), 400

    
    session_id = request.json.get("session_id", "user-session-id")
    session = dialogflow_session_client.session_path(DIALOGFLOW_PROJECT_ID, session_id)
    text_input = dialogflow.TextInput(text=user_message, language_code="en")
    query_input = dialogflow.QueryInput(text=text_input)
    response = dialogflow_session_client.detect_intent(session=session, query_input=query_input)

    intent_name = response.query_result.intent.display_name

    if intent_name == "MoreRecipesIntent":
        return handle_more_recipes(session_id)

    elif intent_name == "TextIngredientsIntent":
        
        
        user_message_cleaned = re.sub(r'[^\w\s,]', '', user_message.lower())
        prefixes = [
            "what can i cook with",
            "what can i make with",
            "how can i cook with",
            "can you recommend me a recipe with"
        ]

        for prefix in prefixes:
            if user_message_cleaned.startswith(prefix):
                user_message_cleaned = user_message_cleaned[len(prefix):].strip()
                break
        
        user_message_cleaned = user_message_cleaned.replace(" and ", ",")
        raw_items = re.split(r'[,\s]+', user_message_cleaned)
        ingredients = [i.strip() for i in raw_items if i.strip()]
        

        logging.info(f"[chat] Parsed ingredients from text: {ingredients}")

        USER_STATE[session_id] = {
            "ingredients": ingredients,
            "shown_recipe_ids": [],
            "request_count": 0
        }
        
        request_data = {"ingredients": ingredients, "session_id": session_id}
        with app.test_request_context('/recipe-suggestions', method='POST', json=request_data):
            return recipe_suggestions()

    elif intent_name == "WhatCanICookTodayIntent":
        user_id = session_id  
        recipe_ids = get_user_recipe_ids(user_id)
        USER_STATE[session_id] = {
            "base_recipe_ids": recipe_ids,
            "shown_similar_ids": []
        }

        if not recipe_ids:
            return jsonify({"reply": "You have no favorite or planned recipes to base suggestions on."})
        collected_recipes = []
        seen_ids = set()

        for rid in recipe_ids:
            url = f"https://api.spoonacular.com/recipes/{rid}/similar"
            params = {"number": 5, "apiKey": SPOONACULAR_API_KEY}
            resp = requests.get(url, params=params)
            if resp.status_code != 200:
                continue
            for item in resp.json():
                if item["id"] not in seen_ids:
                    USER_STATE[session_id]["shown_similar_ids"].append(item["id"])
                    collected_recipes.append({
                        "id": item["id"],
                        "title": item["title"],
                        "image": f"https://spoonacular.com/recipeImages/{item['id']}-312x231.jpg"
                    })
                    seen_ids.add(item["id"])
                    if len(collected_recipes) == 10:
                        break
            if len(collected_recipes) == 10:
                break

        return jsonify({"reply": "Here are some ideas based on your taste!", "recipes": collected_recipes})

    elif intent_name == "Default Fallback Intent":
        logging.info(f"[chat] Falling back to Gemini for: {user_message}")
        return handle_gemini_fallback(session_id, user_message)

    return jsonify({'reply': response.query_result.fulfillment_text})


@app.route('/analyze-image', methods=['POST'])
def analyze_image():
    try:
        UNWANTED_WORDS = {"aliment", "micronutrient", "pasture", "comestible"}
        CONFIDENCE_THRESHOLD = 0.6

        image_file = request.files['image']  
        image = Image.open(image_file.stream).convert("RGB")
        resized = image.resize((300, 300))
        logging.info(f"Image resized to: {resized.size}")

        buffered = io.BytesIO()
        resized.save(buffered, format="JPEG")
        image_bytes = buffered.getvalue()

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

        ingredients = []
        logging.info("Clarifai results:")
        for concept in response.outputs[0].data.concepts:
            logging.info(f"- {concept.name} ({concept.value:.2f})")
            if concept.value > CONFIDENCE_THRESHOLD and concept.name not in UNWANTED_WORDS:
                ingredients.append(concept.name)

        USER_STATE["user-session-id"] = {
            "ingredients": ingredients,
            "shown_recipe_ids": [],
            "chosen_recipe": None  
        }

        return jsonify({"ingredients": ingredients})

    except Exception as e:
        logging.exception("Clarifai image analysis failed")
        return jsonify({"error": str(e)}), 500


@app.route('/recipe-suggestions', methods=['POST'])
def recipe_suggestions():
    try:
        data = request.json
        ingredients = data.get('ingredients', [])
        session_id = data.get('session_id', 'user-session-id')

        logging.info(f"[recipe-suggestions] Session: {session_id}")  # added to see on render
        logging.info(f"[recipe-suggestions] Ingredients used: {ingredients}")  # added to see on render

        if not ingredients:
            return jsonify({"error": "No ingredients provided."}), 400


        if session_id not in USER_STATE:  
            USER_STATE[session_id] = {}
        USER_STATE[session_id]["ingredients"] = ingredients
        USER_STATE[session_id].setdefault("shown_recipe_ids", [])
        USER_STATE[session_id].setdefault("request_count", 0)  

        already_shown = set(USER_STATE[session_id].get("shown_recipe_ids", []))
        logging.info(f"[recipe-suggestions] Already shown for {session_id}: {already_shown}")

        url = "https://api.spoonacular.com/recipes/findByIngredients"

        request_count = USER_STATE[session_id].get("request_count", 0)  # added 04/05/2025
        ranking = 2 if request_count < 3 else 1  # added 04/05/2025
        logging.info(f"[recipe-suggestions] Ranking value: {ranking}")  # added to see on render
        USER_STATE[session_id]["request_count"] = request_count + 1  # added 04/05/2025

        params = {
            "ingredients": ",".join(ingredients),
            "number": 60,  
            "ranking": ranking,  
            "ignorePantry": True,
            "sort": "random",  
            "apiKey": SPOONACULAR_API_KEY
        }

        new_recipes = []
        attempts = 0

        while len(new_recipes) < 10 and attempts < 5:  
            response = requests.get(url, params=params)
            recipes_data = response.json()
           

            for recipe in recipes_data:
                if recipe["id"] not in already_shown:
                    new_recipes.append({
                        "id": recipe["id"],
                        "title": recipe["title"],
                        "image": recipe["image"],
                        "usedIngredients": [{"id": i["id"], "name": i["name"]} for i in
                                            recipe.get("usedIngredients", [])],

                        "missedIngredients": [{"id": i["id"], "name": i["name"]} for i in
                                              recipe.get("missedIngredients", [])]
                        
                    })
                    already_shown.add(recipe["id"])
                    if len(new_recipes) == 10:  # changed from 5 to 10
                        break
            attempts += 1

        USER_STATE[session_id].setdefault("shown_recipe_ids", [])
        USER_STATE[session_id]["shown_recipe_ids"] += [r["id"] for r in new_recipes if
                                                       r["id"] not in USER_STATE[session_id]["shown_recipe_ids"]]

        logging.info(f"[recipe-suggestions] Returned {len(new_recipes)} new recipes")  # added to see on render
        logging.info(f"[recipe-suggestions] Recipe IDs: {[r['id'] for r in new_recipes]}")  # added to see on render
        return jsonify({"recipes": new_recipes})

    except Exception as e:
        logging.exception("Recipe fetch failed")
        return jsonify({"error": str(e)}), 500


def handle_more_recipes(session_id):
    try:
        user_data = USER_STATE.get(session_id)
        logging.info(f"User state for {session_id}: {user_data}")  # added for test
        
        if user_data and "base_recipe_ids" in user_data:
            return get_more_similar_recipes(session_id)
        if not user_data or not user_data.get("ingredients"):
            return jsonify({"reply": "Sorry, I couldn't find your ingredients. Please send a new image."})

        if "base_recipe_ids" in user_data:
            similar_recipes = get_more_similar_recipes(session_id)
            if not similar_recipes:
                return jsonify({"reply": "I’ve already shown you all the similar recipes I could find!"})
            return jsonify({"reply": "Here are more ideas based on your taste!", "recipes": similar_recipes})

        ingredients = user_data["ingredients"]
        logging.info(f"[handle_more_recipes] Session: {session_id}")  # added to see on render
        logging.info(f"[handle_more_recipes] Ingredients used: {ingredients}")  # added to see on render
        if not ingredients:  
            return jsonify({"reply": "Your ingredient list seems empty. Please send a new one."})  

        already_shown = set(user_data.get("shown_recipe_ids", []))
        logging.info(f"[handle_more_recipes] Already shown for {session_id}: {already_shown}")

        request_count = user_data.get("request_count", 0)  
        ranking = 2 if request_count < 2 else 1  
        logging.info(f"[handle_more_recipes] Ranking value: {ranking}")  
        user_data["request_count"] = request_count + 1  

        url = "https://api.spoonacular.com/recipes/findByIngredients"
        params = {
            "ingredients": ",".join(ingredients),
            "number": 60,  # changed to 50 from 15
            "ranking": ranking,  
            "ignorePantry": True,
            "sort": "random",  
            "apiKey": SPOONACULAR_API_KEY
        }

        new_recipes = []
        attempts = 0

        while len(
                new_recipes) < 10 and attempts < 5:  
            response = requests.get(url, params=params)
            recipes_data = response.json()

            for recipe in recipes_data:
                if recipe["id"] not in already_shown:
                    new_recipes.append({
                        "id": recipe["id"],
                        "title": recipe["title"],
                        "image": recipe["image"],
                        "usedIngredients": [{"id": i["id"], "name": i["name"]} for i in
                                            recipe.get("usedIngredients", [])],
                        "missedIngredients": [{"id": i["id"], "name": i["name"]} for i in
                                              recipe.get("missedIngredients", [])]
                    })
                    already_shown.add(recipe["id"])
                    if len(new_recipes) == 10:  # changed from 5 to 10
                        break
            attempts += 1

        USER_STATE[session_id].setdefault("shown_recipe_ids", [])
        USER_STATE[session_id]["shown_recipe_ids"] += [r["id"] for r in new_recipes if
                                                       r["id"] not in USER_STATE[session_id]["shown_recipe_ids"]]
        recipe_ids = [r["id"] for r in new_recipes]
        if not new_recipes:
            return jsonify(
                {"reply": "I’ve already shown you all the matching recipes. Try new ingredients!", "recipes": []})

        logging.info(f"[handle_more_recipes] Returned {len(new_recipes)} new recipes")  # added to see on render
        logging.info(f"[handle_more_recipes] Recipe IDs: {[r['id'] for r in new_recipes]}")  # added to see on render
        return jsonify({"reply": f"Here are more recipe suggestions!", "recipes": new_recipes})

    except Exception as e:
        logging.exception("More recipe fetch failed")
        return jsonify({"error": str(e)}), 500


def get_user_recipe_ids(user_id):
    try:
        favorite_ids = []
        planner_ids = []

       
        favorites_ref = db.collection("users").document(user_id).collection("favorites")
        favorites_docs = favorites_ref.stream()
        favorite_ids = [doc.id for doc in favorites_docs]

        planner_ref = db.collection("users").document(user_id).collection("mealplanner")
        planner_docs = planner_ref.stream()
        for doc in planner_docs:
            data = doc.to_dict()
            recipe_id = data.get("recipe_id")
            if recipe_id:
                planner_ids.append(recipe_id)

        all_ids = list(set(favorite_ids + planner_ids))  
        logging.info(f"[Firebase] User {user_id} recipes from Firebase: {all_ids}")
        return all_ids

    except Exception as e:
        logging.exception("Failed to fetch Firebase recipe IDs")
        return []


def recommend_from_favorites(session_id):
    user_id = session_id  
    recipe_ids = get_user_recipe_ids(user_id)

    if not recipe_ids:
        return jsonify({"reply": "You have no favorite or planned recipes to base suggestions on."})

    collected_recipes = []
    seen_ids = set()

    for rid in recipe_ids:
        url = f"https://api.spoonacular.com/recipes/{rid}/similar"
        params = {"number": 5, "apiKey": SPOONACULAR_API_KEY}
        resp = requests.get(url, params=params)
        if resp.status_code != 200:
            continue
        for item in resp.json():
            if item["id"] not in seen_ids:
                collected_recipes.append({
                    "id": item["id"],
                    "title": item["title"],
                    "image": f"https://spoonacular.com/recipeImages/{item['id']}-312x231.jpg"
                })
                seen_ids.add(item["id"])
                if len(collected_recipes) == 10:
                    break
                if len(collected_recipes) == 10:
                    break

    return jsonify({"reply": "Here are some ideas based on your taste!", "recipes": collected_recipes})


def get_more_similar_recipes(session_id):
    user_data = USER_STATE.get(session_id)
    if not user_data or "base_recipe_ids" not in user_data:
        return []

    collected_recipes = []
    seen_ids = set(user_data.get("shown_similar_ids", []))

    for rid in user_data["base_recipe_ids"]:
        url = f"https://api.spoonacular.com/recipes/{rid}/similar"
        params = {"number": 5, "apiKey": SPOONACULAR_API_KEY}
        resp = requests.get(url, params=params)
        if resp.status_code != 200:
            continue

        for item in resp.json():
            if item["id"] not in seen_ids:
                collected_recipes.append({
                    "id": item["id"],
                    "title": item["title"],
                    "image": f"https://spoonacular.com/recipeImages/{item['id']}-312x231.jpg"
                })
                seen_ids.add(item["id"])
                user_data["shown_similar_ids"].append(item["id"])

                if len(collected_recipes) == 10:
                    break
        if len(collected_recipes) == 10:
            break

    return collected_recipes
def handle_gemini_fallback(session_id, user_message):
    try:
        user_data = USER_STATE.get(session_id, {})
        recipe = user_data.get("chosen_recipe", {})
        ingredients = user_data.get("ingredients", [])

        context_parts = []

        if recipe:
            title = recipe.get("title", "this meal")
            context_parts.append(f"Recipe title: {title}")
            if "ingredients" in recipe:
                context_parts.append(f"Ingredients: {', '.join(recipe['ingredients'])}")
        elif ingredients:
            context_parts.append(f"Ingredients provided by the user: {', '.join(ingredients)}")
        else:
            context_parts.append("No specific meal or ingredients were provided.")

        
        prompt = (
            "You are a smart cooking assistant.\n"
            "You are only allowed to talk about nourishment-based questions with users.\n\n"
            "Do not talk about anything but nourishment-based questions.\n\n"
            "All your responses need to be detailed, readable, sincere, and more human.\n\n"
            "You may use emojis in your replies.\n\n"
            f"The user asked: \"{user_message}\"\n\n"
            f"Context:\n{chr(10).join(context_parts)}\n\n"
            "Provide a helpful, concise, and natural-language answer that would make sense "
            "in a meal planning or recipe suggestion app."
        )

        logging.info(f"[Gemini Fallback] Prompt to Gemini:\n{prompt}")
        response = gemini_model.generate_content(prompt)
        return jsonify({"reply": response.text})

    except Exception as e:
        logging.exception("Gemini fallback failed")
        return jsonify({"reply": "Sorry, I couldn't generate a response right now."})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
