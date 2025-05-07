from flask import Flask, request, jsonify  # Yeni çalışan
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

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)

DIALOGFLOW_PROJECT_ID = "recipechef-noml"
DIALOGFLOW_CREDENTIALS = service_account.Credentials.from_service_account_file(
    "/etc/secrets/dialogflow_key.json"
)
dialogflow_session_client = dialogflow.SessionsClient(credentials=DIALOGFLOW_CREDENTIALS)

CLARIFAI_API_KEY = "4a4ea9088cfa42c29e63f7b6806ad272"
clarifai_channel = ClarifaiChannel.get_grpc_channel()
clarifai_stub = service_pb2_grpc.V2Stub(clarifai_channel)
clarifai_metadata = (("authorization", f"Key {CLARIFAI_API_KEY}"),)

SPOONACULAR_API_KEY = "b97364cb57314c0fb18b8d7e93d7e5fc"

USER_STATE = {}

cred = credentials.Certificate("/etc/secrets/firebase_key.json")
firebase_admin.initialize_app(cred)
db = firestore.client()

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
        user_message_lower = user_message.lower()
        for prefix in ["what can i cook with", "what can i make with", "how can i cook with"]:
            if user_message_lower.startswith(prefix):
                user_message_lower = user_message_lower.replace(prefix, "")

        ingredients = [i.strip() for i in user_message_lower.replace("and", ",").split(",") if i.strip()]
        logging.info(f"[chat] Parsed ingredients from text: {ingredients}")

        USER_STATE[session_id] = {
            "ingredients": ingredients,
            "shown_recipe_ids_by_complexity": {},
            "recipes_by_complexity": {},
            "complexity": "basic"
        }
        request_data = {"ingredients": ingredients, "session_id": session_id, "complexity": "basic"}
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

    return jsonify({'reply': response.query_result.fulfillment_text})

@app.route('/recipe-suggestions', methods=['POST'])
def recipe_suggestions():
    try:
        data = request.json
        ingredients = data.get('ingredients', [])
        session_id = data.get('session_id', 'user-session-id')
        complexity = data.get("complexity", "basic")

        if session_id not in USER_STATE:
            USER_STATE[session_id] = {}

        USER_STATE[session_id].setdefault("ingredients", ingredients)
        USER_STATE[session_id].setdefault("shown_recipe_ids_by_complexity", {})
        USER_STATE[session_id].setdefault("recipes_by_complexity", {})
        USER_STATE[session_id]["complexity"] = complexity

        already_shown = set(USER_STATE[session_id]["shown_recipe_ids_by_complexity"].get(complexity, []))
        ranking = 2 if complexity == "basic" else 1

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
            response = requests.get("https://api.spoonacular.com/recipes/findByIngredients", params=params)
            recipes_data = response.json()
            for recipe in recipes_data:
                if recipe["id"] not in already_shown:
                    new_recipes.append({
                        "id": recipe["id"],
                        "title": recipe["title"],
                        "image": recipe["image"],
                        "usedIngredients": [{"id": i["id"], "name": i["name"]} for i in recipe.get("usedIngredients", [])],
                        "missedIngredients": [{"id": i["id"], "name": i["name"]} for i in recipe.get("missedIngredients", [])]
                    })
                    already_shown.add(recipe["id"])
                    if len(new_recipes) == 10:
                        break
            attempts += 1

        if not new_recipes:
            return jsonify({"reply": "No new recipes found."})

        USER_STATE[session_id]["shown_recipe_ids_by_complexity"].setdefault(complexity, [])
        USER_STATE[session_id]["shown_recipe_ids_by_complexity"][complexity] += [r["id"] for r in new_recipes]
        USER_STATE[session_id]["recipes_by_complexity"][complexity] = new_recipes

        return jsonify({"recipes": new_recipes})

    except Exception as e:
        logging.exception("Recipe fetch failed")
        return jsonify({"error": str(e)}), 500


def handle_more_recipes(session_id):
    try:
        user_data = USER_STATE.get(session_id)
        if not user_data:
            return jsonify({"reply": "Sorry, I couldn't find your session. Please start over."})

        ingredients = user_data.get("ingredients", [])
        complexity = user_data.get("complexity", "basic")
        already_shown = set(user_data.get("shown_recipe_ids_by_complexity", {}).get(complexity, []))
        ranking = 2 if complexity == "basic" else 1

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
            response = requests.get("https://api.spoonacular.com/recipes/findByIngredients", params=params)
            recipes_data = response.json()
            for recipe in recipes_data:
                if recipe["id"] not in already_shown:
                    new_recipes.append({
                        "id": recipe["id"],
                        "title": recipe["title"],
                        "image": recipe["image"],
                        "usedIngredients": [{"id": i["id"], "name": i["name"]} for i in recipe.get("usedIngredients", [])],
                        "missedIngredients": [{"id": i["id"], "name": i["name"]} for i in recipe.get("missedIngredients", [])]
                    })
                    already_shown.add(recipe["id"])
                    if len(new_recipes) == 10:
                        break
            attempts += 1

        if not new_recipes:
            return jsonify({"reply": "No more new recipes found. Try changing ingredients!"})

        user_data.setdefault("shown_recipe_ids_by_complexity", {})
        user_data.setdefault("recipes_by_complexity", {})
        user_data["shown_recipe_ids_by_complexity"].setdefault(complexity, [])
        user_data["shown_recipe_ids_by_complexity"][complexity] += [r["id"] for r in new_recipes]
        user_data["recipes_by_complexity"][complexity] = new_recipes

        return jsonify({"reply": "Here are more recipe suggestions!", "recipes": new_recipes})

    except Exception as e:
        logging.exception("More recipe fetch failed")
        return jsonify({"error": str(e)}), 500
