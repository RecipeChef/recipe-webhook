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
import firebase_admin
from firebase_admin import credentials, firestore

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

# === Firebase Setup ===
# cred = credentials.Certificate("firebase_key.json")
# firebase_admin.initialize_app(cred)
# db = firestore.client()

# === /chat ===
@app.route('/chat', methods=['POST'])
def chat():
    user_message = request.json.get('message')
    
    if not user_message:
        return jsonify({"error": "Missing 'message' field in request"}), 400
        
    session_id = "user-session-id"

    session = dialogflow_session_client.session_path(DIALOGFLOW_PROJECT_ID, session_id)
    text_input = dialogflow.TextInput(text=user_message, language_code="en")
    query_input = dialogflow.QueryInput(text=text_input)
    response = dialogflow_session_client.detect_intent(session=session, query_input=query_input)

    intent_name = response.query_result.intent.display_name

    if intent_name == "MoreRecipesIntent":
        return handle_more_recipes(session_id)

    elif intent_name == "TextIngredientsIntent":
        # Extract ingredients from the raw message text
        user_message_lower = user_message.lower()

        # Clean common phrases
        for prefix in ["what can i cook with", "what can i make with", "how can i cook with"]:
            if user_message_lower.startswith(prefix):
                user_message_lower = user_message_lower.replace(prefix, "")

        # Example: Extract comma-separated words
        ingredients = [i.strip() for i in user_message_lower.replace("and", ",").split(",") if i.strip()]
    
        logging.info(f"[chat] Parsed ingredients from text: {ingredients}")
        
        USER_STATE[session_id] = {
            "ingredients": ingredients,
            "shown_recipe_ids": [],
            "request_count": 0
        }
        # Reuse /recipe-suggestions logic
        request_data = {"ingredients": ingredients, "session_id": session_id}
        with app.test_request_context('/recipe-suggestions', method='POST', json=request_data):
            return recipe_suggestions()
            
    elif intent_name == "WhatCanICookTodayIntent":
        user_id = session_id  # Or extract from request if you support real user IDs
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
        
    # else:
    # return jsonify({'reply': body['queryResult']['fulfillmentText']})
    return jsonify({'reply': response.query_result.fulfillment_text})
    

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
        
        # request_count = USER_STATE[session_id].get("request_count", 0) # added 04/05/2025. Made comment line 06/05/2025
        # ranking = 2 if request_count < 3 else 1 # added 04/05/2025. Made comment line 06/05/2025
        # logging.info(f"[recipe-suggestions] Ranking value: {ranking}") #added to see on render.Made comment line 06/05/2025
        # USER_STATE[session_id]["request_count"] = request_count + 1 # added 04/05/2025. Made comment line 06/05/2025

        # ranking = data.get('ranking') #Added this lines instead of commented lines till. Commented 06/05/2025 in 17.51
        # if ranking is not None:
        #     ranking = int(ranking)
        # else:
        #     request_count = USER_STATE[session_id].get("request_count", 0)
        #     ranking = 2 if request_count < 3 else 1
        #     USER_STATE[session_id]["request_count"] = request_count + 1
            
        # logging.info(f"[recipe-suggestions] Ranking value: {ranking}") #here

        # params = {
        #     "ingredients": ",".join(ingredients),
        #     "number": 60, #changed to 60 from 15
        #     "ranking": ranking, # added 04/05/2025
        #     "ignorePantry": True,
        #     "sort": "random", #"sort": "random",
        #     "apiKey": SPOONACULAR_API_KEY
        # }

        # new_recipes = []
        # attempts = 0

        # while len(new_recipes) < 10 and attempts < 5: #changed from 5 to 10
        #     response = requests.get(url, params=params)
        #     recipes_data = response.json()
        #     # recipes_data.sort( #This will be deleted
        #     #     key=lambda r: (-len(r.get("usedIngredients", [])), len(r.get("missedIngredients", [])))
        #     # ) #.

        #     for recipe in recipes_data:
        #         if recipe["id"] not in already_shown:
        #             new_recipes.append({
        #                 "id": recipe["id"],
        #                 "title": recipe["title"],
        #                 "image": recipe["image"],
        #                 # "usedIngredients": [i["name"] for i in recipe.get("usedIngredients", [])],
        #                 "usedIngredients": [{"id": i["id"], "name": i["name"]} for i in recipe.get("usedIngredients", [])], #UsedIngredients id's will be sent to Flutter
        #                 # "missedIngredients": [i["name"] for i in recipe.get("missedIngredients", [])]
        #                 "missedIngredients": [{"id": i["id"], "name": i["name"]} for i in recipe.get("missedIngredients", [])] #MissedIngredients id's will be sent to Flutter
        #             })
        #             already_shown.add(recipe["id"])
        #             if len(new_recipes) == 10: #changed from 5 to 10
        #                 break
        #     attempts += 1 #Commented 06/05/2025 in 17.51

        basic_recipes = [] # Added 06/05/2025 in 17.51
        complex_recipes = []
        already_shown = set(USER_STATE[session_id].get("shown_recipe_ids", []))
        
        for rank_value, storage_key in [(2, "basic_recipes"), (1, "complex_recipes")]:
            url = "https://api.spoonacular.com/recipes/findByIngredients"
            params = {
                "ingredients": ",".join(ingredients),
                "number": 60,
                "ranking": rank_value,
                "ignorePantry": True,
                "sort": "random",
                "apiKey": SPOONACULAR_API_KEY
            }
        
            temp_recipes = []
            attempts = 0
            while len(temp_recipes) < 10 and attempts < 5:
                response = requests.get(url, params=params)
                recipes_data = response.json()
                for recipe in recipes_data:
                    if recipe["id"] not in already_shown:
                        temp_recipes.append({
                            "id": recipe["id"],
                            "title": recipe["title"],
                            "image": recipe["image"],
                            "usedIngredients": [{"id": i["id"], "name": i["name"]} for i in recipe.get("usedIngredients", [])],
                            "missedIngredients": [{"id": i["id"], "name": i["name"]} for i in recipe.get("missedIngredients", [])]
                        })
                        already_shown.add(recipe["id"])
                        if len(temp_recipes) == 10:
                            break
                attempts += 1
        
            USER_STATE[session_id][storage_key] = temp_recipes # Added 06/05/2025 in 17.51
        USER_STATE[session_id]["shown_recipe_ids"] += [ # Added 06/05/2025 in 17.51
            r["id"] for r in basic_recipes + complex_recipes # Added 06/05/2025 in 17.51
            if r["id"] not in USER_STATE[session_id]["shown_recipe_ids"] # Added 06/05/2025 in 17.51
        ] # Added 06/05/2025 in 17.51



        USER_STATE[session_id].setdefault("shown_recipe_ids", [])
        # USER_STATE[session_id]["shown_recipe_ids"] += [r["id"] for r in new_recipes if r["id"] not in USER_STATE[session_id]["shown_recipe_ids"]] #Commented 06/05/2025 in 17.51
        USER_STATE[session_id]["shown_recipe_ids"] += [
            r["id"]
            for r in USER_STATE[session_id].get("basic_recipes", []) + USER_STATE[session_id].get("complex_recipes", [])
            if r["id"] not in USER_STATE[session_id]["shown_recipe_ids"]
        ]

        
        logging.info(f"[recipe-suggestions] Returned {len(new_recipes)} new recipes") #added to see on render
        logging.info(f"[recipe-suggestions] Recipe IDs: {[r['id'] for r in new_recipes]}") #added to see on render
        # return jsonify({"recipes": new_recipes}) #Commented 06/05/2025 in 17.51
        return jsonify({
            "basic_recipes": USER_STATE[session_id].get("basic_recipes", []),
            "complex_recipes": USER_STATE[session_id].get("complex_recipes", [])
        })


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

        # request_count = user_data.get("request_count", 0) # added 04/05/2025. Commented 06/05/2025 at 17.37
        # ranking = 2 if request_count < 3 else 1 # added 04/05/2025. Made comment line 06/05/2025
        # requested_ranking = request.json.get("ranking") #Added this instead of commented lines till. Commented 06/05/2025 at 17.37
        # if requested_ranking in [1, 2]:
        #     ranking = requested_ranking
        # else:
        #     ranking = 2 if request_count < 3 else 1 #here. continue

        # logging.info(f"[handle_more_recipes] Ranking value: {ranking}") #added to see on render
        # user_data["request_count"] = request_count + 1 # added 04/05/2025. here

        basic_recipes = []
        complex_recipes = []
        
        for rank_value, storage_key in [(2, "basic_recipes"), (1, "complex_recipes")]:
            url = "https://api.spoonacular.com/recipes/findByIngredients"
            params = {
                "ingredients": ",".join(ingredients),
                "number": 60,
                "ranking": rank_value,
                "ignorePantry": True,
                "sort": "random",
                "apiKey": SPOONACULAR_API_KEY
            }
            attempts = 0
            temp_recipes = []
            while len(temp_recipes) < 10 and attempts < 5:
                response = requests.get(url, params=params)
                recipes_data = response.json()
                for recipe in recipes_data:
                    if recipe["id"] not in already_shown:
                        temp_recipes.append({
                            "id": recipe["id"],
                            "title": recipe["title"],
                            "image": recipe["image"],
                            "usedIngredients": [{"id": i["id"], "name": i["name"]} for i in recipe.get("usedIngredients", [])],
                            "missedIngredients": [{"id": i["id"], "name": i["name"]} for i in recipe.get("missedIngredients", [])]
                        })
                        already_shown.add(recipe["id"])
                        if len(temp_recipes) == 10:
                            break
                attempts += 1
        
            user_data[storage_key] = temp_recipes
        USER_STATE[session_id].setdefault("shown_recipe_ids", [])
        USER_STATE[session_id]["shown_recipe_ids"] += [
            r["id"] for r in basic_recipes + complex_recipes 
            if r["id"] not in USER_STATE[session_id]["shown_recipe_ids"]
        ]


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
        return jsonify({
            "reply": "Here are new recipe suggestions for both basic and complex!",
            "basic_recipes": user_data.get("basic_recipes", []),
            "complex_recipes": user_data.get("complex_recipes", [])
        })
        # return jsonify({"reply": f"Here are more recipe suggestions! (IDs: {recipe_ids})", "recipes": new_recipes}) #Commented 06/05/2025 at 17.46


    except Exception as e:
        logging.exception("More recipe fetch failed")
        return jsonify({"error": str(e)}), 500

def get_user_recipe_ids(user_id):
    try:
        user_doc = db.collection("users").document(user_id)
        
        favorites_doc = user_doc.collection("data").document("favorites").get()
        planner_doc = user_doc.collection("data").document("meal_planner").get()

        favorite_ids = favorites_doc.to_dict().get("recipe_ids", []) if favorites_doc.exists else []
        planner_ids = planner_doc.to_dict().get("recipe_ids", []) if planner_doc.exists else []

        all_ids = list(set(favorite_ids + planner_ids))  # deduplicate
        logging.info(f"[Firebase] User {user_id} recipes from Firebase: {all_ids}")
        return all_ids

    except Exception as e:
        logging.exception("Failed to fetch Firebase recipe IDs")
        return []
        
def recommend_from_favorites(session_id):
    user_id = session_id  # Assuming session_id is the Firebase user ID
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

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
