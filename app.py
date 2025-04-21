from flask import Flask, request, jsonify
import requests
import base64
import logging
from PIL import Image
import io
from clarifai_grpc.channel.clarifai_channel import ClarifaiChannel
from clarifai_grpc.grpc.api import service_pb2, service_pb2_grpc
from clarifai_grpc.grpc.api import resources_pb2
from clarifai_grpc.grpc.api.status import status_code_pb2
from google.generativeai import configure, GenerativeModel  # ✅ Gemini import

# Logging
logging.basicConfig(level=logging.INFO)
app = Flask(__name__)

# API Keys
CLARIFAI_API_KEY = "4a4ea9088cfa42c29e63f7b6806ad272"
SPOONACULAR_API_KEY = "b97364cb57314c0fb18b8d7e93d7e5fc"
GEMINI_API_KEY = "AIzaSyBsSAzqCApmUMVyCkxmj1VBmZOPuTYf6eM"  # ✅ Add your Gemini API key here

# Gemini Configuration ✅
configure(api_key=GEMINI_API_KEY)
gemini_model = GenerativeModel("models/gemini-1.5-flash")

# Clarifai setup
channel = ClarifaiChannel.get_grpc_channel()
stub = service_pb2_grpc.V2Stub(channel)
metadata = (("authorization", f"Key {CLARIFAI_API_KEY}"),)

UNWANTED_WORDS = {"pasture", "micronutrient", "aliment", "comestible"}
CONFIDENCE_THRESHOLD = 0.5
RECIPE_CACHE = []
TEMP_INGREDIENTS = []
LAST_RECIPE_SHOWN = None

# Resize helper
def safely_resize_base64(base64_str, max_size=(300, 300)):
    base64_str = base64_str.strip().replace("\n", "").replace("\r", "")
    base64_str += "=" * ((4 - len(base64_str) % 4) % 4)
    try:
        image_data = base64.b64decode(base64_str)
        img = Image.open(io.BytesIO(image_data))
        img.thumbnail(max_size)
        buffered = io.BytesIO()
        img.save(buffered, format="JPEG")
        return base64.b64encode(buffered.getvalue()).decode("utf-8")
    except Exception as e:
        logging.error(f"Image resizing error: {e}")
        return base64_str

# Clarifai recognition
def recognize_ingredients_from_base64(base64_image):
    base64_image = safely_resize_base64(base64_image)
    base64_image += "=" * ((4 - len(base64_image) % 4) % 4)
    image_bytes = base64.b64decode(base64_image)
    request = service_pb2.PostModelOutputsRequest(
        model_id="food-item-v1-recognition",
        inputs=[resources_pb2.Input(data=resources_pb2.Data(image=resources_pb2.Image(base64=image_bytes)))]
    )
    response = stub.PostModelOutputs(request, metadata=metadata)
    if response.status.code != status_code_pb2.SUCCESS:
        return []
    return [
        concept.name.lower()
        for concept in response.outputs[0].data.concepts
        if concept.value >= CONFIDENCE_THRESHOLD and concept.name.lower() not in UNWANTED_WORDS
    ]

# Spoonacular utilities
def get_recipe_details(recipe_id):
    url = f"https://api.spoonacular.com/recipes/{recipe_id}/information?apiKey={SPOONACULAR_API_KEY}"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        return {
            "title": data.get("title"),
            "sourceUrl": data.get("sourceUrl", ""),
            "ingredients": [i["original"].lower() for i in data.get("extendedIngredients", [])],
            "instructions": data.get("instructions", ""),
            "readyInMinutes": data.get("readyInMinutes", "N/A"),
            "servings": data.get("servings", "N/A")
        }
    return None

def get_recipes(ingredients):
    ingredients_query = ",".join(ingredients)
    url = f"https://api.spoonacular.com/recipes/complexSearch?includeIngredients={ingredients_query}&number=25&apiKey={SPOONACULAR_API_KEY}"
    response = requests.get(url)
    if response.status_code != 200:
        logging.error(f"Spoonacular error: {response.status_code} - {response.text}")
        return []

    raw_recipes = response.json().get("results", [])
    matched = []
    for recipe in raw_recipes:
        details = get_recipe_details(recipe["id"])
        if not details:
            continue
        lower_ings = [ing.lower() for ing in details.get("ingredients", [])]
        if all(any(i in ing for ing in lower_ings) for i in ingredients):
            matched.append(details)
        if len(matched) >= 5:
            break
    return matched

# 🔧 Gemini fallback function
def handle_with_gemini_fallback(user_query):
    try:
        response = gemini_model.generate_content(user_query)
        return response.text.strip()
    except Exception as e:
        logging.error(f"Gemini error: {e}")
        return "I couldn't answer that question right now."

# Main webhook
@app.route("/webhook", methods=["POST"])
def webhook():
    global RECIPE_CACHE, TEMP_INGREDIENTS
    req = request.get_json()
    intent = req["queryResult"]["intent"]["displayName"]
    parameters = req["queryResult"].get("parameters", {})

    if intent == "UploadImageIntent":
        base64_image = parameters.get("imageBase64")
        TEMP_INGREDIENTS = recognize_ingredients_from_base64(base64_image)
        if TEMP_INGREDIENTS:
            return jsonify({
                "fulfillmentText": f"I found these ingredients: {', '.join(TEMP_INGREDIENTS)}. Would you like to add or remove any?"
            })
        else:
            return jsonify({"fulfillmentText": "No ingredients found in the image."})

    elif intent == "ConfirmIngredientsIntent":
        add_list = parameters.get("addList", "")
        remove_list = parameters.get("removeList", "")
        if remove_list:
            for item in remove_list.lower().split(","):
                if item.strip() in TEMP_INGREDIENTS:
                    TEMP_INGREDIENTS.remove(item.strip())
        if add_list:
            for item in add_list.lower().split(","):
                if item.strip() and item.strip() not in TEMP_INGREDIENTS:
                    TEMP_INGREDIENTS.append(item.strip())
        if TEMP_INGREDIENTS:
            return jsonify({"fulfillmentText": f"Updated ingredients: {', '.join(TEMP_INGREDIENTS)}. Should I find recipes?"})
        else:
            return jsonify({"fulfillmentText": "No ingredients left after changes."})

    elif intent == "GetRecipesIntent":
        raw = parameters.get("ingredients", [])
        ingredients = [i.strip().lower() for i in raw.split(" and ")] if isinstance(raw, str) else raw
        if not ingredients:
            ingredients = TEMP_INGREDIENTS
        RECIPE_CACHE = get_recipes(ingredients)
        if RECIPE_CACHE:
            response_text = "\n".join([f"{i+1}. {r['title']} - {r['sourceUrl']}" for i, r in enumerate(RECIPE_CACHE)])
        else:
            response_text = "Sorry, no recipes found with all those ingredients."
        return jsonify({"fulfillmentText": response_text})

    elif intent == "ShowRecipeDetailsIntent":
        recipe_number = parameters.get("recipeNumber")
        recipe_name = parameters.get("recipeName", "").strip().lower()
        recipe = None
    
        if recipe_number:
            recipe_number = int(recipe_number)
            if 1 <= recipe_number <= len(RECIPE_CACHE):
                recipe = RECIPE_CACHE[recipe_number - 1]
        elif recipe_name:
            for r in RECIPE_CACHE:
                if r["title"].lower() == recipe_name:
                    recipe = r
                    break
        if recipe:
            LAST_RECIPE_SHOWN = recipe
            return jsonify({
                "fulfillmentText": (
                    f"🍽️ {recipe['title']}\n"
                    f"🕒 Ready in: {recipe['readyInMinutes']} mins | Servings: {recipe['servings']}\n"
                    f"📋 Ingredients:\n" + "\n".join(recipe['ingredients']) +
                    f"\n🧑‍🍳 Instructions:\n{recipe['instructions']}\n🔗 {recipe['sourceUrl']}"
                )
            })
        return jsonify({"fulfillmentText": "Recipe not found. Try a different number or name."})

    elif intent == "RandomRecipeIntent":
        url = f"https://api.spoonacular.com/recipes/random?number=5&apiKey={SPOONACULAR_API_KEY}"
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            RECIPE_CACHE.clear()
            for r in data.get("recipes", []):
                RECIPE_CACHE.append({
                    "title": r["title"],
                    "sourceUrl": r.get("sourceUrl", "No URL"),
                    "readyInMinutes": r.get("readyInMinutes", "N/A"),
                    "servings": r.get("servings", "N/A"),
                    "ingredients": [i["original"] for i in r.get("extendedIngredients", [])],
                    "instructions": r.get("instructions", "Instructions not available.")
                })
            titles = "\n".join([f"{i+1}. {r['title']}" for i, r in enumerate(RECIPE_CACHE)])
            return jsonify({"fulfillmentText": f"🍽️ Here are 5 random recipes:\n{titles}"})
        return jsonify({"fulfillmentText": "Couldn't fetch random recipes right now."})

    # ✅ Gemini fallback
    elif intent == "Default Fallback Intent":
        fallback_question = req["queryResult"]["queryText"]

        # Try to give Gemini more context
        meal_context = ""
        if LAST_RECIPE_SHOWN:
            meal_context = (
                f"The user just asked: '{fallback_question}'\n"
                f"The last meal was: {LAST_RECIPE_SHOWN['title']}\n"
                f"Ingredients: {', '.join(LAST_RECIPE_SHOWN['ingredients'])}\n"
                f"Instructions: {LAST_RECIPE_SHOWN['instructions']}\n"
                f"Suggest a drink pairing for this meal or answer the question."
            )
        else:
            meal_context = f"The user just asked: '{fallback_question}'. Try to help with nourishment advice."

        try:
            response = gemini_model.generate_content(meal_context)
            return jsonify({"fulfillmentText": response.text.strip()})
        except Exception as e:
            logging.error(f"Gemini error: {e}")
            return jsonify({"fulfillmentText": "I'm still learning. Let me try again or ask something else!"})


    return jsonify({"fulfillmentText": "Sorry, I didn't understand. Try uploading an image or asking for a recipe."})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
