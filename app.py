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

# Logging for debugging
logging.basicConfig(level=logging.INFO)
app = Flask(__name__)

# API keys
CLARIFAI_API_KEY = "4a4ea9088cfa42c29e63f7b6806ad272"
SPOONACULAR_API_KEY = "b97364cb57314c0fb18b8d7e93d7e5fc"

# Clarifai setup
channel = ClarifaiChannel.get_grpc_channel()
stub = service_pb2_grpc.V2Stub(channel)
metadata = (("authorization", f"Key {CLARIFAI_API_KEY}"),)

UNWANTED_WORDS = {"pasture", "micronutrient", "aliment", "comestible"}
CONFIDENCE_THRESHOLD = 0.5

RECIPE_CACHE = []
TEMP_INGREDIENTS = []

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

def recognize_ingredients_from_base64(base64_image):
    base64_image = safely_resize_base64(base64_image)
    base64_image = base64_image.strip().replace("\n", "").replace("\r", "")
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

def get_recipe_details(recipe_id):
    url = f"https://api.spoonacular.com/recipes/{recipe_id}/information?apiKey={SPOONACULAR_API_KEY}"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        return {
            "title": data.get("title"),
            "sourceUrl": data.get("sourceUrl", ""),
            "ingredients": [ing["original"].lower() for ing in data.get("extendedIngredients", [])],
            "instructions": data.get("instructions", "Instructions not available."),
            "readyInMinutes": data.get("readyInMinutes", "N/A"),
            "servings": data.get("servings", "N/A")
        }
    return None

def get_recipes(ingredients):
    def normalize(text):
        return text.lower().replace("-", " ").replace(",", "").strip()

    def all_ingredients_in_recipe(recipe_ingredients, user_ingredients):
        normalized_recipe_ings = [normalize(ri) for ri in recipe_ingredients]
        return all(
            any(user_ing in ri.split() for ri in normalized_recipe_ings)
            for user_ing in user_ingredients
        )

    ingredients_query = ",".join(ingredients)
    search_url = f"https://api.spoonacular.com/recipes/complexSearch?includeIngredients={ingredients_query}&number=15&apiKey={SPOONACULAR_API_KEY}"
    response = requests.get(search_url)

    if response.status_code != 200:
        logging.error("Recipe search failed.")
        return []

    raw_results = response.json().get("results", [])
    valid_recipes = []

    for recipe in raw_results:
        details = get_recipe_details(recipe["id"])
        if not details:
            continue
        if all_ingredients_in_recipe(details["ingredients"], ingredients):
            valid_recipes.append(details)
        if len(valid_recipes) >= 5:
            break

    return valid_recipes

@app.route("/webhook", methods=["POST"])
def webhook():
    global TEMP_INGREDIENTS, RECIPE_CACHE
    req = request.get_json()
    logging.info(f"Request: {req}")
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
            return jsonify({"fulfillmentText": "I couldn't detect any ingredients."})

    elif intent == "ConfirmIngredientsIntent":
        add_list = parameters.get("addList", "")
        remove_list = parameters.get("removeList", "")
        if remove_list:
            for item in remove_list.lower().split(","):
                item = item.strip()
                if item in TEMP_INGREDIENTS:
                    TEMP_INGREDIENTS.remove(item)
        if add_list:
            for item in add_list.lower().split(","):
                item = item.strip()
                if item and item not in TEMP_INGREDIENTS:
                    TEMP_INGREDIENTS.append(item)
        return jsonify({
            "fulfillmentText": f"Updated ingredients: {', '.join(TEMP_INGREDIENTS)}. Should I search for recipes now?"
        })

    elif intent == "GetRecipesIntent":
        raw = parameters.get("ingredients", [])
        ingredients = [i.strip().lower() for i in raw.split(" and ")] if isinstance(raw, str) else raw
        if not ingredients:
            ingredients = TEMP_INGREDIENTS
        RECIPE_CACHE = get_recipes(ingredients)
        if RECIPE_CACHE:
            text = "\n".join([f"{idx + 1}. {r['title']} - {r['sourceUrl']}" for idx, r in enumerate(RECIPE_CACHE)])
        else:
            text = "Sorry, I couldn't find any recipes with *all* of those ingredients."
        return jsonify({"fulfillmentText": text})

    elif intent == "ShowRecipeDetailsIntent":
        try:
            recipe_number = parameters.get("recipeNumber")
            recipe_name = parameters.get("recipeName", "").strip().lower()
            recipe = None
            if recipe_number:
                idx = int(recipe_number)
                if 1 <= idx <= len(RECIPE_CACHE):
                    recipe = RECIPE_CACHE[idx - 1]
            elif recipe_name:
                for r in RECIPE_CACHE:
                    if r["title"].lower() == recipe_name:
                        recipe = r
                        break
            if recipe:
                ingredients = "\n".join(recipe.get("ingredients", []))
                return jsonify({
                    "fulfillmentText": (
                        f"🍽️ {recipe['title']}\n"
                        f"🕒 {recipe['readyInMinutes']} min | Servings: {recipe['servings']}\n"
                        f"📋 Ingredients:\n{ingredients}\n"
                        f"🧑‍🍳 Instructions:\n{recipe['instructions']}\n"
                        f"🔗 Source: {recipe['sourceUrl']}"
                    )
                })
        except Exception as e:
            logging.error(f"Details error: {e}")
        return jsonify({"fulfillmentText": "Sorry, something went wrong getting the recipe."})

    elif intent == "RandomRecipeIntent":
        url = f"https://api.spoonacular.com/recipes/random?number=5&apiKey={SPOONACULAR_API_KEY}"
        res = requests.get(url)
        if res.status_code == 200:
            data = res.json()
            if "recipes" in data:
                RECIPE_CACHE = [{
                    "title": r.get("title", "Unknown"),
                    "sourceUrl": r.get("sourceUrl", "No URL"),
                    "readyInMinutes": r.get("readyInMinutes", "N/A"),
                    "servings": r.get("servings", "N/A"),
                    "ingredients": [i["original"] for i in r.get("extendedIngredients", [])],
                    "instructions": r.get("instructions", "Instructions not available.")
                } for r in data["recipes"]]
                titles = "\n".join([f"{i+1}. {r['title']}" for i, r in enumerate(RECIPE_CACHE)])
                return jsonify({"fulfillmentText": f"🍲 Here are some ideas:\n{titles}"})

        return jsonify({"fulfillmentText": "Sorry, couldn't fetch random recipes right now."})

    return jsonify({"fulfillmentText": "I'm not sure what you meant. Try again!"})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
