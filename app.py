from flask import Flask, request, jsonify
import requests
import base64
import logging
from clarifai_grpc.channel.clarifai_channel import ClarifaiChannel
from clarifai_grpc.grpc.api import service_pb2, service_pb2_grpc
from clarifai_grpc.grpc.api import resources_pb2
from clarifai_grpc.grpc.api.status import status_code_pb2

# Logging
logging.basicConfig(level=logging.INFO)

app = Flask(__name__)

# API keys
CLARIFAI_API_KEY = "4a4ea9088cfa42c29e63f7b6806ad272"
SPOONACULAR_API_KEY = "b97364cb57314c0fb18b8d7e93d7e5fc"

# Clarifai config
channel = ClarifaiChannel.get_grpc_channel()
stub = service_pb2_grpc.V2Stub(channel)
metadata = (("authorization", f"Key {CLARIFAI_API_KEY}"),)

UNWANTED_WORDS = {"pasture", "micronutrient", "aliment", "comestible"}
CONFIDENCE_THRESHOLD = 0.5

# Global variables
RECIPE_CACHE = []
TEMP_INGREDIENTS = []  # 🆕

# 📸 Recognize ingredients from image
def recognize_ingredients_from_base64(base64_image):
    base64_image = base64_image.strip().replace("\n", "").replace("\r", "")
    base64_image += "=" * ((4 - len(base64_image) % 4) % 4)
    image_bytes = base64.b64decode(base64_image)

    request = service_pb2.PostModelOutputsRequest(
        model_id="food-item-v1-recognition",
        inputs=[resources_pb2.Input(
            data=resources_pb2.Data(
                image=resources_pb2.Image(base64=image_bytes)
            )
        )]
    )
    response = stub.PostModelOutputs(request, metadata=metadata)

    if response.status.code != status_code_pb2.SUCCESS:
        return []

    return [
        concept.name.lower()
        for concept in response.outputs[0].data.concepts
        if concept.value >= CONFIDENCE_THRESHOLD and concept.name.lower() not in UNWANTED_WORDS
    ]

# 🍲 Fetch recipe details
def get_recipe_details(recipe_id):
    url = f"https://api.spoonacular.com/recipes/{recipe_id}/information?apiKey={SPOONACULAR_API_KEY}"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        return {
            "title": data.get("title"),
            "sourceUrl": data.get("sourceUrl", "No URL Available"),
            "ingredients": [i["original"] for i in data.get("extendedIngredients", [])],
            "instructions": data.get("instructions", "Instructions not available."),
            "readyInMinutes": data.get("readyInMinutes", "N/A"),
            "servings": data.get("servings", "N/A")
        }
    return None

# 📋 Get recipe list from ingredients
def get_recipes(ingredients):
    query = ",".join(ingredients)
    url = f"https://api.spoonacular.com/recipes/findByIngredients?ingredients={query}&number=5&apiKey={SPOONACULAR_API_KEY}"
    response = requests.get(url)
    if response.status_code != 200:
        return []
    return [get_recipe_details(r["id"]) for r in response.json() if r.get("id")]

# 🌐 Webhook route
@app.route("/webhook", methods=["POST"])
def webhook():
    global RECIPE_CACHE, TEMP_INGREDIENTS
    req = request.get_json()
    logging.info(f"Request: {req}")

    intent = req["queryResult"]["intent"]["displayName"]
    params = req["queryResult"].get("parameters", {})

    # 1️⃣ Image uploaded
    if intent == "UploadImageIntent":
        base64_image = params.get("imageBase64")
        TEMP_INGREDIENTS = recognize_ingredients_from_base64(base64_image)
        if TEMP_INGREDIENTS:
            return jsonify({
                "fulfillmentText": f"I found: {', '.join(TEMP_INGREDIENTS)}. Would you like to add or remove any?"
            })
        else:
            return jsonify({
                "fulfillmentText": "No ingredients detected. Please try another image."
            })

    # 2️⃣ Confirm ingredients
    elif intent == "ConfirmIngredientsIntent":
        add = params.get("addList", "")
        remove = params.get("removeList", "")

        if remove:
            for item in remove.lower().split(","):
                item = item.strip()
                if item in TEMP_INGREDIENTS:
                    TEMP_INGREDIENTS.remove(item)

        if add:
            for item in add.lower().split(","):
                item = item.strip()
                if item and item not in TEMP_INGREDIENTS:
                    TEMP_INGREDIENTS.append(item)

        if TEMP_INGREDIENTS:
            return jsonify({
                "fulfillmentText": f"Updated list: {', '.join(TEMP_INGREDIENTS)}. Should I get recipes now?"
            })
        else:
            return jsonify({
                "fulfillmentText": "All ingredients were removed. Please try again."
            })

    # 3️⃣ Get recipes
    elif intent == "GetRecipesIntent":
        raw = params.get("ingredients", [])
        ingredients = [i.strip() for i in raw.split(" and ")] if isinstance(raw, str) else raw
        if not ingredients:
            ingredients = TEMP_INGREDIENTS
        RECIPE_CACHE = get_recipes(ingredients)
        if RECIPE_CACHE:
            return jsonify({
                "fulfillmentText": "\n".join([
                    f"{i+1}. {r['title']} - {r['sourceUrl']}" for i, r in enumerate(RECIPE_CACHE)
                ])
            })
        return jsonify({"fulfillmentText": "Sorry, no recipes found."})

    # ❓ Fallback
    return jsonify({
        "fulfillmentText": "Sorry, I didn’t understand. Try uploading an image or asking for a recipe."
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
