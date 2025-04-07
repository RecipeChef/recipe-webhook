
from flask import Flask, request, jsonify
import requests
import base64
import logging
from clarifai_grpc.channel.clarifai_channel import ClarifaiChannel
from clarifai_grpc.grpc.api import service_pb2, service_pb2_grpc
from clarifai_grpc.grpc.api import resources_pb2
from clarifai_grpc.grpc.api.status import status_code_pb2

# Logging for debugging on Render
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

# Recognize ingredients from base64 image using Clarifai
def recognize_ingredients_from_base64(base64_image):
    request = service_pb2.PostModelOutputsRequest(
        model_id="food-item-v1-recognition",
        inputs=[
            resources_pb2.Input(
                data=resources_pb2.Data(
                    image=resources_pb2.Image(base64=base64.b64decode(base64_image))
                )
            )
        ]
    )
    response = stub.PostModelOutputs(request, metadata=metadata)

    if response.status.code != status_code_pb2.SUCCESS:
        return []

    filtered_ingredients = []
    for concept in response.outputs[0].data.concepts:
        if concept.value >= CONFIDENCE_THRESHOLD and concept.name.lower() not in UNWANTED_WORDS:
            filtered_ingredients.append(concept.name.lower())

    return filtered_ingredients

# Get full recipe details from Spoonacular
def get_recipe_details(recipe_id):
    url = f"https://api.spoonacular.com/recipes/{recipe_id}/information?apiKey={SPOONACULAR_API_KEY}"
    response = requests.get(url)
    if response.status_code == 200:
        recipe_data = response.json()
        return {
            "title": recipe_data.get("title"),
            "sourceUrl": recipe_data.get("sourceUrl", "No URL Available")
        }
    return None

# Get recipes based on a list of ingredients
def get_recipes(ingredients):
    ingredients_query = ",".join(ingredients)
    url = f"https://api.spoonacular.com/recipes/findByIngredients?ingredients={ingredients_query}&number=5&apiKey={SPOONACULAR_API_KEY}"
    response = requests.get(url)
    if response.status_code != 200:
        return []
    recipes = response.json()
    return [get_recipe_details(recipe["id"]) for recipe in recipes if recipe.get("id")]

# Flask route for Dialogflow webhook
@app.route("/webhook", methods=["POST"])
def webhook():
    req = request.get_json()
    logging.info(f"Incoming request: {req}")

    intent = req["queryResult"]["intent"]["displayName"]
    parameters = req["queryResult"].get("parameters", {})

    if intent == "UploadImageIntent":
        base64_image = parameters.get("imageBase64")
        ingredients = recognize_ingredients_from_base64(base64_image)
        if ingredients:
            return jsonify({
                "fulfillmentText": f"I found these ingredients: {', '.join(ingredients)}. Want to see recipes?"
            })
        else:
            return jsonify({
                "fulfillmentText": "I couldn't detect any ingredients from the image. Please try another photo."
            })

    elif intent == "GetRecipesIntent":
        raw = parameters.get("ingredients", [])
        ingredients = [i.strip() for i in raw.split(" and ")] if isinstance(raw, str) else raw
        recipes = get_recipes(ingredients)
        if recipes:
            response_text = "\n".join([f"{r['title']} - {r['sourceUrl']}" for r in recipes])
        else:
            response_text = "Sorry, I couldn't find any recipes with those ingredients."
        return jsonify({"fulfillmentText": response_text})

    elif intent == "RandomRecipeIntent":
        url = f"https://api.spoonacular.com/recipes/random?number=1&apiKey={SPOONACULAR_API_KEY}"
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            if data.get("recipes"):
                recipe = data["recipes"][0]
                return jsonify({
                    "fulfillmentText": f"🍽️ {recipe['title']} - {recipe['sourceUrl']}"
                })
        return jsonify({"fulfillmentText": "Sorry, I couldn't fetch a random recipe right now."})

    return jsonify({
        "fulfillmentText": "I'm not sure what you meant. Try saying 'upload an image' or 'give me a recipe for chicken and rice'."
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
