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

# Global recipe cache
RECIPE_CACHE = []

def recognize_ingredients_from_base64(base64_image):
    base64_image = base64_image.strip().replace("\n", "").replace("\r", "")
    base64_image += "=" * ((4 - len(base64_image) % 4) % 4)
    image_bytes = base64.b64decode(base64_image)

    request = service_pb2.PostModelOutputsRequest(
        model_id="food-item-v1-recognition",
        inputs=[
            resources_pb2.Input(
                data=resources_pb2.Data(
                    image=resources_pb2.Image(base64=image_bytes)
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

def get_recipe_details(recipe_id):
    url = f"https://api.spoonacular.com/recipes/{recipe_id}/information?apiKey={SPOONACULAR_API_KEY}"
    response = requests.get(url)
    if response.status_code == 200:
        recipe_data = response.json()
        return {
            "title": recipe_data.get("title"),
            "sourceUrl": recipe_data.get("sourceUrl", "No URL Available"),
            "ingredients": [ing["original"] for ing in recipe_data.get("extendedIngredients", [])],
            "instructions": recipe_data.get("instructions", "Instructions not available."),
            "readyInMinutes": recipe_data.get("readyInMinutes", "N/A"),
            "servings": recipe_data.get("servings", "N/A")
        }
    return None

def get_recipes(ingredients):
    ingredients_query = ",".join(ingredients)
    url = f"https://api.spoonacular.com/recipes/findByIngredients?ingredients={ingredients_query}&number=5&apiKey={SPOONACULAR_API_KEY}"
    response = requests.get(url)
    if response.status_code != 200:
        return []
    recipes = response.json()
    return [get_recipe_details(recipe["id"]) for recipe in recipes if recipe.get("id")]

@app.route("/webhook", methods=["POST"])
def webhook():
    global RECIPE_CACHE
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
        RECIPE_CACHE = get_recipes(ingredients)
        if RECIPE_CACHE:
            response_text = "\n".join([f"{idx + 1}. {r['title']} - {r['sourceUrl']}" for idx, r in enumerate(RECIPE_CACHE)])
        else:
            response_text = "Sorry, I couldn't find any recipes with those ingredients."
        return jsonify({"fulfillmentText": response_text})

    elif intent == "ShowRecipeDetailsIntent":
        try:
            recipe_number = int(parameters.get("recipeNumber"))
            if 1 <= recipe_number <= len(RECIPE_CACHE):
                recipe = RECIPE_CACHE[recipe_number - 1]
                ingredients = "\n".join(recipe.get("ingredients", []))
                instructions = recipe.get("instructions", "Instructions not available.")
                return jsonify({
                    "fulfillmentText": (
                        f"🍽️ {recipe['title']}\n"
                        f"🕒 Ready in: {recipe['readyInMinutes']} minutes | Servings: {recipe['servings']}\n"
                        f"📋 Ingredients:\n{ingredients}\n"
                        f"🧑‍🍳 Instructions:\n{instructions}\n"
                        f"🔗 Source: {recipe['sourceUrl']}"
                    )
                })
            else:
                return jsonify({
                    "fulfillmentText": "Invalid recipe number. Please pick one from the list."
                })
        except Exception as e:
            logging.error(f"Error in ShowRecipeDetailsIntent: {e}")
            return jsonify({
                "fulfillmentText": "Something went wrong trying to get that recipe's details."
            })

    elif intent == "RandomRecipeIntent":
        url = f"https://api.spoonacular.com/recipes/random?number=5&apiKey={SPOONACULAR_API_KEY}"
        response = requests.get(url)

        if response.status_code == 200:
            data = response.json()
            if data.get("recipes"):
                RECIPE_CACHE = []  # ✅ OK now because it's declared global at the top
    
                for recipe in data["recipes"]:
                    RECIPE_CACHE.append({
                        "title": recipe.get("title", "Unknown"),
                        "sourceUrl": recipe.get("sourceUrl", "No URL"),
                        "readyInMinutes": recipe.get("readyInMinutes", "N/A"),
                        "servings": recipe.get("servings", "N/A"),
                        "ingredients": [ing["original"] for ing in recipe.get("extendedIngredients", [])],
                        "instructions": recipe.get("instructions", "Instructions not available.")
                    })
    
                text = "\n".join([f"{idx + 1}. {r['title']}" for idx, r in enumerate(RECIPE_CACHE)])
                return jsonify({
                    "fulfillmentText": f"🍽️ Here are 5 random recipes:\n{text}\n\nSay something like 'Show me recipe 2' for details."
                })

        return jsonify({"fulfillmentText": "Sorry, I couldn't fetch random recipes right now."})

    return jsonify({
        "fulfillmentText": "I'm not sure what you meant. Try saying 'upload an image' or 'give me a recipe for chicken and rice'."
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
