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

# Logging for debugging on Render
logging.basicConfig(level=logging.INFO)

app = Flask(__name__)

# API keys
CLARIFAI_API_KEY = "4a4ea9088cfa42c29e63f7b6806ad272"
SPOONACULAR_API_KEY = "b97364cb57314c0fb18b8d7e93d7e5fc"

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
        logging.error(f"Image resize error: {e}")
        return base64_str

def recognize_ingredients_from_base64(base64_image):
    base64_image = safely_resize_base64(base64_image)
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
    return [
        concept.name.lower()
        for concept in response.outputs[0].data.concepts
        if concept.value >= CONFIDENCE_THRESHOLD and concept.name.lower() not in UNWANTED_WORDS
    ]

@app.route("/webhook", methods=["POST"])
def webhook():
    global RECIPE_CACHE, TEMP_INGREDIENTS
    req = request.get_json()
    logging.info(f"Incoming request: {req}")
    intent = req["queryResult"]["intent"]["displayName"]
    parameters = req["queryResult"].get("parameters", {})

    if intent == "UploadImageIntent":
        base64_image = parameters.get("imageBase64")
        TEMP_INGREDIENTS = recognize_ingredients_from_base64(base64_image)
        if TEMP_INGREDIENTS:
            session = req.get("session", "")
            return jsonify({
                "fulfillmentText": f"I found these ingredients: {', '.join(TEMP_INGREDIENTS)}. Would you like to add or remove any?",
                "outputContexts": [
                    {
                        "name": f"{session}/contexts/ingredient-followup",
                        "lifespanCount": 5,
                        "parameters": {
                            "ingredients": TEMP_INGREDIENTS
                        }
                    }
                ]
            })
        else:
            return jsonify({"fulfillmentText": "I couldn't detect any ingredients from the image."})

    elif intent == "ConfirmIngredientsIntent":
        add_list = parameters.get("addList", "")
        remove_list = parameters.get("removeList", "")
        # Restore context if any
        for ctx in req["queryResult"].get("outputContexts", []):
            if "ingredient-followup" in ctx["name"]:
                TEMP_INGREDIENTS = ctx.get("parameters", {}).get("ingredients", TEMP_INGREDIENTS)

        if remove_list:
            for item in remove_list.lower().split(","):
                if item.strip() in TEMP_INGREDIENTS:
                    TEMP_INGREDIENTS.remove(item.strip())

        if add_list:
            for item in add_list.lower().split(","):
                item = item.strip()
                if item and item not in TEMP_INGREDIENTS:
                    TEMP_INGREDIENTS.append(item)

        if TEMP_INGREDIENTS:
            return jsonify({
                "fulfillmentText": f"Updated ingredients: {', '.join(TEMP_INGREDIENTS)}. Should I search for recipes now?"
            })
        else:
            return jsonify({"fulfillmentText": "No ingredients left. Please restart with a new image."})

    return jsonify({"fulfillmentText": "Unhandled intent."})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
