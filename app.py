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

# App setup
app = Flask(__name__)
logging.basicConfig(level=logging.INFO)

# === 🔐 Dialogflow Setup ===
DIALOGFLOW_PROJECT_ID = "recipechef-noml"  # <== CHANGE THIS!
DIALOGFLOW_CREDENTIALS = service_account.Credentials.from_service_account_file(
    "/etc/secrets/dialogflow_key.json"
)
dialogflow_session_client = dialogflow.SessionsClient(credentials=DIALOGFLOW_CREDENTIALS)

# === 🔐 Clarifai Setup ===
CLARIFAI_API_KEY = "4a4ea9088cfa42c29e63f7b6806ad272"  # <== CHANGE THIS!
clarifai_channel = ClarifaiChannel.get_grpc_channel()
clarifai_stub = service_pb2_grpc.V2Stub(clarifai_channel)
clarifai_metadata = (("authorization", f"Key {CLARIFAI_API_KEY}"),)

# === 🔐 Spoonacular Setup ===
SPOONACULAR_API_KEY = "your-spoonacular-api-key"  # <== CHANGE THIS!

@app.route('/analyze-image', methods=['POST'])
def analyze_image():
    try:
        UNWANTED_WORDS = {"aliment", "micronutrient", "pasture", "comestible"}
        CONFIDENCE_THRESHOLD = 0.3

        image_file = request.files['file']
        image = Image.open(image_file.stream)

        resized = image.resize((300, 300))
        logging.info(f"Image size after resize: {resized.size}")

        buffered = io.BytesIO()
        resized.save(buffered, format="JPEG")
        image_bytes = buffered.getvalue()
        image_base64 = base64.b64encode(image_bytes)
        logging.info(f"Base64 length: {len(image_base64)}")

        # Clarifai call
        request_clarifai = service_pb2.PostModelOutputsRequest(
            model_id="food-item-v1-recognition",
            inputs=[
                resources_pb2.Input(
                    data=resources_pb2.Data(
                        image=resources_pb2.Image(base64=image_base64)
                    )
                )
            ]
        )
        response = clarifai_stub.PostModelOutputs(request_clarifai, metadata=clarifai_metadata)

        logging.info("RAW Clarifai response:")
        logging.info(str(response))

        ingredients = []
        if response.status.code == status_code_pb2.SUCCESS:
            logging.info("Clarifai detected concepts:")
            for concept in response.outputs[0].data.concepts:
                logging.info(f"- {concept.name} ({concept.value:.2f})")
                if concept.value > CONFIDENCE_THRESHOLD and concept.name not in UNWANTED_WORDS:
                    ingredients.append(concept.name)

        return jsonify({"ingredients": ingredients})
    except Exception as e:
        logging.exception("Clarifai image analysis failed")
        return jsonify({"error": str(e)}), 500



# === Run Flask server ===
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
