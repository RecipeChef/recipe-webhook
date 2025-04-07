from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route("/webhook", methods=["POST"])
def webhook():
    req = request.get_json()
    intent = req["queryResult"]["intent"]["displayName"]

    if intent == "GetRecipesIntent":
        ingredients = req["queryResult"]["parameters"].get("ingredients", [])
        reply = f"You sent me these ingredients: {', '.join(ingredients)}"
    else:
        reply = "I don't recognize that intent."

    return jsonify({"fulfillmentText": reply})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
