from flask import Flask, request, jsonify
from flask_cors import CORS
import openai
import boto3
import os

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load environment variables
KENDRA_INDEX_ID = os.getenv("KENDRA_INDEX_ID")
AWS_REGION = os.getenv("AWS_REGION")
ROLE_ARN = os.getenv("ROLE_ARN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Validate required environment variables
if not all([KENDRA_INDEX_ID, AWS_REGION, ROLE_ARN, OPENAI_API_KEY]):
    raise EnvironmentError("One or more required environment variables are missing!")

# Configure OpenAI API
openai.api_key = OPENAI_API_KEY

# Initialize AWS Kendra client
session = boto3.Session(
    aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
    region_name=AWS_REGION
)
kendra_client = session.client("kendra")


@app.route('/')
def home():
    """Root route to confirm the API is live."""
    return jsonify({"message": "Welcome to the RightHome AI Chatbot Backend!"})


@app.route('/chat', methods=['POST'])
def chat():
    """Route to handle OpenAI API calls."""
    data = request.json
    user_message = data.get("message")
    
    if not user_message:
        return jsonify({"error": "Message is required"}), 400

    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": user_message}
            ]
        )
        reply = response.choices[0].message['content']
        return jsonify({"reply": reply})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/kendra', methods=['POST'])
def kendra_search():
    """Route to query AWS Kendra."""
    data = request.json
    query = data.get("query")

    if not query:
        return jsonify({"error": "Query is required"}), 400

    try:
        response = kendra_client.query(
            IndexId=KENDRA_INDEX_ID,
            QueryText=query
        )
        results = [{"document_title": result["DocumentTitle"]["Text"],
                    "document_excerpt": result["DocumentExcerpt"]["Text"],
                    "document_url": result["DocumentURI"]}
                   for result in response.get("ResultItems", [])]

        return jsonify({"results": results})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 5000)))
