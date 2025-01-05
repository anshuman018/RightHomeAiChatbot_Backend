from flask import Flask, request, jsonify
from flask_cors import CORS
import boto3
import openai
import logging
import os

# === Logging Setup ===
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# === Load Environment Variables ===
KENDRA_INDEX_ID = os.getenv("KENDRA_INDEX_ID")
AWS_REGION = os.getenv("AWS_REGION")
ROLE_ARN = os.getenv("ROLE_ARN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not (KENDRA_INDEX_ID and AWS_REGION and ROLE_ARN and OPENAI_API_KEY):
    logger.error("One or more environment variables are missing. Please check the setup.")
    raise EnvironmentError("Environment variables not configured properly.")

# === Initialize AWS Kendra Client ===
kendra_client = boto3.client("kendra", region_name=AWS_REGION)

# === Initialize OpenAI API Key ===
openai.api_key = OPENAI_API_KEY

# === Flask App Initialization ===
app = Flask(__name__)
CORS(app)

# === AWS Kendra Search Function ===
def kendra_search(query):
    """
    Query AWS Kendra for search results.
    """
    try:
        response = kendra_client.query(
            IndexId=KENDRA_INDEX_ID,
            QueryText=query,
            PageSize=3
        )
        results = [
            item.get("DocumentExcerpt", {}).get("Text", "")
            for item in response.get("ResultItems", [])
        ]
        return results
    except Exception as e:
        logger.error(f"Error querying Kendra: {e}")
        return []

# === OpenAI GPT Response Function ===
def generate_combined_response(user_query, search_results):
    """
    Generate a response using OpenAI GPT with Kendra search results.
    """
    try:
        context = (
            "You are a helpful real estate chatbot working for RightHome AI. "
            "Your goal is to assist users in finding properties by providing accurate and real-time information.\n\n"
        )
        if search_results:
            context += "Here are some relevant property matches based on the query:\n"
            for idx, result in enumerate(search_results, 1):
                context += f"\nMatch {idx}:\n{result}\n"
        else:
            context += "No exact matches were found.\n"
        context += f"\nUser Query: {user_query}\n"

        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": context},
                {"role": "user", "content": user_query},
            ],
        )
        return response.choices[0].message["content"]
    except Exception as e:
        logger.error(f"Error generating GPT response: {e}")
        return "I'm sorry, I couldn't process your request at the moment."

# === Flask Routes ===
@app.route("/")
def home():
    return jsonify({"message": "Welcome to the Property Chatbot API!"})

@app.route("/property-query", methods=["POST"])
def property_query():
    """
    Handle property queries with AWS Kendra and GPT.
    """
    try:
        data = request.json
        user_query = data.get("userQuery")
        if not user_query:
            return jsonify({"error": "Missing 'userQuery' in request body"}), 400

        # Query AWS Kendra
        logger.info("Searching AWS Kendra for user query...")
        search_results = kendra_search(user_query)

        # Generate GPT Response
        logger.info("Generating GPT response...")
        response = generate_combined_response(user_query, search_results)

        return jsonify({
            "user_query": user_query,
            "search_results": search_results,
            "response": response
        }), 200

    except Exception as e:
        logger.error(f"Error handling property query: {e}")
        return jsonify({"error": "An internal server error occurred"}), 500

# === Main App Entry Point ===
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))  # Default to port 5000 for local testing
    app.run(host="0.0.0.0", port=port)
