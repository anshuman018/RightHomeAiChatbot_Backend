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
    """Route to handle OpenAI GPT responses with or without Kendra results."""
    data = request.json
    user_message = data.get("message")

    if not user_message:
        return jsonify({"error": "Message is required"}), 400

    try:
        # Query AWS Kendra for relevant property data
        kendra_response = kendra_client.query(
            IndexId=KENDRA_INDEX_ID,
            QueryText=user_message
        )

        # Extract and format Kendra search results
        search_results = [
            {
                "title": result.get("DocumentTitle", {}).get("Text", "No title available"),
                "excerpt": result.get("DocumentExcerpt", {}).get("Text", "No description available"),
                "url": result.get("DocumentURI", "No URL available")
            }
            for result in kendra_response.get("ResultItems", [])
        ]

        # Prepare context for OpenAI GPT
        context = """
        You are a helpful real estate chatbot working for RightHome AI, 
        an AI-based property broker system. Your goal is to assist users in 
        finding properties by providing accurate and real-time information. If 
        a user says something meaningless or irrelevant, politely redirect the 
        conversation back to meaningful topics, such as property searches or 
        real estate advice.

        Here are some property details from our database:
        """
        if search_results:
            for idx, result in enumerate(search_results, 1):
                context += f"\nMatch {idx}:\nTitle: {result['title']}\nExcerpt: {result['excerpt']}\n"
        else:
            context += """
            While we couldn't find exact matches, here are some general property suggestions:
            - Spacious 2BHK apartments in major cities.
            - Affordable housing near educational institutions.
            - Properties with amenities like swimming pools, gyms, and gardens.
            """

        # Generate response using OpenAI
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": context},
                {"role": "user", "content": user_message}
            ]
        )

        reply = response.choices[0].message['content']
        return jsonify({"reply": reply})

    except boto3.exceptions.Boto3Error as e:
        return jsonify({"error": f"AWS Kendra error: {str(e)}"}), 500
    except openai.error.OpenAIError as e:
        return jsonify({"error": f"OpenAI API error: {str(e)}"}), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 5000)))
