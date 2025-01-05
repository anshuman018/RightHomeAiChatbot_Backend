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

# In-memory buffer for conversation history (temporary solution)
conversation_memory = {}

@app.route('/')
def home():
    """Root route to confirm the API is live."""
    return jsonify({"message": "Welcome to the RightHome AI Chatbot Backend!"})


@app.route('/chat', methods=['POST'])
def chat():
    """Route to handle user messages and provide responses."""
    data = request.json
    user_message = data.get("message")
    session_id = data.get("session_id")  # Identify user session

    if not user_message or not session_id:
        return jsonify({"error": "Message and session_id are required"}), 400

    # Retrieve or initialize memory for the session
    if session_id not in conversation_memory:
        conversation_memory[session_id] = []

    try:
        # Query AWS Kendra for property data
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

        # Prepare dynamic context from memory and search results
        context = "You are a professional real estate broker working for RightHome AI. Assist users effectively.\n\n"
        
        # Add previous conversations to the context
        if conversation_memory[session_id]:
            context += "Here is the conversation so far:\n"
            for idx, entry in enumerate(conversation_memory[session_id], 1):
                context += f"{idx}. User: {entry['user']}\n   Chatbot: {entry['bot']}\n"

        # Add new context based on Kendra results
        if search_results:
            context += "\nHere are some properties that match the user's query:\n"
            for idx, result in enumerate(search_results, 1):
                context += f"Property {idx}: {result['title']}\nDetails: {result['excerpt']}\n"
        else:
            context += """
            While we don't have exact matches in the database, here are some excellent property options:
            - Spacious 2BHK apartments in city centers.
            - Luxurious villas in gated communities with amenities like swimming pools and gyms.
            - Affordable housing options near schools, colleges, and transport hubs.
            - Commercial spaces ideal for offices or retail stores in prime locations.
            """

        # Generate a response using OpenAI
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": context},
                {"role": "user", "content": user_message}
            ]
        )

        reply = response.choices[0].message['content']

        # Update memory with the latest exchange
        conversation_memory[session_id].append({"user": user_message, "bot": reply})

        # Limit memory to the last 10 exchanges per session (FIFO)
        if len(conversation_memory[session_id]) > 10:
            conversation_memory[session_id] = conversation_memory[session_id][-10:]

        return jsonify({"reply": reply})

    except boto3.exceptions.Boto3Error as e:
        return jsonify({"error": f"AWS Kendra error: {str(e)}"}), 500
    except openai.error.OpenAIError as e:
        return jsonify({"error": f"OpenAI API error: {str(e)}"}), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 5000)))
