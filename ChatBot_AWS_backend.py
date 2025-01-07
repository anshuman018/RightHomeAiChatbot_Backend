from flask import Flask, request, jsonify
from flask_cors import CORS
import openai
import boto3
import os

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load environment variables for AWS and OpenAI
KENDRA_INDEX_ID = os.getenv("KENDRA_INDEX_ID")
AWS_REGION = os.getenv("AWS_REGION")
ROLE_ARN = os.getenv("ROLE_ARN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")

# Validate required environment variables
if not all([KENDRA_INDEX_ID, AWS_REGION, ROLE_ARN, OPENAI_API_KEY, AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY]):
    raise EnvironmentError("One or more required environment variables are missing!")

# Configure OpenAI API
openai.api_key = OPENAI_API_KEY

# Initialize AWS Kendra client with credentials
kendra_client = boto3.client(
    "kendra",
    region_name=AWS_REGION,
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY
)

# Memory Buffer for conversation history
conversation_history = []

# Chatbot Context
chatbot_context = """
You are a real estate chatbot working for RightHome AI. Your goal is to assist users in finding properties by providing accurate, real-time information in a simple, clear, and structured format.

Response Guidelines:
1. Provide property details using the following structure:
   - Property Name
   - Price
   - Location
   - Area
   - Features
   - Highlights
   - Image: [Image description](Image URL)

2. Format Specifications:
   - Avoid using special symbols or characters such as *, +, or ().
   - Maintain a plain text format for all outputs.

3. If no matching properties are found, acknowledge politely and suggest alternatives or ways to assist further.

4. Example of the desired format:
   Property 1: Green Valley Apartments
   Price: ₹50 Lakhs
   Location: Jabalpur City Center
   Area: 1500 sq ft
   Features: 3 BHK, Garden view, Parking space
   Highlights: Near schools, hospitals, and markets
   Image: [Green Valley Apartments](https://example.com/green-valley-image)

5. User Queries:
   - Respond accurately to property-related queries.
   - Redirect irrelevant or off-topic conversations back to property-related topics.

6. Image Links:
   - Ensure every property includes an image link where applicable. If no image is available, acknowledge this by stating: "No image available for this property."
"""

# AWS Kendra Search Function
def kendra_search(query):
    """Query AWS Kendra for search results."""
    try:
        response = kendra_client.query(
            IndexId=KENDRA_INDEX_ID,
            QueryText=query,
            PageSize=3
        )
        results = []
        for item in response.get("ResultItems", []):
            content = item.get("DocumentExcerpt", {}).get("Text", "")
            image_link = "No image available"
            document_attributes = item.get("DocumentAttributes", [])
            if document_attributes:
                image_link = document_attributes[0].get("Value", {}).get("TextWithLinksValue", "No image available")
            results.append({
                "content": content,
                "image_link": image_link
            })
        return results
    except Exception as e:
        print(f"Error querying Kendra: {e}")
        return []

# OpenAI Response Generator
def generate_combined_response(user_query, search_results):
    """Generate a response combining user query and Kendra search results."""
    try:
        # Prepare context from search results
        context = chatbot_context
        if search_results:
            context += "Here are some relevant property matches based on the query:\n"
            for idx, result in enumerate(search_results, 1):
                context += (
                    f"\nProperty {idx}: {result['content']}\n"
                    f"Image: {result['image_link']}\n"
                )
        else:
            context += "\nUnfortunately, no matching properties were found. Let me assist you further with tailored suggestions.\n"

        # Add conversation history to the context
        messages = [{"role": "system", "content": context}]
        messages.extend(conversation_history)
        messages.append({"role": "user", "content": user_query})

        # Generate response using OpenAI
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=messages
        )

        chatbot_response = response.choices[0].message["content"]

        # Update conversation history
        conversation_history.append({"role": "user", "content": user_query})
        conversation_history.append({"role": "assistant", "content": chatbot_response})

        return chatbot_response

    except Exception as e:
        print(f"Error generating response with GPT: {e}")
        return "I'm sorry, I couldn't process your request at the moment."

# Flask Routes
@app.route('/')
def home():
    return jsonify({"message": "Welcome to the RightHome AI Chatbot Backend!"})

@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    user_message = data.get("message")

    if not user_message:
        return jsonify({"error": "Message is required"}), 400

    try:
        search_results = kendra_search(user_message)
        reply = generate_combined_response(user_message, search_results)
        return jsonify({"reply": reply})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 5000)))
