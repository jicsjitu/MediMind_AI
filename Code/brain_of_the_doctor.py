# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Step 1: Setup GROQ API key
import os
import base64
from groq import Groq

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

# Step 2: Convert image to required format
def encode_image(image_path):
    """
    Encodes an image to Base64 format for API input.
    """
    if not image_path or not os.path.exists(image_path):
        return None  # Return None if no valid image is provided
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

# Step 3: Analyze Image with Query using Groq's Multimodal LLM
def analyze_image_with_query(query, encoded_image=None, model="meta-llama/llama-4-scout-17b-16e-instruct"):
    """
    Sends a text query and an optional image to Groq's AI model for analysis.

    Parameters:
    - query (str): The text prompt/question (should contain explanation context).
    - encoded_image (str or None): The Base64 encoded image (if available).
    - model (str): The AI model name (default: "meta-llama/llama-4-scout-17b-16e-instruct").

    Returns:
    - str: The AI-generated response.
    """
    client = Groq()

    # Construct message using provided query
    messages = [{
        "role": "user",
        "content": [
            {"type": "text", "text": query}
        ]
    }]

    # Append image if provided
    if encoded_image:
        messages[0]["content"].append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{encoded_image}"}
        })

    try:
        chat_completion = client.chat.completions.create(
            messages=messages,
            model=model
        )
        return chat_completion.choices[0].message.content
    except Exception as e:
        return f"Error in AI response: {str(e)}"

# --- UPDATE: Explanation-only prompt replaces classification prompt ---
explanation_prompt = """
You are a dermatologist AI. Your task is to classify the given skin image as *Acne, Eczema, Melanoma, or Unknown*.

🔹 *Rules for your response:*
1️⃣ Strictly return the classification using *only one* of these options:
   - Acne
   - Eczema
   - Melanoma
   - Unknown (if condition does not match any of the above)

2️⃣ Follow this exact response format:

---
Possible Skin Condition: (Acne, Eczema, Melanoma, or Unknown)

Reason: (Briefly explain the visible symptoms in the image.)

Medical Explanation: (Describe the causes and contributing factors of the condition.)

Causes: (Describe the causes)

Symptoms: (Comma-or-bullet list of common symptoms)

Recommendation: (Simple advice; do not mention any product names, or next steps.)
---"""

def construct_doctor_query(predicted_class):
    """
    Constructs the query for the AI based on the predicted class.
    """
    if predicted_class == "Unknown":
        return "This condition does not match any known skin diseases in my database. Please consult a dermatologist."
    else:
        return f"{explanation_prompt}\nCondition: {predicted_class}. Please provide a medical explanation and recommendation for this condition."

# Example usage:
# predicted_class = ... # Your classifier output here
# doctor_query = construct_doctor_query(predicted_class)
# encoded_image = encode_image(image_path)
# response = analyze_image_with_query(doctor_query, encoded_image)
