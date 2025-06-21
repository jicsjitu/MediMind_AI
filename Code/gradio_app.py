# Load environment variables
from dotenv import load_dotenv  # For loading API keys and config from .env file
load_dotenv()

# Import necessary modules
import os  # For file operations and path handling
import gradio as gr  # Gradio for web UI
import tensorflow as tf  # TensorFlow for model loading and inference
from tensorflow import keras  # Keras API for model operations
import numpy as np  # Numerical operations on arrays
from PIL import Image  # Image processing
from deep_translator import GoogleTranslator  # For translating doctor responses
from fpdf import FPDF  # For generating PDF reports
import datetime  # For timestamping PDF filenames
import threading  # For delayed shutdown in exit tab
import matplotlib.pyplot as plt  # For plotting prediction bar charts

# Custom modules for AI processing
from brain_of_the_doctor import encode_image, analyze_image_with_query  # Functions for LLM-based insights
from voice_of_the_patient import transcribe_with_groq  # For speech-to-text conversion
from voice_of_the_doctor import text_to_speech_with_gtts  # For text-to-speech responses

# Load the trained models
def load_skin_disease_model():
    print("Loading Skin Disease Model...")
    model = keras.models.load_model("skin_disease_model.keras")
    print("Skin Disease Model loaded successfully!")
    return model

def load_resnet_model():
    print("Loading ResNet Model...")
    model = keras.models.load_model("ResNet_model.keras")
    print("ResNet Model loaded successfully!")
    return model

def load_densenet_model():
    print("Loading DenseNet121 Model...")
    model = keras.models.load_model("DenseNet121_skin_disease_model.keras")
    print("DenseNet121 Model loaded successfully!")
    return model

# Instantiate models at startup
skin_disease_model = load_skin_disease_model()
resnet_model = load_resnet_model()
densenet_model = load_densenet_model()

# Define class names for predictions
class_names = ["Acne", "Eczema", "Melanoma", "Unknown"]

# Mapping of supported languages to codes and font files
LANGUAGE_CODES = {
    "English": ("en", "NotoSans-Regular.ttf"),
    "Hindi": ("hi", "NotoSansDevanagari-Regular.ttf"),
    "Bengali": ("bn", "NotoSansBengali-Regular.ttf"),
    "Tamil": ("ta", "NotoSansTamil-Regular.ttf"),
    "Telugu": ("te", "NotoSansTelugu-Regular.ttf"),
    "Marathi": ("mr", "NotoSansDevanagari-Regular.ttf")
}

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
---
"""

# Image preprocessing function
def preprocess_image(image_path):
    print(f"Processing image: {image_path}")
    image = Image.open(image_path).convert("RGB").resize((224, 224))
    arr = np.array(image, dtype=np.float32) / 255.0
    image_tensor = np.expand_dims(arr, axis=0)
    print("Image preprocessed successfully!")
    return image_tensor

# Grad-CAM helper function
def get_gradcam_heatmap(model, img_array, last_conv_layer_name=None):
    if last_conv_layer_name is None:
        for layer in reversed(model.layers):
            if isinstance(layer, tf.keras.layers.Conv2D):
                last_conv_layer_name = layer.name
                break
        if last_conv_layer_name is None:
            raise ValueError("No convolutional layer found in the model.")
    print(f"Using last conv layer for Grad-CAM: {last_conv_layer_name}")
    grad_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[model.get_layer(last_conv_layer_name).output, model.output]
    )
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        loss = predictions[:, tf.argmax(predictions[0])]
    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy()

# Generate PDF report with Unicode support
def generate_pdf(doctor_response, skin_disease_prediction, language):
    lang_code, font_file = LANGUAGE_CODES.get(language, ("en", "NotoSans-Regular.ttf"))
    font_path = os.path.join(".", font_file)
    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)
    if not os.path.exists(font_path):
        print(f"Font file missing: {font_path}")
        return None
    pdf.add_font('LangFont', '', font_path, uni=True)
    pdf.set_font('LangFont', '', 16)
    pdf.cell(200, 10, 'MediMind AI - Diagnosis Report', ln=True, align='C')
    pdf.ln(10)
    pdf.set_font('LangFont', '', 12)
    pdf.cell(200, 10, 'Skin Disease Prediction:', ln=True)
    pdf.multi_cell(0, 10, skin_disease_prediction)
    pdf.ln(5)
    pdf.cell(200, 10, "Doctor's Response:", ln=True)
    pdf.multi_cell(0, 10, doctor_response)
    pdf.ln(5)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    pdf_output_path = f"skin_disease_report_{timestamp}.pdf"
    pdf.output(pdf_output_path)
    return pdf_output_path

# Main processing function for Gradio interface
def process_inputs(image_filepath, language, model_choice, prediction_mode):
    doctor_response = ""
    skin_disease_prediction = ""
    pdf_path = ""
    predicted_class = "Unknown"
    confidence = 0.0

    if image_filepath:
        try:
            img_tensor = preprocess_image(image_filepath)

            # Choose model based on user selection
            if prediction_mode == "Single Model Prediction":
                if model_choice == "MobileNetV3":
                    preds = skin_disease_model.predict(img_tensor)[0]
                    model = skin_disease_model
                elif model_choice == "ResNet Model":
                    preds = resnet_model.predict(img_tensor)[0]
                    model = resnet_model
                else:
                    preds = densenet_model.predict(img_tensor)[0]
                    model = densenet_model

                confidence = float(np.max(preds) * 100)
                predicted_class = class_names[int(np.argmax(preds))]
                if confidence < 60:
                    predicted_class = "Unknown"
                skin_disease_prediction = (
                    "I cannot confidently identify this condition. Please consult a dermatologist." 
                    if predicted_class == "Unknown" 
                    else f"Disease: {predicted_class} (Confidence: {confidence:.2f}%)"
                )
            else:
                results = []
                for name, mdl in [("MobileNetV3", skin_disease_model), ("ResNet Model", resnet_model), ("DenseNet Model", densenet_model)]:
                    p = mdl.predict(img_tensor)[0]
                    c = float(np.max(p) * 100)
                    cl = class_names[int(np.argmax(p))]
                    if c < 60:
                        cl = "Unknown"
                    results.append((name, cl, f"{c:.2f}%"))
                table = (
                    "| Model | Predicted Class | Confidence |\n"
                    "|-------|------------------|------------|\n"
                )
                for r in results:
                    table += f"| {r[0]} | {r[1]} | {r[2]} |\n"
                skin_disease_prediction = table
                model = skin_disease_model
                preds = model.predict(img_tensor)[0]
                predicted_class = class_names[int(np.argmax(preds))]
        except Exception as e:
            skin_disease_prediction = f"Error in classification: {str(e)}"

    # --- Grad-CAM and bar chart ---
    heatmap = get_gradcam_heatmap(model, img_tensor)
    heatmap_img = Image.fromarray((heatmap * 255).astype('uint8')).resize((224, 224)).convert('RGB')
    orig = Image.open(image_filepath).resize((224, 224))
    overlay = Image.blend(orig, heatmap_img, alpha=0.4)
    fig, ax = plt.subplots()
    ax.bar(class_names, preds * 100)
    ax.set_ylabel('Confidence (%)')
    fig.tight_layout()

    # --- UPDATE: Construct explanation-only query ---
    if predicted_class == "Unknown":
        doctor_query = "This condition does not match any known skin diseases in my database. Please consult a dermatologist."
    else:
        doctor_query = f"{explanation_prompt}\nCondition: {predicted_class}. Please provide a medical explanation and recommendation for this condition."

    # Get LLM explanation (no classification)
    doctor_response = analyze_image_with_query(
        query=doctor_query,
        encoded_image=encode_image(image_filepath),
        model="meta-llama/llama-4-scout-17b-16e-instruct"
    )

    # Translate doctor response if needed
    if language != "English":
        lang_codes = {"Hindi":"hi","Bengali":"bn","Telugu":"te","Marathi":"mr","Tamil":"ta"}
        target_lang = lang_codes.get(language, "en")
        doctor_response = GoogleTranslator(source='en', target=target_lang).translate(doctor_response)

    # Convert text response to speech
    voice_of_doctor = text_to_speech_with_gtts(input_text=doctor_response)

    # Generate downloadable PDF report
    pdf_path = generate_pdf(doctor_response, skin_disease_prediction, language)

    return (
        doctor_response,
        voice_of_doctor,
        skin_disease_prediction,
        pdf_path,
        fig,
        overlay,
        round(confidence, 2)
    )

# Define Gradio interfaces
main_app = gr.Interface(
    fn=process_inputs,
    inputs=[
        gr.Image(type="filepath", label="Upload Skin Image"),
        gr.Radio(list(LANGUAGE_CODES.keys()), value="English", label="Select Language"),
        gr.Radio(["MobileNetV3", "ResNet Model", "DenseNet Model"], value="MobileNetV3", label="Select Model"),
        gr.Radio(["Single Model Prediction", "Compare All Models"], value="Single Model Prediction", label="Prediction Mode")
    ],
    outputs=[
        gr.Textbox(label="MediMind AI Response"),
        gr.Audio(label="MediMind AI Voice"),
        gr.Textbox(label="MediMind AI Skin Disease Prediction"),
        gr.File(label="Download PDF Report"),
        gr.Plot(label="Prediction Probabilities"),
        gr.Image(label="Grad-CAM Heatmap Overlay"),
        gr.Slider(0, 100, step=0.1, label="Confidence Meter (%)")
    ],
    title="MediMind AI"
)

about_text = """
# About MediMind AI

**MediMind AI** is an intelligent dermatology assistant designed to help users identify potential skin conditions through advanced deep learning models.  
It combines powerful AI models, explainable Grad-CAM visualizations and multilingual support in an easy-to-use interface.

### Key Features:
- **Skin Disease Prediction:** Detects Acne, Eczema, Melanoma, or Unknown conditions using MobileNetV3, ResNet, and DenseNet121 models.
- **Explainable AI:** Grad-CAM heatmaps to visually explain model predictions.
- **AI Doctor Consultation:** Generates AI-based diagnosis explanations and treatment recommendations.
- **Multilingual Support:** Provides results in English, Hindi, Bengali, Tamil, Telugu, and Marathi.
- **Voice Interaction:** Text-to-speech responses to enhance user accessibility.
- **Downloadable Report:** PDF diagnosis reports with detailed findings and advice.

---
**Developed with ❤️ using AI and Deep Learning.**
"""
about_app = gr.Interface(
    fn=lambda: about_text,
    inputs=[],
    outputs=[gr.Markdown()],
    title="About MediMind AI"
)

# Help Tab
help_text = """
# How to Use MediMind AI

1. Upload a clear skin image...
2. Select language...
3. Choose model and mode...
4. Click Submit and view results.
"""
help_app = gr.Interface(fn=lambda: help_text, inputs=[], outputs=[gr.Markdown()], title="Help")

# Feedback Tab
import pandas as pd
import re

# Directory & file setup
feedback_dir = "feedbacks"
feedback_file = os.path.join(feedback_dir, "feedback.csv")
os.makedirs(feedback_dir, exist_ok=True)

# Basic spam/bad word filtering
BAD_WORDS = ["spam", "abuse", "hack", "fraud", "scam", "attack", "hate"]

def is_spam(message):
    message_lower = message.lower()
    for word in BAD_WORDS:
        if word in message_lower:
            return True
    if len(message.strip()) < 10:  # too short
        return True
    return False

# Main feedback submit function
def submit_feedback(name, email, message):
    if is_spam(message):
        return "⚠️ Your feedback seems suspicious or too short. Please provide valid feedback."

    ts = datetime.datetime.now().isoformat()

    # Create or append feedback
    feedback_entry = {
        "timestamp": ts,
        "name": name if name else "Anonymous",
        "email": email if email else "Not Provided",
        "message": message.strip()
    }

    try:
        if not os.path.exists(feedback_file):
            # Create new file
            df = pd.DataFrame([feedback_entry])
        else:
            # Append to existing
            df_existing = pd.read_csv(feedback_file)
            df = pd.concat([df_existing, pd.DataFrame([feedback_entry])], ignore_index=True)
        
        df.to_csv(feedback_file, index=False)
        return "✅ Thank you for your feedback! We appreciate your time."
    except Exception as e:
        return f"❌ Error saving feedback: {str(e)}"

# Gradio Interface
feedback_app = gr.Interface(
    fn=submit_feedback,
    inputs=[
        gr.Textbox(label="Name (optional)"),
        gr.Textbox(label="Email (optional)"),
        gr.Textbox(label="Your Feedback", lines=5, placeholder="Write your feedback in detail...")
    ],
    outputs=gr.Textbox(label="Response"),
    title="📝 Submit Your Feedback",
    theme="default",
    css="body {background-color: #f7f7f7;}"
)

# Exit Tab (Client-side window close)
def exit_app_fn():
    # Delay shutdown slightly to allow response
    threading.Timer(0.5, lambda: os._exit(0)).start()
    return "Exiting application..."

exit_app = gr.Interface(
    fn=exit_app_fn,
    inputs=[],
    outputs=[gr.Textbox(label="Status")],
    title="Exit"
)

# Combine in Tabs
app = gr.TabbedInterface(
    [main_app, about_app, help_app, feedback_app, exit_app],
    ["MediMind AI","About","Help","Feedback","Exit"]
)

if __name__ == "__main__":
    app.launch(debug=True)  # Launch the Gradio app in debug mode
