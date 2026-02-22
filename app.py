# from flask import Flask, render_template, request
# import tensorflow as tf
# import numpy as np
# import json
# from PIL import Image
# import os
# import google.generativeai as genai

# # -------------------------
# # ENV SETTINGS
# # -------------------------
# os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# # -------------------------
# # GEMINI CONFIG (LLM)
# # -------------------------
# genai.configure(api_key="AIzaSyA6XV7FJJLZXJQy3rVBdii0MgFggpY_dPs")
# llm_model = genai.GenerativeModel("gemini-pro")

# def get_disease_info(disease_name):
#     try:
#         prompt = f"""
#         You are an agricultural expert.
#         Explain the plant disease "{disease_name}" in simple language.

#         Include:
#         1. What is this disease?
#         2. Common symptoms
#         3. Main causes
#         4. Prevention methods
#         5. Treatment or control methods

#         Keep it short and farmer-friendly.
#         """

#         response = llm_model.generate_content(prompt)
#         return response.text

#     except Exception as e:
#         # 🔴 IMPORTANT: fallback response
#         return f"""
#         Disease: {disease_name}

#         This disease affects plant health and reduces crop yield.

#         Recommended actions:
#         • Remove infected leaves
#         • Avoid overwatering
#         • Maintain proper spacing
#         • Use recommended fungicides
#         • Consult agricultural experts

#         (LLM service temporarily unavailable)
#         """

# # -------------------------
# # INITIALIZE FLASK APP
# # -------------------------
# app = Flask(__name__)
# app.config["TEMPLATES_AUTO_RELOAD"] = True

# # -------------------------
# # LOAD MODEL
# # -------------------------
# model = tf.keras.models.load_model("plant_disease_model.keras")

# # -------------------------
# # LOAD CLASS NAMES
# # -------------------------
# with open("class_names.json", "r") as f:
#     class_names = json.load(f)

# # -------------------------
# # CROP → CLASS INDEX MAP
# # -------------------------
# crop_map = {}
# for idx, class_name in enumerate(class_names):
#     crop = class_name.split("___")[0]
#     crop_map.setdefault(crop, []).append(idx)

# # -------------------------
# # IMAGE PREPROCESS
# # -------------------------
# def preprocess_image(image):
#     image = image.resize((224, 224))
#     image = np.expand_dims(np.array(image), axis=0)
#     image = tf.keras.applications.mobilenet_v2.preprocess_input(image)
#     return image

# # -------------------------
# # ROUTE
# # -------------------------
# @app.route("/", methods=["GET", "POST"])
# def index():
#     prediction = None
#     confidence = None
#     disease_info = None   # ✅ ALWAYS INIT

#     if request.method == "POST":
#         selected_crop = request.form["crop"]
#         image_file = request.files["image"]

#         image = Image.open(image_file).convert("RGB")
#         processed_image = preprocess_image(image)

#         preds = model.predict(processed_image)[0]

#         valid_indices = crop_map[selected_crop]
#         best_index = valid_indices[np.argmax(preds[valid_indices])]

#         prediction = class_names[best_index]
#         confidence = round(preds[best_index] * 100, 2)

#         # ✅ LLM CALL
#         disease_info = get_disease_info(prediction)

#     return render_template(
#         "index.html",
#         crops=crop_map.keys(),
#         prediction=prediction,
#         confidence=confidence,
#         disease_info=disease_info
#     )

# # -------------------------
# # RUN
# # -------------------------
# if __name__ == "__main__":
#     app.run(debug=False)


from flask import Flask, render_template, request
import tensorflow as tf
import numpy as np
import json
from PIL import Image
import os
import requests

# -------------------------
# ENV SETTINGS
# -------------------------
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# -------------------------
# INITIALIZE FLASK APP
# -------------------------
app = Flask(__name__)
app.config["TEMPLATES_AUTO_RELOAD"] = True

# -------------------------
# LOAD TRAINED MODEL
# -------------------------
model = tf.keras.models.load_model("plant_disease_model_f.keras")

# -------------------------
# LOAD CLASS NAMES
# -------------------------
with open("class_names_f.json", "r") as f:
    class_names = json.load(f)

# -------------------------
# CREATE CROP → CLASS INDEX MAP
# -------------------------
crop_map = {}
for idx, class_name in enumerate(class_names):
    crop = class_name.split("___")[0]
    crop_map.setdefault(crop, []).append(idx)

# -------------------------
# IMAGE PREPROCESSING
# -------------------------
def preprocess_image(image):
    # image = image.resize((224, 224))
    IMG_SIZE = (160, 160)
    image = image.resize(IMG_SIZE)
    image = np.expand_dims(np.array(image), axis=0)
    image = tf.keras.applications.mobilenet_v2.preprocess_input(image)
    return image

# -------------------------
# OLLAMA LLM FUNCTION
# -------------------------
def get_disease_info(disease_name):
    try:
        print("🧠 Calling Ollama for:", disease_name)

        payload = {
            "model": "tinyllama",
            "prompt": f"""
            Explain plant disease {disease_name}.
            Give:
            - Symptoms
            - Cause
            - Prevention
            - Treatment
            """,
            "stream": False
        }

        response = requests.post(
            "http://localhost:11434/api/generate",
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=300
        )

        response.raise_for_status()
        data = response.json()

        print("✅ Ollama response received")
        return data.get("response", "No response from local AI.")

    except Exception as e:
        print("❌ Ollama ERROR:", e)
        return f"""
        Disease: {disease_name}

        Basic advice:
        • Remove infected leaves
        • Avoid overwatering
        • Maintain proper spacing
        • Use recommended fungicides

        (Local AI service unavailable)
        """



# -------------------------
# FLASK ROUTE
# -------------------------
@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    confidence = None
    disease_info = None

    if request.method == "POST":
        selected_crop = request.form["crop"]
        image_file = request.files["image"]

        image = Image.open(image_file).convert("RGB")
        processed_image = preprocess_image(image)

        preds = model.predict(processed_image)[0]

        valid_indices = crop_map[selected_crop]
        best_index = valid_indices[np.argmax(preds[valid_indices])]

        prediction = class_names[best_index]
        confidence = round(preds[best_index] * 100, 2)

        # -------------------------
        # HANDLE HEALTHY CASE
        # -------------------------
        if "healthy" in prediction.lower():
            disease_info = f"""
            ✅ Plant Health Status: HEALTHY

            This crop does not show signs of any disease.

            Recommendations:
            • Continue regular irrigation
            • Maintain proper nutrition
            • Monitor leaves periodically
            • Follow good agricultural practices

            No treatment is required.
            """
        else:
            disease_info = get_disease_info(prediction)

    return render_template(
        "index.html",
        crops=crop_map.keys(),
        prediction=prediction,
        confidence=confidence,
        disease_info=disease_info
    )

# -------------------------
# RUN APPLICATION
# -------------------------
if __name__ == "__main__":
    app.run(debug=False)
