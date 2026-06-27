from flask import Flask, render_template, request, jsonify
import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import requests
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")

# Load local disease database for offline/fallback detailed reports
DB_PATH = os.path.join(os.path.dirname(__file__), "disease_database.json")
try:
    with open(DB_PATH, "r", encoding="utf-8") as f:
        disease_db = json.load(f)
    print(f"Loaded local disease database with {len(disease_db)} detailed reports.")
except Exception as e:
    print("Error loading local disease database:", e)
    disease_db = {}

# -------------------------
# MODEL ARCHITECTURE DEFINITION
# -------------------------

class ChannelAttention(nn.Module):
    def __init__(self, in_channels: int, reduction: int = 16):
        super().__init__()
        mid = max(in_channels // reduction, 8)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_channels, mid, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid, in_channels, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg = self.fc(self.avg_pool(x))
        mx  = self.fc(self.max_pool(x))
        scale = self.sigmoid(avg + mx).unsqueeze(-1).unsqueeze(-1)
        return x * scale


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size: int = 7):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size,
                              padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_c = x.mean(dim=1, keepdim=True)
        max_c = x.max(dim=1, keepdim=True).values
        cat   = torch.cat([avg_c, max_c], dim=1)
        scale = self.sigmoid(self.conv(cat))
        return x * scale


class CBAM(nn.Module):
    def __init__(self, in_channels: int, reduction: int = 16, kernel_size: int = 7):
        super().__init__()
        self.channel  = ChannelAttention(in_channels, reduction)
        self.spatial  = SpatialAttention(kernel_size)

    def forward(self, x):
        x = self.channel(x)
        x = self.spatial(x)
        return x


class ResidualBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int,
                 stride: int = 1, use_cbam: bool = True):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride,
                               padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, stride=1,
                               padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(out_ch)
        self.relu  = nn.ReLU(inplace=True)
        self.drop  = nn.Dropout2d(p=0.05)
        self.cbam  = CBAM(out_ch) if use_cbam else nn.Identity()

        self.shortcut = nn.Sequential()
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.drop(out)
        out = self.bn2(self.conv2(out))
        out = out + self.shortcut(x)
        out = self.relu(out)
        out = self.cbam(out)
        return out


class PlantDiseaseNet(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )
        self.stage1 = nn.Sequential(
            ResidualBlock(64,  64,  stride=1, use_cbam=False),
            ResidualBlock(64,  64,  stride=1, use_cbam=False),
        )
        self.stage2 = nn.Sequential(
            ResidualBlock(64,  128, stride=2, use_cbam=False),
            ResidualBlock(128, 128, stride=1, use_cbam=False),
        )
        self.stage3 = nn.Sequential(
            ResidualBlock(128, 256, stride=2, use_cbam=True),
            ResidualBlock(256, 256, stride=1, use_cbam=True),
        )
        self.stage4 = nn.Sequential(
            ResidualBlock(256, 512, stride=2, use_cbam=True),
            ResidualBlock(512, 512, stride=1, use_cbam=True),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(
            nn.Dropout(p=0.4),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.head(x)
        return x

# -------------------------
# LOAD PYTORCH CHECKPOINT
# -------------------------
DEVICE = torch.device("cpu")
MODEL_PATH = os.path.join(os.path.dirname(__file__), "plant_disease_model_final.pth")

print(f"Loading checkpoint from: {MODEL_PATH}")
checkpoint = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
class_names = checkpoint["class_names"]
num_classes = checkpoint["num_classes"]

# Initialize model and load weights
model = PlantDiseaseNet(num_classes=num_classes)
model.load_state_dict(checkpoint["model_state_dict"])
model.to(DEVICE)
model.eval()

# Free checkpoint from memory to reduce RAM usage on free-tier hosting
del checkpoint
import gc
gc.collect()

print(f"Successfully loaded model with {num_classes} classes.")

# -------------------------
# IMAGE PREPROCESSING
# -------------------------
val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# -------------------------
# DYNAMIC CROP MAP CREATION
# -------------------------
crop_map = {}
for idx, class_name in enumerate(class_names):
    # Group names intelligently
    if "___" in class_name:
        crop = class_name.split("___")[0]
    elif "__" in class_name:
        crop = class_name.split("__")[0]
    elif class_name.startswith("Tomato"):
        crop = "Tomato"
    elif class_name.startswith("Potato"):
        crop = "Potato"
    elif class_name.startswith("Pepper"):
        crop = "Pepper"
    else:
        crop = class_name.split("_")[0]
    
    crop_clean = crop.replace("_", " ").strip().title()
    crop_map.setdefault(crop_clean, []).append(idx)

# -------------------------
# MISTRAL API FUNCTION
# -------------------------
def get_disease_info(disease_name):
    # Try calling the Mistral API first
    use_api = MISTRAL_API_KEY and "your_mistral_api_key" not in MISTRAL_API_KEY
    if use_api:
        try:
            print("calling Mistral API for:", disease_name)
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {MISTRAL_API_KEY}"
            }
            
            # Clean disease name for prompt
            clean_name = disease_name.replace("___", " ").replace("__", " ").replace("_", " ")
            
            payload = {
                "model": "open-mistral-7b",
                "messages": [
                    {
                        "role": "system",
                        "content": (
                            "You are a senior plant pathologist and agricultural scientist with 20+ years of field experience. "
                            "Your role is to provide deeply detailed, accurate, and farmer-friendly disease reports. "
                            "Write in simple, clear, and highly understandable language for a beginner or new farmer. "
                            "Avoid overly dense scientific jargon without explaining it first. "
                            "Always write full paragraphs with rich scientific detail AND practical guidance. "
                            "Use markdown formatting: ## for main headers, ### for sub-headers, **bold** for key terms, "
                            "and - for bullet lists. Never give vague or one-line answers."
                        )
                    },
                    {
                        "role": "user",
                        "content": (
                            f"Write a comprehensive, detailed, and easy-to-understand plant disease report on **{clean_name}**.\n\n"
                            f"Structure your response with these four sections:\n\n"
                            f"## Symptoms\n"
                            f"Describe in detail the visual symptoms on leaves, stems, fruits, and roots in simple terms. "
                            f"Explain what they look like, starting from early stages to late stages, so a new person can recognize them.\n\n"
                            f"## Causes\n"
                            f"Explain what pathogen causes it (fungus, bacteria, virus, or pest), how it spreads, and the environmental conditions that make it worse.\n\n"
                            f"## Prevention\n"
                            f"Provide clear, actionable steps to prevent the disease from starting (like crop rotation, keeping leaves dry, proper spacing, resistant seeds).\n\n"
                            f"## Treatment\n"
                            f"List both organic/natural remedies and standard chemical treatments, detailing how to apply them safely."
                        )
                    }
                ]
            }
            
            # Increase timeout to 35 seconds to ensure detailed generation doesn't cut off
            response = requests.post(
                "https://api.mistral.ai/v1/chat/completions",
                json=payload,
                headers=headers,
                timeout=35
            )
            response.raise_for_status()
            data = response.json()
            return data["choices"][0]["message"]["content"]
            
        except Exception as e:
            print("Mistral API Exception, falling back to local database:", e)
            
    # Fallback to local high-quality, pre-written database
    entry = disease_db.get(disease_name)
    if entry:
        return f"""
        {entry['Overview']}

        ## Symptoms
        {entry['Symptoms']}

        ## Causes
        {entry['Causes']}

        ## Prevention
        {entry['Prevention']}

        ## Treatment
        {entry['Treatment']}
        """
    else:
        return f"""
        ## Symptoms
        • Dark lesions, spots, or discoloration on leaf surface.
        • Browning, wilting, or yellowing leaves.

        ## Causes
        • Fungal, bacterial, or viral pathogen spreading under wet, humid weather.

        ## Prevention
        • Clean crop debris and rotate plant families.
        • Water at soil level and prune lower leaves.

        ## Treatment
        • Remove diseased sections.
        • Apply protective or organic copper fungicides.
        """

# -------------------------
# FLASK ROUTE DEFINITIONS
# -------------------------
app = Flask(__name__)
app.config["TEMPLATES_AUTO_RELOAD"] = True

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    confidence = None
    disease_info = None

    if request.method == "POST":
        selected_crop = request.form.get("crop")
        image_file = request.files.get("image")

        if selected_crop and image_file:
            # Read and process image
            image = Image.open(image_file).convert("RGB")
            processed_image = val_transform(image).unsqueeze(0).to(DEVICE)

            with torch.no_grad():
                logits = model(processed_image)
                probs = F.softmax(logits, dim=1)[0]

            # Limit argmax search to classes within the selected crop category
            valid_indices = crop_map[selected_crop]
            sub_probs = probs[valid_indices]
            best_sub_idx = torch.argmax(sub_probs).item()
            best_global_index = valid_indices[best_sub_idx]

            prediction = class_names[best_global_index]
            confidence = round(probs[best_global_index].item() * 100, 2)

            # Handle healthy classes
            if "healthy" in prediction.lower():
                disease_info = f"""
                ✅ Crop Health Status: OPTIMAL / HEALTHY

                No signs of active pathogens or foliage disease detected.

                Proactive Recommendations:
                • Continue structured irrigation schedules.
                • Monitor soil composition and balance nutrients.
                • Conduct routine leaf inspections.
                
                No treatment is required.
                """
            else:
                disease_info = get_disease_info(prediction)

    return render_template(
        "index.html",
        crops=sorted(crop_map.keys()),
        prediction=prediction,
        confidence=confidence,
        disease_info=disease_info
    )

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
