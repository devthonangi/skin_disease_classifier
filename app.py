from flask import Flask, request, jsonify
import torch
import torchvision.transforms as transforms
from PIL import Image
import os
import logging

#  Initialize Flask App with Logging
logging.basicConfig(level=logging.DEBUG)
app = Flask(__name__)
app.logger.info(" Flask app is initializing...")

# Define CNN Model (Ensure it matches the trained model)
class SimpleCNN(torch.nn.Module):
    def __init__(self, num_classes=5):
        super(SimpleCNN, self).__init__()
        self.conv1 = torch.nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.relu = torch.nn.ReLU()
        self.pool = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = torch.nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.fc1 = torch.nn.Linear(64 * 32 * 32, 512)
        self.fc2 = torch.nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

#  Load Trained Model with Exception Handling
model_path = "cnn_skin_model.pth"

try:
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file '{model_path}' not found! Please download and place it in the project folder.")
    
    model = SimpleCNN(num_classes=5)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    app.logger.info(" Model loaded successfully!")

except Exception as e:
    app.logger.error(f" Error loading model: {e}")
    model = None  # Prevents crashes if model fails to load

#  Define Disease Classes
classes = ["Acne", "Eczema", "Psoriasis", "Rosacea", "Melanoma"]

#  Define Image Preprocessing
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

def predict_skin_disease(image_path):
    """Loads an image and predicts the skin disease."""
    try:
        image = Image.open(image_path).convert("RGB")
        image = transform(image).unsqueeze(0)

        with torch.no_grad():
            output = model(image)
            probabilities = torch.nn.functional.softmax(output[0], dim=0)

        predicted_class = torch.argmax(probabilities).item()
        confidence = probabilities[predicted_class].item() * 100
        return classes[predicted_class], confidence
    except Exception as e:
        app.logger.error(f" Error processing image: {e}")
        return None, None

#  Home Route
@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": " Flask API is running! Use the `/predict` endpoint to classify images."})

#  Flask Route for Prediction
@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    
    file = request.files["file"]
    file_path = "uploaded_image.jpg"
    file.save(file_path)

    prediction, confidence = predict_skin_disease(file_path)

    if prediction is None:
        return jsonify({"error": "Failed to process image"}), 500

    return jsonify({"disease": prediction, "confidence": f"{confidence:.2f}%"})

# Run Flask Server
if __name__ == "__main__":
    app.logger.info(" Flask is running on http://127.0.0.1:5000/")
    app.run(debug=True)
