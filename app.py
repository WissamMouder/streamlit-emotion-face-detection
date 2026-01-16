# app.py
import torch
from torchvision import models, transforms
from PIL import Image
from layout import render_layout

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Classes
classes = ["Fear", "Surprise", "Angry", "Sad", "Happy"]
num_classes = len(classes)

# Load Model
model = models.resnet50(weights=None)
model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
model.load_state_dict(torch.load("emotion_model.pth", map_location=device))
model.to(device)
model.eval()

# Transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])

# Prediction Function
def predict(image: Image.Image):
    input_tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(input_tensor)
        probs = torch.softmax(outputs, dim=1)[0]

    top_idx = torch.argmax(probs).item()
    return classes[top_idx], float(probs[top_idx])

# Run UI
render_layout(predict)
