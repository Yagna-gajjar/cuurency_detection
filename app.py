import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image

class CurrencyCNN(nn.Module):
    def __init__(self, num_classes):
        super(CurrencyCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.fc1 = nn.Linear(128 * 16 * 16, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, 128 * 16 * 16)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

num_classes = 7
class_names = ['1Hundrednote', '2Hundrednote', '2Thousandnote', '5Hundrednote', 'Fiftynote', 'Tennote', 'Twentynote']

model = torch.load('currency_cnn_model.pkl', map_location=device, weights_only=False)

model.eval()

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5],
                         std=[0.5, 0.5, 0.5])
])

def predict(image, model, transform):
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)
    
    return class_names[predicted.item()]

st.title("Currency Classification App")

st.write("Upload an image of a currency note, and the model will predict its denomination.")

uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    prediction = predict(image, model, transform)

    col1, col2 = st.columns(2)

    with col1:
        st.image(image, caption='Uploaded Image.', width=300)

    with col2:
        st.markdown("### 🧾 Prediction")
        st.markdown(
            f"<div style='padding:20px; border-radius:10px;border:2px solid white; text-align:center;'>"
            f"<span style='font-size:24px; font-weight:bold;'> {prediction} </span>"
            f"</div>",
            unsafe_allow_html=True
        )

