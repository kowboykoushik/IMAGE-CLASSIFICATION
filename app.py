# app.py
import streamlit as st
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import torchvision.transforms as transforms

class AdvancedCNN(nn.Module):
    def __init__(self, input_channels, num_classes):
        super(AdvancedCNN, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=input_channels, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.relu3 = nn.ReLU()

        self.flatten = nn.Flatten()

        self.fc1 = nn.Linear(128 * 7 * 7, 512)
        self.relu4 = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.pool2(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)

        x = self.flatten(x)

        x = self.fc1(x)
        x = self.relu4(x)
        x = self.dropout(x)

        x = self.fc2(x)

        return x

input_channels = 3
num_classes = 36
model = AdvancedCNN(input_channels, num_classes)
model.load_state_dict(torch.load('./senthil4_selvaku2_assignment2_part3.pth',map_location=torch.device('cpu')))
model.eval()

class_labels = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9','A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J','K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
'U', 'V', 'W', 'X', 'Y', 'Z']

# Streamlit app
def main():
    st.title("Image Classification Model Deployment")

    # User input for image upload
    uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_image is not None:
        # Display the uploaded image
        st.image(uploaded_image, caption="Uploaded Image.", use_column_width=True)

        # Preprocess the image for prediction
        image = Image.open(uploaded_image)
        transformed_image = preprocess_image(image)

        # Make predictions using your model
        with torch.no_grad():
            prediction = make_prediction(transformed_image, model)

        # Display the prediction
        st.success(f"Prediction: {class_labels[prediction]}")

def preprocess_image(image):
    if image.mode != 'RGB':
        image = image.convert('RGB')

    # Preprocess the image for prediction
    transform = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
    ])
    return transform(image).unsqueeze(0)

def make_prediction(image, model):
    # Make predictions using your model
    with torch.no_grad():
        prediction = model(image).argmax().item()
    return prediction

if __name__ == "__main__":
    main()
