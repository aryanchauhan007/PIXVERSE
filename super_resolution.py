import torch
import torch.nn as nn
from PIL import Image
import torchvision.transforms as transforms
import streamlit as st

# Define the simple super-resolution model
class SimpleSRModel(nn.Module):
    def __init__(self):
        super(SimpleSRModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=9, padding=4)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(64, 3, kernel_size=9, padding=4)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.conv2(x)
        return x


# Load the pre-trained model
device = torch.device('cpu')  # Change to 'cuda' if using a GPU
model = SimpleSRModel()
model.load_state_dict(torch.load("sr_model_final.pth", map_location=device), strict=False)
model = model.to(device)
model.eval()

# Define image transformations
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize to [-1, 1]
])

# Function to perform super-resolution
# The `super_resolve` function remains the same
def super_resolve(image):
    low_res_image = transform(image).unsqueeze(0).to(device)  # Now we pass a proper PIL image
    with torch.no_grad():
        high_res_image = model(low_res_image)
    high_res_image = high_res_image.squeeze(0).cpu().clamp(0, 1)  # Remove batch dimension and clamp to [0, 1]
    return transforms.ToPILImage()(high_res_image)  # Convert tensor to PIL image


# Streamlit app layout
st.title("Image Super Resolution")
st.write("Upload an image to increase its resolution.")

# File uploader for input images
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Read and display the uploaded image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Perform super-resolution and display the result
    high_res_image = super_resolve(image)
    st.image(high_res_image, caption='Super-Resolved Image', use_column_width=True)

# Download link for the super-resolved image
if uploaded_file is not None:
    high_res_image.save("super_resolved_image.png")
    st.download_button(label="Download Super-Resolved Image", data=open("super_resolved_image.png", "rb"), file_name="super_resolved_image.png")