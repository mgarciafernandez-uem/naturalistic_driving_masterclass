import os
import torch
from PIL import Image
import numpy as np
from torchvision import transforms
from torch.utils.data import TensorDataset, DataLoader, random_split
import pytorch_grad_cam
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
import shap


torch.manual_seed(626)

data_dir = "data"
images = []
filenames = []

# Read all images and their filenames
for fname in os.listdir(data_dir):
    if fname.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
        img_path = os.path.join(data_dir, fname)
        images.append(img_path)
        filenames.append(fname)

# Read steering angles from txt file
angles = {}
txt_file = os.path.join("./", "data.txt")
with open(txt_file, "r") as f:
    for line in f:
        parts = line.strip().split()
        file = parts[0]
        angle = parts[1].split(',')[0]

        angles[file] = float(angle)

# Example: print filename and corresponding steering angle
# Load images into a tensor
image_list = []
angle_list = []
for fname in filenames:

    if int(fname.split('.')[0]) > 10000:
        continue

    angle = angles.get(fname)

    img_path = os.path.join(data_dir, fname)
    img = Image.open(img_path).convert('RGB')
    img_array = np.array(img, dtype=np.float32) / 255.0  # normalize to [0,1]
    img_array = torch.tensor(img_array)  # convert to tensor
    img_array = img_array.permute(2, 0, 1)  # change to (C, H, W) format
    image_list.append(img_array)
    angle_list.append(angles[fname])


images_tensor = torch.stack(image_list)  # shape: (N, C, H, W)
angles_tensor = torch.tensor(angle_list, dtype=torch.float32)  # shape: (N,)

print("Images tensor shape:", images_tensor.shape)
print("Angles tensor shape:", angles_tensor.shape)


import torch.nn as nn

class PilotNet(nn.Module):
    def __init__(self):
        super(PilotNet, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=5, stride=2),   # fewer channels
            nn.ReLU(),
            nn.Conv2d(8, 12, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.Conv2d(12, 16, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.Conv2d(16, 24, kernel_size=3),
            nn.ReLU(),
            nn.Conv2d(24, 24, kernel_size=3),
            nn.ReLU()
        )
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(24 * 1 * 18, 20),   # fewer units
            nn.ReLU(),
            nn.Linear(20, 8),
            nn.ReLU(),
            nn.Linear(8, 1)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x

model = PilotNet()
print(model)


# Center crop images to fit the input size expected by PilotNet
# PilotNet expects input images that, after passing through conv layers, result in (64, 18, 18)
# Let's center crop all images to (66, 200) as in the original PilotNet paper

crop_transform = transforms.Compose([
    transforms.CenterCrop((66, 200)),
    transforms.ToTensor()
])

cropped_images = []
for fname in filenames:
    if int(fname.split('.')[0]) > 10000:
        continue
    img_path = os.path.join(data_dir, fname)
    img = Image.open(img_path).convert('RGB')
    img = crop_transform(img)
    cropped_images.append(img)

images_tensor = torch.stack(cropped_images)
print("Cropped images tensor shape:", images_tensor.shape)


# Split images and angles into train and test sets (e.g., 80% train, 20% test)
num_samples = images_tensor.shape[0]
indices = torch.randperm(num_samples)
train_size = int(0.8 * num_samples)
train_indices = indices[:train_size]
test_indices = indices[train_size:]

train_images = images_tensor[train_indices]
train_angles = angles_tensor[train_indices]
test_images = images_tensor[test_indices]
test_angles = angles_tensor[test_indices]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

images_tensor = images_tensor.to(device)
angles_tensor = angles_tensor.to(device)
train_images = train_images.to(device)
train_angles = train_angles.to(device)
test_images = test_images.to(device)
test_angles = test_angles.to(device)

model = model.to(device)

# Create DataLoaders for images and angles separately
batch_size = 32
train_loader = DataLoader(TensorDataset(train_images, train_angles), batch_size=batch_size, shuffle=True)
test_loader = DataLoader(TensorDataset(test_images, test_angles), batch_size=batch_size, shuffle=False)

print(f"Train images: {train_images.shape}, Train angles: {train_angles.shape}")
print(f"Test images: {test_images.shape}, Test angles: {test_angles.shape}")

import torch.optim as optim

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-5)

num_epochs = 1000
train_losses = []
test_losses = []

for epoch in range(num_epochs):
    model.train()
    running_train_loss = 0.0
    for images, angles in train_loader:
        images = images.to(device)
        angles = angles.to(device).unsqueeze(1)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, angles)
        loss.backward()
        optimizer.step()
        running_train_loss += loss.item() * images.size(0)
    avg_train_loss = running_train_loss / len(train_loader.dataset)
    train_losses.append(avg_train_loss)

    model.eval()
    running_test_loss = 0.0
    with torch.no_grad():
        for images, angles in test_loader:
            images = images.to(device)
            angles = angles.to(device).unsqueeze(1)
            outputs = model(images)
            loss = criterion(outputs, angles)
            running_test_loss += loss.item() * images.size(0)
    avg_test_loss = running_test_loss / len(test_loader.dataset)
    test_losses.append(avg_test_loss)

    print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {avg_train_loss:.4f} - Test Loss: {avg_test_loss:.4f}")


import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
fig, axs = plt.subplots(1, 2, figsize=(14, 6))

# Left: Y log axis
axs[0].plot(train_losses, label='Train Loss')
axs[0].plot(test_losses, label='Test Loss')
axs[0].set_xlabel('Epoch')
axs[0].set_ylabel('MSE Loss')
axs[0].set_title('Loss (Log Y)')
axs[0].set_yscale('log')
axs[0].legend()
axs[0].grid(True)

# Right: Y linear axis
axs[1].plot(train_losses, label='Train Loss')
axs[1].plot(test_losses, label='Test Loss')
axs[1].set_xlabel('Epoch')
axs[1].set_ylabel('MSE Loss')
axs[1].set_title('Loss (Linear Y)')
axs[1].legend()
axs[1].grid(True)
plt.savefig('loss_curve.png')


target_layer = model.conv_layers[0]  # First Conv2d layer

cam = GradCAM(model=model, target_layers=[target_layer])

# Visualize 10 test images: left = raw, right = GradCAM heatmap
num_images = min(10, test_images.shape[0])
fig, axs = plt.subplots(num_images, 2, figsize=(8, 4 * num_images))

for i in range(num_images):
    img_tensor = test_images[i].unsqueeze(0)
    img_np = img_tensor.squeeze().cpu().numpy().transpose(1, 2, 0)
    img_np = np.clip(img_np, 0, 1)

    grayscale_cam = cam(input_tensor=img_tensor)[0]
    heatmap_img = show_cam_on_image(img_np, grayscale_cam, use_rgb=True)

    # Left: raw image
    axs[i, 0].imshow(img_np)
    axs[i, 0].set_title(f"Raw Image {i+1}")
    axs[i, 0].axis('off')

    # Right: GradCAM heatmap
    axs[i, 1].imshow(heatmap_img)
    axs[i, 1].set_title(f"GradCAM {i+1}")
    axs[i, 1].axis('off')

plt.tight_layout()
plt.savefig("gradcam_side_by_side.png")


# Use a small subset of test images for SHAP (e.g., 10)
num_shap_images = min(10, test_images.shape[0])
shap_images = test_images[:num_shap_images]
shap_angles = test_angles[:num_shap_images]

# SHAP expects a function that outputs predictions
def predict_fn(x):
    x = torch.tensor(x, dtype=torch.float32).to(device)
    with torch.no_grad():
        return model(x).cpu().numpy()

# SHAP DeepExplainer requires the model and a background dataset
background = train_images[:100].cpu().numpy()  # use 100 images as background

explainer = shap.GradientExplainer(model, torch.tensor(background, dtype=torch.float32).to(device))
shap_values = explainer.shap_values(shap_images.to(device))

# Plot SHAP heatmaps side by side with raw images
fig, axs = plt.subplots(num_shap_images, 2, figsize=(8, 4 * num_shap_images))

for i in range(num_shap_images):
    img_np = shap_images[i].cpu().numpy().transpose(1, 2, 0)
    img_np = np.clip(img_np, 0, 1)

    # SHAP values for this image: shape (channels, height, width)
    shap_img = np.sum(shap_values[i], axis=0)  # sum over channels for visualization

    # Normalize SHAP heatmap to [0, 1]
    shap_img_norm = (shap_img - shap_img.min()) / (shap_img.max() - shap_img.min() + 1e-8)

    # Left: raw image
    axs[i, 0].imshow(img_np)
    axs[i, 0].set_title(f"Raw Image {i+1}")
    axs[i, 0].axis('off')

    # Right: SHAP heatmap overlay
    axs[i, 1].imshow(img_np)
    axs[i, 1].imshow(shap_img_norm, cmap='jet', alpha=0.5)
    axs[i, 1].set_title(f"SHAP {i+1}")
    axs[i, 1].axis('off')

plt.tight_layout()
plt.savefig("shap_side_by_side.png")