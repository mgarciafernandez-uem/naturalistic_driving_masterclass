import os
import torch
from PIL import Image
import numpy as np
from torchvision import transforms
from torch.utils.data import TensorDataset, DataLoader
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
import numpy as np

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

# Load and process images
crop_transform = transforms.Compose([
    transforms.CenterCrop((66, 200)),
    transforms.ToTensor()
])

cropped_images = []
angle_list = []
for fname in filenames:
    if int(fname.split('.')[0]) > 10000:
        continue
    
    angle = angles.get(fname)
    if angle is None:
        continue
        
    img_path = os.path.join(data_dir, fname)
    img = Image.open(img_path).convert('RGB')
    img = crop_transform(img)
    cropped_images.append(img)
    angle_list.append(angle)

images_tensor = torch.stack(cropped_images)
angles_tensor = torch.tensor(angle_list, dtype=torch.float32)

print("Images tensor shape:", images_tensor.shape)
print("Angles tensor shape:", angles_tensor.shape)

# Define PilotNet model
class PilotNet(nn.Module):
    def __init__(self):
        super(PilotNet, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=5, stride=2),
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
            nn.Linear(24 * 1 * 18, 20),
            nn.ReLU(),
            nn.Linear(20, 8),
            nn.ReLU(),
            nn.Linear(8, 1)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
images_tensor = images_tensor.to(device)
angles_tensor = angles_tensor.to(device)

# Split data into 3 non-overlapping subsets
num_samples = images_tensor.shape[0]
indices = torch.randperm(num_samples)

# Split into 3 equal parts
subset1_size = num_samples // 3
subset2_size = num_samples // 3
subset3_size = num_samples - subset1_size - subset2_size

subset1_indices = indices[:subset1_size]
subset2_indices = indices[subset1_size:subset1_size + subset2_size]
subset3_indices = indices[subset1_size + subset2_size:]

# Create subset datasets
subset1_images = images_tensor[subset1_indices]
subset1_angles = angles_tensor[subset1_indices]

subset2_images = images_tensor[subset2_indices]
subset2_angles = angles_tensor[subset2_indices]

subset3_images = images_tensor[subset3_indices]
subset3_angles = angles_tensor[subset3_indices]

print(f"Subset 1: {subset1_images.shape[0]} samples")
print(f"Subset 2: {subset2_images.shape[0]} samples")
print(f"Subset 3: {subset3_images.shape[0]} samples")

# Create test set from a portion of each subset
test_size_per_subset = min(50, subset1_size // 5)  # 20% or max 50 samples per subset
test_images = torch.cat([
    subset1_images[:test_size_per_subset],
    subset2_images[:test_size_per_subset],
    subset3_images[:test_size_per_subset]
])
test_angles = torch.cat([
    subset1_angles[:test_size_per_subset],
    subset2_angles[:test_size_per_subset],
    subset3_angles[:test_size_per_subset]
])

# Remove test samples from training subsets
subset1_images = subset1_images[test_size_per_subset:]
subset1_angles = subset1_angles[test_size_per_subset:]
subset2_images = subset2_images[test_size_per_subset:]
subset2_angles = subset2_angles[test_size_per_subset:]
subset3_images = subset3_images[test_size_per_subset:]
subset3_angles = subset3_angles[test_size_per_subset:]

# Create data loaders
batch_size = 32
subset1_loader = DataLoader(TensorDataset(subset1_images, subset1_angles), batch_size=batch_size, shuffle=True)
subset2_loader = DataLoader(TensorDataset(subset2_images, subset2_angles), batch_size=batch_size, shuffle=True)
subset3_loader = DataLoader(TensorDataset(subset3_images, subset3_angles), batch_size=batch_size, shuffle=True)
test_loader = DataLoader(TensorDataset(test_images, test_angles), batch_size=batch_size, shuffle=False)

def train_model(model, train_loader, num_epochs=2500, lr=1e-5):
    """Train a model and return training history"""
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    model.train()
    losses = []
    
    for epoch in range(num_epochs):
        running_loss = 0.0
        for images, angles in train_loader:
            images = images.to(device)
            angles = angles.to(device).unsqueeze(1)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, angles)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * images.size(0)
        
        avg_loss = running_loss / len(train_loader.dataset)
        losses.append(avg_loss)
        
        if (epoch + 1) % 100 == 0:
            print(f"Epoch {epoch+1}/{num_epochs} - Loss: {avg_loss:.4f}")
    
    return losses

def federated_averaging(models):
    """Average weights of multiple models"""
    avg_state_dict = {}
    
    # Get all parameter keys from the first model
    for key in models[0].state_dict().keys():
        # Average the parameters across all models
        avg_state_dict[key] = torch.stack([model.state_dict()[key] for model in models]).mean(0)
    
    return avg_state_dict

def train_federated(models, train_loaders, num_epochs=2500, lr=1e-5):
    """Train models using federated learning with per-epoch averaging"""
    criterion = nn.MSELoss()
    optimizers = [optim.Adam(model.parameters(), lr=lr) for model in models]
    
    federated_losses = []
    
    for epoch in range(num_epochs):
        # Train each model on its local data for one epoch
        epoch_losses = []
        
        for i, (model, train_loader, optimizer) in enumerate(zip(models, train_loaders, optimizers)):
            model.train()
            running_loss = 0.0
            
            for images, angles in train_loader:
                images = images.to(device)
                angles = angles.to(device).unsqueeze(1)
                
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, angles)
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item() * images.size(0)
            
            avg_loss = running_loss / len(train_loader.dataset)
            epoch_losses.append(avg_loss)
        
        # Average the weights across all models
        avg_state_dict = federated_averaging(models)
        
        # Update all models with the averaged weights
        for model in models:
            model.load_state_dict(avg_state_dict)
        
        # Record the average loss across all clients
        federated_losses.append(np.mean(epoch_losses))
        
        if (epoch + 1) % 5 == 0:
            print(f"Federated Epoch {epoch+1}/{num_epochs} - Avg Loss: {np.mean(epoch_losses):.4f}")
    
    return federated_losses

def evaluate_model(model, test_loader):
    """Evaluate model on test set"""
    model.eval()
    criterion = nn.MSELoss()
    total_loss = 0.0
    
    with torch.no_grad():
        for images, angles in test_loader:
            images = images.to(device)
            angles = angles.to(device).unsqueeze(1)
            outputs = model(images)
            loss = criterion(outputs, angles)
            total_loss += loss.item() * images.size(0)
    
    return total_loss / len(test_loader.dataset)

# Train individual models
print("Training individual models...")

model1 = PilotNet().to(device)
model2 = PilotNet().to(device)
model3 = PilotNet().to(device)

print("\nTraining Model 1 (Subset 1):")
losses1 = train_model(model1, subset1_loader)

print("\nTraining Model 2 (Subset 2):")
losses2 = train_model(model2, subset2_loader)

print("\nTraining Model 3 (Subset 3):")
losses3 = train_model(model3, subset3_loader)

# Evaluate individual models
test_loss1 = evaluate_model(model1, test_loader)
test_loss2 = evaluate_model(model2, test_loader)
test_loss3 = evaluate_model(model3, test_loader)

print(f"\nIndividual Model Test Losses:")
print(f"Model 1: {test_loss1:.4f}")
print(f"Model 2: {test_loss2:.4f}")
print(f"Model 3: {test_loss3:.4f}")

# Federated Learning: Per-epoch averaging
print("\nTraining federated models with per-epoch averaging...")

federated_model1 = PilotNet().to(device)
federated_model2 = PilotNet().to(device)
federated_model3 = PilotNet().to(device)

federated_models = [federated_model1, federated_model2, federated_model3]
federated_loaders = [subset1_loader, subset2_loader, subset3_loader]

federated_losses = train_federated(federated_models, federated_loaders)

# Evaluate federated model (all models should have identical weights after federated training)
federated_test_loss = evaluate_model(federated_model1, test_loader)
print(f"Federated Model Test Loss: {federated_test_loss:.4f}")

# Train a centralized model for comparison
print("\nTraining centralized model for comparison...")
centralized_model = PilotNet().to(device)

# Combine all training data
all_train_images = torch.cat([subset1_images, subset2_images, subset3_images])
all_train_angles = torch.cat([subset1_angles, subset2_angles, subset3_angles])
centralized_loader = DataLoader(TensorDataset(all_train_images, all_train_angles), batch_size=batch_size, shuffle=True)

centralized_losses = train_model(centralized_model, centralized_loader)
centralized_test_loss = evaluate_model(centralized_model, test_loader)
print(f"Centralized Model Test Loss: {centralized_test_loss:.4f}")

# Plot comparison
plt.figure(figsize=(15, 10))

# Training losses
plt.subplot(2, 2, 1)
plt.plot(losses1, label='Model 1', alpha=0.7)
plt.plot(losses2, label='Model 2', alpha=0.7)
plt.plot(losses3, label='Model 3', alpha=0.7)
plt.plot(federated_losses, label='Federated', alpha=0.7, linewidth=2)
plt.plot(centralized_losses, label='Centralized', alpha=0.7)
plt.xlabel('Epoch')
plt.ylabel('Training Loss')
plt.title('Training Losses Comparison')
plt.legend()
plt.yscale('log')
plt.grid(True)

# Test loss comparison
plt.subplot(2, 2, 2)
models = ['Model 1', 'Model 2', 'Model 3', 'Federated', 'Centralized']
test_losses = [test_loss1, test_loss2, test_loss3, federated_test_loss, centralized_test_loss]
colors = ['blue', 'orange', 'green', 'red', 'purple']
bars = plt.bar(models, test_losses, color=colors, alpha=0.7)
plt.ylabel('Test Loss')
plt.title('Test Loss Comparison')
plt.xticks(rotation=45)
plt.grid(True, alpha=0.3)

# Add value labels on bars
for bar, loss in zip(bars, test_losses):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001, 
             f'{loss:.4f}', ha='center', va='bottom')

# Visualization with GradCAM for federated model
plt.subplot(2, 1, 2)
target_layer = federated_model1.conv_layers[0]
cam = GradCAM(model=federated_model1, target_layers=[target_layer])

# Show 5 test images with GradCAM
num_viz = min(5, test_images.shape[0])
fig_cam, axs_cam = plt.subplots(2, num_viz, figsize=(15, 6))

for i in range(num_viz):
    img_tensor = test_images[i].unsqueeze(0)
    img_np = img_tensor.squeeze().cpu().numpy().transpose(1, 2, 0)
    img_np = np.clip(img_np, 0, 1)
    
    grayscale_cam = cam(input_tensor=img_tensor)[0]
    heatmap_img = show_cam_on_image(img_np, grayscale_cam, use_rgb=True)
    
    # Original image
    axs_cam[0, i].imshow(img_np)
    axs_cam[0, i].set_title(f"Original {i+1}")
    axs_cam[0, i].axis('off')
    
    # GradCAM
    axs_cam[1, i].imshow(heatmap_img)
    axs_cam[1, i].set_title(f"GradCAM {i+1}")
    axs_cam[1, i].axis('off')

plt.tight_layout()
plt.savefig("federated_learning_comparison.png", dpi=300, bbox_inches='tight')

# Print summary
print("\n" + "="*50)
print("FEDERATED LEARNING SUMMARY")
print("="*50)
print(f"Individual Model Test Losses:")
print(f"  Model 1 (Subset 1): {test_loss1:.4f}")
print(f"  Model 2 (Subset 2): {test_loss2:.4f}")
print(f"  Model 3 (Subset 3): {test_loss3:.4f}")
print(f"  Average Individual: {(test_loss1 + test_loss2 + test_loss3) / 3:.4f}")
print(f"\nFederated Model Test Loss: {federated_test_loss:.4f}")
print(f"Centralized Model Test Loss: {centralized_test_loss:.4f}")
print(f"\nFederated vs Centralized: {((federated_test_loss - centralized_test_loss) / centralized_test_loss * 100):+.2f}%")

# Create a comprehensive loss plot
plt.figure(figsize=(12, 8))

# Plot training losses for all scenarios
plt.plot(range(1, len(losses1) + 1), losses1, label='Model 1 (Subset 1)', linewidth=2, alpha=0.8)
plt.plot(range(1, len(losses2) + 1), losses2, label='Model 2 (Subset 2)', linewidth=2, alpha=0.8)
plt.plot(range(1, len(losses3) + 1), losses3, label='Model 3 (Subset 3)', linewidth=2, alpha=0.8)
plt.plot(range(1, len(federated_losses) + 1), federated_losses, label='Federated Model', linewidth=3, alpha=0.9)
plt.plot(range(1, len(centralized_losses) + 1), centralized_losses, label='Centralized Model', linewidth=2, alpha=0.8)

plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Training Loss', fontsize=12)
plt.title('Training Loss vs Epoch - All Scenarios Comparison (Per-Epoch Federated Averaging)', fontsize=14, fontweight='bold')
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.yscale('log')

# Add text annotations for final losses
final_losses = [losses1[-1], losses2[-1], losses3[-1], federated_losses[-1], centralized_losses[-1]]
labels = ['Model 1', 'Model 2', 'Model 3', 'Federated', 'Centralized']
for i, (loss, label) in enumerate(zip(final_losses, labels)):
    plt.annotate(f'{label}: {loss:.4f}', 
                xy=(500, loss), 
                xytext=(10, 10 + i*15), 
                textcoords='offset points',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7),
                fontsize=9)

plt.tight_layout()
plt.savefig("training_loss_comparison.png", dpi=300, bbox_inches='tight')
plt.savefig("training_loss_comparison.pdf", dpi=300, bbox_inches='tight')

# MSI (Model Stability Index) Comparison
print("\nCalculating MSI for all scenarios...")

def calculate_msi(model, all_subsets):
    """
    Calculate Model Stability Index (MSI) by evaluating the model on all subsets
    and measuring the variance in performance across different data distributions.
    """
    model.eval()
    criterion = nn.MSELoss()
    losses = []
    
    with torch.no_grad():
        for subset_images, subset_angles in all_subsets:
            if len(subset_images) == 0:
                continue
            subset_loader = DataLoader(TensorDataset(subset_images, subset_angles), batch_size=batch_size, shuffle=False)
            subset_loss = 0.0
            total_samples = 0
            
            for images, angles in subset_loader:
                images = images.to(device)
                angles = angles.to(device).unsqueeze(1)
                outputs = model(images)
                loss = criterion(outputs, angles)
                subset_loss += loss.item() * images.size(0)
                total_samples += images.size(0)
            
            if total_samples > 0:
                losses.append(subset_loss / total_samples)
    
    # MSI is calculated as the coefficient of variation (std/mean) of losses across subsets
    # Lower MSI indicates better stability across different data distributions
    if len(losses) > 1:
        msi = np.std(losses) / np.mean(losses)
    else:
        msi = 0.0
    
    return msi, losses

# Prepare subset data for MSI calculation
all_subsets = [
    (subset1_images, subset1_angles),
    (subset2_images, subset2_angles),
    (subset3_images, subset3_angles),
    (all_train_images, all_train_angles)  # Full dataset
]

# Calculate MSI for each model
msi1, losses1_subsets = calculate_msi(model1, all_subsets)
msi2, losses2_subsets = calculate_msi(model2, all_subsets)
msi3, losses3_subsets = calculate_msi(model3, all_subsets)
msi_federated, losses_federated_subsets = calculate_msi(federated_model1, all_subsets)
msi_centralized, losses_centralized_subsets = calculate_msi(centralized_model, all_subsets)

# Create MSI comparison bar chart
plt.figure(figsize=(12, 8))

models = ['Model 1\n(Subset 1)', 'Model 2\n(Subset 2)', 'Model 3\n(Subset 3)', 'Federated\nModel', 'Centralized\nModel']
msi_values = [msi1, msi2, msi3, msi_federated, msi_centralized]
colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FECA57']

bars = plt.bar(models, msi_values, color=colors, alpha=0.8, edgecolor='black', linewidth=1)
# Add MSE values as text below the bars
for i, (bar, msi) in enumerate(zip(bars, msi_values)):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001, 
             f'{msi:.4f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # Add corresponding test loss (MSE) below the bar
    mse_values = [test_loss1, test_loss2, test_loss3, federated_test_loss, centralized_test_loss]
    plt.text(bar.get_x() + bar.get_width()/2, -0.01, 
             f'MSE: {mse_values[i]:.4f}', ha='center', va='top', fontsize=9, 
             style='italic', color='darkblue')
plt.ylabel('Model Stability Index (MSI)', fontsize=14)
plt.title('Model Stability Index Comparison Across Training Scenarios\n(Per-Epoch Federated Averaging)', fontsize=16, fontweight='bold')
plt.grid(True, alpha=0.3, axis='y')

# Add value labels on bars
for bar, msi in zip(bars, msi_values):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001, 
             f'{msi:.4f}', ha='center', va='bottom', fontsize=11, fontweight='bold')

# Add explanation text
plt.figtext(0.5, 0.02, 
           'MSI = Standard Deviation / Mean of losses across all subsets\nLower MSI indicates better stability across different data distributions', 
           ha='center', fontsize=10, style='italic')

plt.tight_layout()
plt.subplots_adjust(bottom=0.15)
plt.savefig("msi_comparison.png", dpi=300, bbox_inches='tight')

# Print detailed MSI results
print("\n" + "="*60)
print("MODEL STABILITY INDEX (MSI) ANALYSIS")
print("="*60)
print(f"Model 1 (trained on Subset 1): MSI = {msi1:.4f}")
print(f"  Losses on subsets: {[f'{l:.4f}' for l in losses1_subsets]}")
print(f"Model 2 (trained on Subset 2): MSI = {msi2:.4f}")
print(f"  Losses on subsets: {[f'{l:.4f}' for l in losses2_subsets]}")
print(f"Model 3 (trained on Subset 3): MSI = {msi3:.4f}")
print(f"  Losses on subsets: {[f'{l:.4f}' for l in losses3_subsets]}")
print(f"Federated Model: MSI = {msi_federated:.4f}")
print(f"  Losses on subsets: {[f'{l:.4f}' for l in losses_federated_subsets]}")
print(f"Centralized Model: MSI = {msi_centralized:.4f}")
print(f"  Losses on subsets: {[f'{l:.4f}' for l in losses_centralized_subsets]}")

# Find the most stable model
min_msi_idx = np.argmin(msi_values)
most_stable_model = models[min_msi_idx]
print(f"\nMost stable model: {most_stable_model.replace(chr(10), ' ')} (MSI = {msi_values[min_msi_idx]:.4f})")

# MSE Performance Matrix Analysis
print("\n" + "="*80)
print("MSE PERFORMANCE MATRIX ANALYSIS")
print("="*80)

def evaluate_model_on_subset(model, subset_images, subset_angles, subset_name):
    """Evaluate model on a specific subset and return MSE"""
    model.eval()
    criterion = nn.MSELoss()
    
    if len(subset_images) == 0:
        return float('inf')
    
    subset_loader = DataLoader(TensorDataset(subset_images, subset_angles), batch_size=batch_size, shuffle=False)
    total_loss = 0.0
    total_samples = 0
    
    with torch.no_grad():
        for images, angles in subset_loader:
            images = images.to(device)
            angles = angles.to(device).unsqueeze(1)
            outputs = model(images)
            loss = criterion(outputs, angles)
            total_loss += loss.item() * images.size(0)
            total_samples += images.size(0)
    
    return total_loss / total_samples if total_samples > 0 else float('inf')

# Prepare subset data
subsets = {
    'Subset 1': (subset1_images, subset1_angles),
    'Subset 2': (subset2_images, subset2_angles),
    'Subset 3': (subset3_images, subset3_angles),
    'Full Dataset': (all_train_images, all_train_angles),
    'Test Set': (test_images, test_angles)
}

models_dict = {
    'Model 1': model1,
    'Model 2': model2,
    'Model 3': model3,
    'Federated': federated_model1,
    'Centralized': centralized_model
}

# Create MSE performance matrix
mse_matrix = {}
for model_name, model in models_dict.items():
    mse_matrix[model_name] = {}
    for subset_name, (subset_imgs, subset_angs) in subsets.items():
        mse = evaluate_model_on_subset(model, subset_imgs, subset_angs, subset_name)
        mse_matrix[model_name][subset_name] = mse

# Print MSE matrix as a formatted table
subset_names = list(subsets.keys())
model_names = list(models_dict.keys())

print(f"{'Model':<12}", end="")
for subset_name in subset_names:
    print(f"{subset_name:<15}", end="")
print()
print("-" * (12 + 15 * len(subset_names)))

for model_name in model_names:
    print(f"{model_name:<12}", end="")
    for subset_name in subset_names:
        mse = mse_matrix[model_name][subset_name]
        print(f"{mse:<15.4f}", end="")
    print()

# Create heatmap visualization
import matplotlib.pyplot as plt

plt.figure(figsize=(14, 8))

# Prepare data for heatmap
heatmap_data = []
for model_name in model_names:
    row = []
    for subset_name in subset_names:
        row.append(mse_matrix[model_name][subset_name])
    heatmap_data.append(row)

heatmap_data = np.array(heatmap_data)

# Create heatmap
im = plt.imshow(heatmap_data, cmap='RdYlGn_r', aspect='auto')
plt.colorbar(im, label='MSE')

# Set labels
plt.xticks(range(len(subset_names)), subset_names, rotation=45, ha='right')
plt.yticks(range(len(model_names)), model_names)

# Add text annotations
for i in range(len(model_names)):
    for j in range(len(subset_names)):
        text = plt.text(j, i, f'{heatmap_data[i, j]:.4f}',
                       ha="center", va="center", color="black", fontweight='bold')

plt.title('MSE Performance Matrix: Models vs Training Subsets\n(Lower values are better - Per-Epoch Federated Averaging)', 
          fontsize=14, fontweight='bold')
plt.xlabel('Evaluation Dataset', fontsize=12)
plt.ylabel('Model', fontsize=12)
plt.tight_layout()
plt.savefig("mse_performance_matrix.png", dpi=300, bbox_inches='tight')

# Calculate cross-subset generalization metrics
print("\n" + "="*60)
print("CROSS-SUBSET GENERALIZATION ANALYSIS")
print("="*60)

for model_name, model in models_dict.items():
    # Calculate average MSE across all subsets
    training_mses = [mse_matrix[model_name]['Subset 1'], 
                    mse_matrix[model_name]['Subset 2'], 
                    mse_matrix[model_name]['Subset 3']]
    avg_mse = np.mean(training_mses)
    std_mse = np.std(training_mses)
    cv_mse = std_mse / avg_mse if avg_mse > 0 else float('inf')
    
    print(f"\n{model_name}:")
    print(f"  Average MSE across subsets: {avg_mse:.4f}")
    print(f"  Standard deviation: {std_mse:.4f}")
    print(f"  Coefficient of variation: {cv_mse:.4f}")
    print(f"  Test set MSE: {mse_matrix[model_name]['Test Set']:.4f}")
    print(f"  Full dataset MSE: {mse_matrix[model_name]['Full Dataset']:.4f}")

# Create bar chart comparing average cross-subset performance
plt.figure(figsize=(12, 8))

# Calculate metrics for visualization
avg_mses = []
std_mses = []
test_mses = []

for model_name in model_names:
    training_mses = [mse_matrix[model_name]['Subset 1'], 
                    mse_matrix[model_name]['Subset 2'], 
                    mse_matrix[model_name]['Subset 3']]
    avg_mses.append(np.mean(training_mses))
    std_mses.append(np.std(training_mses))
    test_mses.append(mse_matrix[model_name]['Test Set'])

x = np.arange(len(model_names))
width = 0.35

# Create grouped bar chart
bars1 = plt.bar(x - width/2, avg_mses, width, label='Average Cross-Subset MSE', 
                alpha=0.8, color='skyblue', edgecolor='black')
bars2 = plt.bar(x + width/2, test_mses, width, label='Test Set MSE', 
                alpha=0.8, color='lightcoral', edgecolor='black')

# Add error bars for cross-subset variation
plt.errorbar(x - width/2, avg_mses, yerr=std_mses, fmt='none', color='black', capsize=5)

# Add value labels
for i, (bar1, bar2) in enumerate(zip(bars1, bars2)):
    plt.text(bar1.get_x() + bar1.get_width()/2, bar1.get_height() + 0.001,
             f'{avg_mses[i]:.4f}', ha='center', va='bottom', fontweight='bold')
    plt.text(bar2.get_x() + bar2.get_width()/2, bar2.get_height() + 0.001,
             f'{test_mses[i]:.4f}', ha='center', va='bottom', fontweight='bold')

plt.xlabel('Model', fontsize=12)
plt.ylabel('MSE', fontsize=12)
plt.title('Cross-Subset Generalization Performance\n(Error bars show standard deviation across subsets - Per-Epoch Federated Averaging)', 
          fontsize=14, fontweight='bold')
plt.xticks(x, model_names, rotation=45, ha='right')
plt.legend()
plt.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig("cross_subset_generalization.png", dpi=300, bbox_inches='tight')

# Summary statistics
print("\n" + "="*60)
print("SUMMARY STATISTICS")
print("="*60)

best_test_model = model_names[np.argmin(test_mses)]
best_generalization_model = model_names[np.argmin(avg_mses)]
most_consistent_model = model_names[np.argmin(std_mses)]

print(f"Best Test Performance: {best_test_model} (MSE: {min(test_mses):.4f})")
print(f"Best Cross-Subset Generalization: {best_generalization_model} (Avg MSE: {min(avg_mses):.4f})")
print(f"Most Consistent Across Subsets: {most_consistent_model} (Std: {min(std_mses):.4f})")

# Calculate improvement metrics
federated_idx = model_names.index('Federated')
centralized_idx = model_names.index('Centralized')

test_improvement = ((test_mses[centralized_idx] - test_mses[federated_idx]) / test_mses[centralized_idx]) * 100
generalization_improvement = ((avg_mses[centralized_idx] - avg_mses[federated_idx]) / avg_mses[centralized_idx]) * 100

print(f"\nFederated vs Centralized Performance:")
print(f"  Test Set: {test_improvement:+.2f}% {'improvement' if test_improvement > 0 else 'degradation'}")
print(f"  Cross-Subset Generalization: {generalization_improvement:+.2f}% {'improvement' if generalization_improvement > 0 else 'degradation'}")
