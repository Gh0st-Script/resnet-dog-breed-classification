import torch
import numpy as np
import timm
import random
import torch.nn as nn
from tqdm import tqdm
from torchvision.datasets import ImageFolder
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import classification_report, confusion_matrix


torch.manual_seed(42)
np.random.seed(42)
random.seed(42)


print(torch.__version__)
print("CUDA available:",torch.cuda.is_available())
print("CUDA Device name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A")

# Define the path to your dataset
img_path = r"------\---\Images" # Replace '------\---\Images' with the actual path to your dataset

# Define the transformations for the dataset
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std = [0.229,0.224,0.225])
])

# Load the dataset
base_dataset = datasets.ImageFolder(root = img_path, transform = transform)

train_size = int(0.8 * len(base_dataset))
val_size = len(base_dataset) - train_size

train_dataset, val_dataset = random_split(base_dataset, [train_size, val_size], generator = torch.Generator().manual_seed(42))


# Create DataLoaders for training and validation datasets
train_loader = DataLoader(train_dataset, batch_size = 16, shuffle = True, num_workers = 2, pin_memory = True)
val_loader = DataLoader(val_dataset, batch_size = 16, shuffle = False, num_workers = 2, pin_memory = True)


# Define the model
num_classes = len(base_dataset.classes)
model = timm.create_model('resnet18', pretrained = True,num_classes = 120)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)


# Training function
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(),lr = 1e-4)


# Training loop
num_epochs = 10

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    
    train_loader_iter = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{num_epochs}] Training", leave=False)
    for images, targets in train_loader_iter:
        images = images.to(device)
        targets = targets.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == targets).sum().item()
        total += targets.size(0)
        
    epoch_loss = running_loss / total
    epoch_acc = correct / total
    
    print(f"Epoch [{epoch +1}/{num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}")
    
    model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0
    with torch.no_grad():
        
        val_loader_iter = tqdm(val_loader, desc=f"Epoch [{epoch+1}/{num_epochs}] Validation", leave=False)
        for images, targets in val_loader_iter:
            images = images.to(device)
            targets = targets.to(device)
            outputs = model(images)
            loss = criterion(outputs,targets)
            val_loss += loss.item() * images.size(0)
            _,predicted = torch.max(outputs,1)
            val_correct +=(predicted == targets).sum().item()
            val_total += targets.size(0)
        
        val_epoch_loss = val_loss / val_total
        val_epoch_acc = val_correct / val_total
        print(f"Validation Loss: {val_epoch_loss:.4f}, Validation Accuracy: {val_epoch_acc:.4f}")
        

# Save the model and class mapping
torch.save({
    'model_state_dict': model.state_dict(),
    'class_to_idx': base_dataset.class_to_idx
}, "img_class_mdl_final.pth")

print("Training complete. Model and class mapping saved!")


# Evaluation on the validation set
model.eval()


all_preds = []
all_labels = []

with torch.no_grad():
    for images, targets in val_loader:
        images = images.to(device, non_blocking = True)
        targets = targets.to(device, non_blocking = True)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(targets.cpu().numpy())
        
        
idx_to_class = {v: k for k, v in base_dataset.class_to_idx.items()}
target_names = [idx_to_class[i] for i in range(len(idx_to_class))]

print("\nDetailed classification report on validation set:")
print(classification_report(all_labels, all_preds, target_names=target_names, digits=4))
print("Confusion matrix:")
print(confusion_matrix(all_labels, all_preds))

# Run the Model
def predict_image(img_path, dataset=base_dataset, transform=transform, model=model, device=device):
    model.eval()
    img = Image.open(img_path).convert('RGB')
    x = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(x)
        _, pred = outputs.max(1)
    idx_to_class = {v: k for k, v in dataset.class_to_idx.items()}
    class_name = idx_to_class[pred.item()]
    print(f"Prediction: {class_name}")


    predicted_class_idx = pred.item()
    candidate_paths = [path for path, label in dataset.samples if label == predicted_class_idx]
    shown_examples = random.sample(candidate_paths, min(3, len(candidate_paths)))

    plt.figure(figsize=(12, 4))

    plt.subplot(1, 4, 1)
    plt.imshow(img)
    plt.title("Input Image")
    plt.axis('off')

    for i, cpath in enumerate(shown_examples):
        compare_img = Image.open(cpath).convert('RGB')
        plt.subplot(1, 4, i + 2)
        plt.imshow(compare_img)
        plt.title(f"Class Example {i+1}")
        plt.axis('off')

    plt.suptitle(f"Prediction: {class_name}")
    plt.tight_layout()
    plt.show()

predict_image(r"------------\dog1.jpeg")
