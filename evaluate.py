import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import timm
from sklearn.metrics import classification_report, confusion_matrix

data_dir = r'------------\Images'
model_path = "model.pth"
batch_size = 32
num_workers = 2
num_classes = 120

val_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])
val_dataset = datasets.ImageFolder(root=data_dir, transform=val_transforms)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = timm.create_model('resnet18', pretrained=False, num_classes=num_classes)
checkpoint = torch.load(model_path, map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model.to(device)
model.eval()

all_preds = []
all_labels = []

with torch.no_grad():
    for images, targets in val_loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(targets.cpu().numpy())

idx_to_class = {v: k for k, v in val_dataset.class_to_idx.items()}
target_names = [idx_to_class[i] for i in range(len(idx_to_class))]

print("\nDetailed classification report on validation set:")
print(classification_report(all_labels, all_preds, target_names=target_names, digits=4))
print("Confusion matrix:")
print(confusion_matrix(all_labels, all_preds))