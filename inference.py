import torch
from torchvision import datasets, transforms
from PIL import Image
import matplotlib.pyplot as plt
import random
import timm


img_path = r"-------\Images"

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
])


# Load the model checkpoint
checkpoint = torch.load("img_class_mdl_final.pth", map_location="cpu")
class_to_idx = checkpoint["class_to_idx"]
idx_to_class = {v: k for k, v in class_to_idx.items()}
num_classes = len(class_to_idx)

model = timm.create_model('resnet18', pretrained=False, num_classes=num_classes)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()


# Load the dataset
dataset = datasets.ImageFolder(root=img_path, transform=transform)


# Ensure the model is in evaluation mode
def predict_and_show(img_path_to_test):
    
    img = Image.open(img_path_to_test).convert('RGB')
    input_tensor = transform(img).unsqueeze(0)
    with torch.no_grad():
        outputs = model(input_tensor)
        _, pred = torch.max(outputs, 1)
        pred_class_idx = pred.item()
        pred_class_name = idx_to_class[pred_class_idx]

    print(f"Predicted class: {pred_class_name}")

    plt.figure(figsize=(10,4))
    plt.subplot(1, 4, 1)
    plt.imshow(img)
    plt.title("Input Image")
    plt.axis('off')

    class_idx = pred_class_idx
    candidates = [path for path, label in dataset.samples if label == class_idx]
    if len(candidates) >= 3:
        chosen = random.sample(candidates, 3)
    else:
        chosen = candidates

    for i, cpath in enumerate(chosen):
        cimg = Image.open(cpath).convert('RGB')
        plt.subplot(1, 4, i+2)
        plt.imshow(cimg)
        plt.title(f"Class Example {i+1}")
        plt.axis('off')

    plt.suptitle(f"Prediction: {pred_class_name}")
    plt.tight_layout()
    plt.show()

# Example usage
predict_and_show(r"------------\dog1.jpeg")