# Dog Breed Classification with PyTorch & ResNet18

A deep learning project that classifies images of dogs into 120 breeds using PyTorch and a fine-tuned ResNet18 model.  
Trained on the [Stanford Dog Dataset](http://vision.stanford.edu/aditya86/ImageNetDogs/).

---

## Features

- Classifies 120 dog breeds from images
- PyTorch implementation with ResNet18 backbone
- Easy-to-use training and inference scripts
- Outputs prediction label and confidence score
- Visualizes predictions and class example images

---

## Installation

```bash
git clone https://github.com/Gh0st-Script/resnet-dog-breed-classification.git
cd resnet-dog-breed-classification
pip install -r requirements.txt
```

---

## Dataset

- **Stanford Dog Dataset:** [Download here](http://vision.stanford.edu/aditya86/ImageNetDogs/)
- **Note:** Dataset is not included in this repo due to size. Please download and extract it as instructed below.

**Directory Structure:**
```
dog_dataset/
    breed1/
    breed2/
    ...
```

---

## Usage

### **1. Train the Model**

```bash
python train.py --data_dir ./dog_dataset --epochs 10 --save_model ./model/img_class_mdl_final.pth
```

### **2. Inference on a New Image**

```bash
python inference.py --img_path path/to/your/image.jpg
```

This will print the predicted breed and show comparison images.

---

## Results

| Metric      | Value   |
|-------------|---------|
| Accuracy    | 93.55%     |

**Example Output:**

![example](images/example_prediction_1.jpg)

---

## Project Structure

```
dog-breed-classification-pytorch/
├── README.md
├── requirements.txt
├── train.py
├── inference.py
├── model/
├── utils.py
├── dog_dataset/
```

---

## License

[MIT](LICENSE)

---

## References

- [Stanford Dogs Dataset](http://vision.stanford.edu/aditya86/ImageNetDogs/)
- [PyTorch Documentation](https://pytorch.org/)
- [ResNet Paper](https://arxiv.org/abs/1512.03385)
