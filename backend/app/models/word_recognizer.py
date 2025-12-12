import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import joblib
import numpy as np


class WordRecognizer:
    def __init__(self, model_path: str, encoder_path: str, device: str = "cpu"):
        """
        Initialize word recognizer
        """
        self.device = torch.device(device)
        
        # Load label encoder
        self.label_encoder = joblib.load(encoder_path)
        self.num_classes = len(self.label_encoder.classes_)
        print(f"Loaded recognizer with {self.num_classes} classes")
        
        # Load model
        self.model = self._build_model()
        self.model.load_state_dict(
            torch.load(model_path, map_location=self.device)
        )
        self.model.to(self.device)
        self.model.eval()
        
        # Define transforms (same as training)
        self.transform = transforms.Compose([
            transforms.Resize((48, 160)),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize([0.5]*3, [0.5]*3)
        ])
    
    def _build_model(self) -> nn.Module:
        """Build ResNet18 model with custom classifier"""
        model = models.resnet18(pretrained=False)
        model.fc = nn.Linear(model.fc.in_features, self.num_classes)
        return model
    
    def recognize_word(self, image: Image.Image) -> str:
        """Recognize word from image"""
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        img_tensor = self.transform(image).unsqueeze(0)
        img_tensor = img_tensor.to(self.device)
        
        with torch.no_grad():
            outputs = self.model(img_tensor)
            _, predicted = torch.max(outputs, 1)
        
        word = self.label_encoder.inverse_transform([predicted.item()])[0]
        return word
    
    def recognize_words_batch(self, images: list) -> list:
        """Recognize multiple words at once"""
        if not images:
            return []
        
        img_tensors = torch.stack([
            self.transform(img.convert('RGB') if img.mode != 'RGB' else img)
            for img in images
        ]).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(img_tensors)
            _, predicted = torch.max(outputs, 1)
        
        words = self.label_encoder.inverse_transform(predicted.cpu().numpy())
        return words.tolist()