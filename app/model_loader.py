import torch
import torch.nn as nn
import json
import logging
from typing import List

from blackd.middlewares import F

logger = logging.getLogger(__name__)

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding= 1 )
        self.conv2 = nn.Conv2d(32, 64, 3, padding= 1 )
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(4096, 512)
        self.fc2 = nn.Linear(512, 10)
        self.dropout = nn.Dropout(0.25)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 4096)
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

class ModelLoader:

    def __init__(self):
        self.model = None
        self.class_names = []
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.is_loaded = False

    def load_model(self, model_path: str = "models/cifar10_baseline.pth",
                   class_path: str = "models/class_names.json"):
        try:
            with open(class_path, "r") as f:
                class_names = json.load(f)

            self.model = SimpleCNN()
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            self.model.to(self.device)
            self.model.eval()

            self.is_loaded = True
            logger.info(f"Модель загруженна на устройство: {self.device}")
            logger.info(f"Загруженно классов: {len(self.class_names)}")

            return True

        except Exception as e:
            logger.error(f"Ошибка загрузки модели: {e}")
            self.is_loaded = False
            return False

    def predict(self, image_tensor: torch.Tensor) -> dict:

        if not self.is_loaded:
            raise ValueError("Модель не загружена")

        with torch.no_grad():
            outputs = self.model(image_tensor)
            probabilities = torch.nn.functional.softmax(outputs[0], dim=0)

        predicted_idx = torch.argmax(probabilities).item()
        predicted_class = self.class_names[predicted_idx]
        confidence = probabilities[predicted_idx].item()

        all_predictions = [
            {"class": self.class_names[i], "confidence": prob.item()}
            for i, prob in enumerate(probabilities)
        ]

        return {
            "predicted_class": predicted_class,
            "confidence": confidence,
            "all_predictions": all_predictions
        }

model_loader = ModelLoader()