import torch
import torch.nn as nn
import json
import logging
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 8 * 8, 512)
        self.fc2 = nn.Linear(512, 10)
        self.dropout = nn.Dropout(0.25)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 64 * 8 * 8)
        x = self.dropout(torch.relu(self.fc1(x)))
        x = self.fc2(x)
        return x


class ModelLoader:
    def __init__(self):
        self.model = None
        self.class_names = []
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.is_loaded = False

    def load_model(self):
        """Упрощенная загрузка модели"""
        try:
            logger.info("🔄 Загрузка модели...")

            # Проверяем файлы
            if not os.path.exists("models/cifar10_baseline.pth"):
                logger.error("❌ Файл модели не найден")
                return False

            if not os.path.exists("models/class_names.json"):
                logger.error("❌ Файл классов не найден")
                return False

            # Загружаем классы
            with open("models/class_names.json", 'r') as f:
                self.class_names = json.load(f)

            logger.info(f"✅ Классы: {self.class_names}")

            # Создаем и загружаем модель
            self.model = SimpleCNN()
            self.model.load_state_dict(torch.load("models/cifar10_baseline.pth", map_location=self.device))
            self.model.to(self.device)
            self.model.eval()

            self.is_loaded = True
            logger.info("✅ Модель загружена успешно!")
            return True

        except Exception as e:
            logger.error(f"❌ Ошибка: {e}")
            return False

    def predict(self, image_tensor):
        """Упрощенное предсказание"""
        if not self.is_loaded:
            return {"error": "Модель не загружена"}

        with torch.no_grad():
            outputs = self.model(image_tensor)
            probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
            predicted_idx = torch.argmax(probabilities).item()

            return {
                "predicted_class": self.class_names[predicted_idx],
                "confidence": probabilities[predicted_idx].item(),
                "all_predictions": [
                    {"class": self.class_names[i], "confidence": prob.item()}
                    for i, prob in enumerate(probabilities)
                ]
            }


# Глобальный экземпляр
model_loader = ModelLoader()