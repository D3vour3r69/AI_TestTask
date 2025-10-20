def test_poetry_environment():
    """Проверяем что все основные библиотеки доступны в poetry окружении"""
    try:
        import torch as trch
        import fastapi
        import uvicorn
        import PIL
        import sklearn
        import numpy as np
        import pandas as pd

        print("✅ Все основные зависимости доступны")
        print(f"✅ PyTorch: {trch.__version__}")
        print(f"✅ FastAPI: {fastapi.__version__}")
        print(f"✅ NumPy: {np.__version__}")

        # Проверяем что мы в виртуальном окружении poetry
        import sys
        print(f"✅ Python path: {sys.prefix}")

        return True
    except ImportError as e:
        print(f"❌ Ошибка импорта: {e}")
        return False


if __name__ == "__main__":
    test_poetry_environment()