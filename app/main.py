from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image
import io
import logging
import torchvision.transforms as transforms
import asyncio

from starlette.middleware.cors import CORSMiddleware

from app.model_loader import model_loader

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ✅ ОБСЛУЖИВАЕМ СТАТИЧЕСКИЕ ФАЙЛЫ
app.mount("/static", StaticFiles(directory="static"), name="static")

# Простые трансформации
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])


@app.on_event("startup")
async def startup():
    """Загружаем модель при старте"""
    logger.info("Запуск API...")
    model_loader.load_model()


# ✅ УДАЛИЛ ВСЕ СТАРЫЕ ЭНДПОИНТЫ И ОСТАВИЛ ТОЛЬКО ЭТОТ
@app.get("/")
async def read_root():
    """Главная страница с веб-интерфейсом"""
    try:
        # Читаем HTML файл
        with open("static/index.html", "r", encoding="utf-8") as f:
            html_content = f.read()
        logger.info("HTML интерфейс загружен")
        return HTMLResponse(content=html_content)
    except Exception as e:
        logger.error(f"Ошибка загрузки HTML: {e}")
        # Если файл не найден, возвращаем простой HTML
        return HTMLResponse(content="""
        <html>
            <head><title>CIFAR-10 Classifier</title></head>
            <body>
                <h1>🎯 CIFAR-10 Классификатор изображений</h1>
                <p>Файл интерфейса не найден. Проверьте что файл static/index.html существует.</p>
                <p>Ошибка: """ + str(e) + """</p>
            </body>
        </html>
        """)


@app.get("/health")
async def health():
    return {
        "status": "healthy" if model_loader.is_loaded else "error",
        "model_loaded": model_loader.is_loaded
    }


@app.post("/predict")
async def predict(image: UploadFile = File(...)):
    """Простой эндпоинт для предсказания"""
    if not model_loader.is_loaded:
        raise HTTPException(status_code=503, detail="Модель не загружена")

    try:
        # Читаем изображение
        contents = await image.read()

        # Обрабатываем в отдельном потоке чтобы не блокировать
        def process_image():
            img = Image.open(io.BytesIO(contents)).convert('RGB')
            tensor = transform(img).unsqueeze(0).to(model_loader.device)
            return model_loader.predict(tensor)

        # Запускаем в отдельном потоке
        result = await asyncio.get_event_loop().run_in_executor(None, process_image)

        return result

    except Exception as e:
        logger.error(f"Ошибка: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/classes")
async def get_classes():
    if not model_loader.is_loaded:
        raise HTTPException(status_code=503, detail="Модель не загружена")
    return {"classes": model_loader.class_names}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8001)