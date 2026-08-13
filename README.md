# erase-it

Веб-приложение для удаления нежелательных объектов с фотографий: пользователь загружает
изображение, указывает объект (кликом), а ИИ вырезает его и правдоподобно закрашивает
получившуюся дыру фоном.

## Как это работает

1. **Локализация объекта** — пользователь кликает по объекту на canvas, точка отправляется
   в предобученную модель [SAM (Segment Anything)](https://github.com/facebookresearch/segment-anything),
   которая возвращает маску объекта.
2. **Инпейнтинг** — маска и изображение передаются в собственную модель на PyTorch
   (gated-convolution генератор + PatchGAN дискриминатор, обучена с нуля), которая
   закрашивает вырезанную область реалистичным фоном.

Локализация объекта используется готовая — обучать сегментацию/детекцию с нуля
нецелесообразно. Модель инпейнтинга обучается самостоятельно на PyTorch — это основная
ML-часть проекта.

## Структура репозитория

```
app/
  backend/     # FastAPI: /segment, /inpaint
  frontend/    # веб-интерфейс (upload + canvas)
ml/
  datasets/    # генерация масок, PyTorch Dataset
  models/      # генератор, дискриминатор
  training/    # тренировочный цикл, лоссы, конфиг
  notebooks/   # ноутбук для обучения на Kaggle GPU
  checkpoints/ # веса моделей (gitignored)
tests/         # тесты датасета, моделей, backend
data/          # датасеты для обучения (gitignored)
```

## Roadmap

- [x] Data pipeline для инпейнтинга (irregular masks + датасет сцен)
- [x] Baseline модель инпейнтинга (encoder-decoder, L1 loss)
- [ ] Adversarial-обучение (PatchGAN, GAN loss)
- [ ] Интеграция SAM для выбора объекта по клику
- [ ] Backend API (FastAPI)
- [ ] Web-фронтенд
- [ ] Полировка и финальное тестирование

## Обучение модели

Обучение инпейнтинг-модели происходит на [Kaggle Notebooks](https://www.kaggle.com/) (бесплатный
GPU) на подвыборке датасета Places365-small. Ноутбук для запуска: `ml/notebooks/kaggle_train.ipynb`.

## Запуск (появится по мере готовности backend/frontend)

```bash
pip install -r requirements.txt
uvicorn app.backend.main:app --reload
```
