# Детекция объектов в видеопотоке (YOLO)

## О проекте
Данный репозиторий демонстрирует использование моделей семейства **YOLO** для обработки видео и сравнения их производительности. В ноутбуке проводился анализ стандартного инференса и оптимизаций (пакетная обработка и вычисления в формате **FP16**), что отражено в данном проекте.

## Теоретическая часть
- **YOLO (You Only Look Once)** — однопроходный детектор, который определяет и локализует объекты за один проход по изображению.
- **Пакетная обработка** позволяет обрабатывать несколько кадров одновременно и тем самым уменьшает суммарное время инференса.
- **FP16** (half precision) ускоряет вычисления на GPU при незначительной потере точности.
- **FiftyOne** используется для визуализации и оценки результатов инференса.

## Архитектура YOLO
Модели YOLO делят изображение на сетку и для каждой ячейки предсказывают ограничивающие рамки с вероятностями классов. Новые версии (YOLOv5, YOLOv8) используют улучшенные блоки (например, CSP) и поддерживают вычисления в FP16. Для удаления избыточных рамок применяется Non‑Maximum Suppression.

## Подготовка данных
В качестве примера используется набор данных **VisDrone2019-VID**. Аннотации конвертируются в формат YOLO, а изображения собираются в видеопоток. Для анализа используется интерфейс FiftyOne.

## Выполнение инференса и оценка
Скрипт `video_detection.py` скачивает тестовый датасет, создаёт видео, применяет несколько моделей YOLO и выводит метрики. Также предоставлены функции для оптимизации инференса (батчинг и FP16). Результаты выводятся в виде таблицы и в интерфейсе FiftyOne.

## Основные результаты
- **YOLOv8** показала лучшую скорость в оптимизированном режиме, что делает её подходящей для задач реального времени.
- **YOLOv5** демонстрирует баланс между точностью и производительностью.
- Использование **FP16** уменьшает среднее время инференса, однако прирост FPS ограничен пропускной способностью GPU.

## Запуск
---

## Запуск


```bash
pip install -r requirements.txt
python video_detection.py
```

Скрипт содержит полный набор шагов: загрузку датасета, преобразование аннотаций, создание видео и сравнение нескольких моделей YOLO.
=======

Файл `video_detection.py` содержит полный скрипт для подготовки датасета VisDrone, конвертации аннотаций в формат YOLO, создания видео и сравнения нескольких версий модели YOLO.

