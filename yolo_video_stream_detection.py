!pip install -U ultralytics fiftyone> /dev/null 2>&1

import gdown
import os
from ultralytics import YOLO
import cv2
import os
import time
import fiftyone as fo
import fiftyone.utils.yolo as fouy
from PIL import Image
from tqdm import tqdm
import subprocess
import numpy as np
from tabulate import tabulate

file_id = "1230efb28xUv-C7uQC5Q6bW8DWYUcaJ8R"
output = "VisDrone2019-VID-test.zip"
gdown.download(f"https://drive.google.com/uc?id={file_id}", output, quiet=False)

!unzip VisDrone2019-VID-test.zip -d ./> /dev/null 2>&1

images_dir = "./VisDrone2019-VID-test-dev/sequences/uav0000077_00720_v"
annotation_file = "./VisDrone2019-VID-test-dev/annotations/uav0000077_00720_v.txt"
labels_dir = "./VisDrone2019-VID-test-dev/annotations_yolo"

def convert_folder_to_yolo(images_dir, annotation_file, labels_dir):
    """
    Конвертирует аннотации из исходного формата VisDrone в YOLO.

    Args:
        images_dir (str): Путь к папке с изображениями.
        annotation_file (str): Путь к файлу исходных аннотаций VisDrone.
        labels_dir (str): Путь для сохранения аннотаций YOLO.
    """
    os.makedirs(labels_dir, exist_ok=True)

    # Список классов VisDrone скорректированый для YOLO
    class_names = [
        "ignored_region", "person", "person", "bicycle", "car", "car",
        "truck", "tricycle", "awning_tricycle", "bus", "motor", "others"
    ]

    def convert_box(size, box):
        dw = 1. / size[0]
        dh = 1. / size[1]
        return (box[0] + box[2] / 2) * dw, (box[1] + box[3] / 2) * dh, box[2] * dw, box[3] * dh

    with open(annotation_file, 'r') as f:
        all_rows = [x.split(',') for x in f.read().strip().splitlines()]

    for image_file in sorted(os.listdir(images_dir)):
        if not image_file.endswith('.jpg'):
            continue

        image_path = os.path.join(images_dir, image_file)
        output_file = os.path.join(labels_dir, image_file.replace('.jpg', '.txt'))

        img_size = Image.open(image_path).size
        frame_id = int(os.path.splitext(image_file)[0])

        lines = []
        for row in all_rows:
            if int(row[0]) != frame_id or int(row[-1]) == 1:  # Пропуск "ignored regions" - это области, не использущиеся для детекции.
                continue

            cls_idx = int(row[7])  # Индекс класса
            cls_name = class_names[cls_idx] if cls_idx < len(class_names) else f"class_{cls_idx}"
            xmin, ymin, width, height = map(int, row[2:6])
            box = convert_box(img_size, (xmin, ymin, width, height)) # координаты преобразуются из абсолютных значений в нормализованные (YOLO-формат).
            lines.append(f"{cls_name} {' '.join(f'{x:.6f}' for x in box)}\n")

        if lines:
            with open(output_file, 'w') as fl:
                fl.writelines(lines)

convert_folder_to_yolo(images_dir, annotation_file, labels_dir)

def create_video_in_h264(images_dir, output_video):
    """
    Создаёт видео в формате H.264 из изображений.

    Args:
        images_dir (str): Путь к папке с изображениями.
        output_video (str): Имя выходного видео.
    """
    import cv2
    import os
    import subprocess

    image_files = sorted([f for f in os.listdir(images_dir) if f.endswith('.jpg')])
    if not image_files:
        print("No images found in the directory.")
        return

    first_image = cv2.imread(os.path.join(images_dir, image_files[0]))
    height, width, _ = first_image.shape

    temp_video = "temp_output.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(temp_video, fourcc, 30, (width, height)) # Создаётся временный видеопоток для преобразования в H.264.

    for image_file in image_files:
        frame = cv2.imread(os.path.join(images_dir, image_file))
        out.write(frame)

    out.release()

    # Перекодирование в H.264
    cmd = [
        "ffmpeg", "-y", "-i", temp_video,
        "-vcodec", "libx264", "-pix_fmt", "yuv420p", "-movflags", "faststart",
        "-acodec", "aac", output_video
    ]
    try:
        subprocess.run(cmd, check=True) # Ffmpeg используется для перекодирования видео в более совместимый формат H.264.
        print(f"Видео успешно создано: {output_video}")
        os.remove(temp_video)
    except subprocess.CalledProcessError as e:
        print(f"Ошибка при создании видео: {e.stderr}")

output_video = "visdrone_h264.mp4"
create_video_in_h264(images_dir, output_video)

dataset_name = "visdrone_video_with_annotations"
if dataset_name in fo.list_datasets():
    fo.delete_dataset(dataset_name)

dataset = fo.Dataset(dataset_name)
sample = fo.Sample(filepath=output_video)

frame_annotations = []
for annotation_file in sorted(os.listdir(labels_dir)):
    if annotation_file.endswith('.txt'):
        frame_detections = []
        with open(os.path.join(labels_dir, annotation_file), 'r') as f:
            for line in f:
                fields = line.strip().split()
                cls_name = fields[0]
                x_center, y_center, bbox_width, bbox_height = map(float, fields[1:])
                bounding_box = [
                    x_center - bbox_width / 2,
                    y_center - bbox_height / 2,
                    bbox_width,
                    bbox_height,
                ]
                frame_detections.append(fo.Detection(label=cls_name, bounding_box=bounding_box))

        frame_annotations.append(fo.Detections(detections=frame_detections))

sample["frames"] = {
    idx + 1: {"ground_truth": frame_annotations[idx]} for idx in range(len(frame_annotations))
}
dataset.add_sample(sample)

session = fo.launch_app(dataset)

# Загрузка моделей YOLO с настройкой batch size
yolov3 = YOLO("yolov3.pt")

yolov5 = YOLO("yolov5n.pt")

yolov8 = YOLO("yolov8n.pt")

yolov11 = YOLO("yolo11n.pt")

# Выполнение инференса для каждой модели
dataset.apply_model(yolov3, label_field="yolov3")
dataset.apply_model(yolov5, label_field="yolov5")
dataset.apply_model(yolov8, label_field="yolov8")
dataset.apply_model(yolov11, label_field="yolov11")

session = fo.launch_app(dataset)

yolov3_results = dataset.evaluate_detections(
    "frames.yolov3",
    gt_field="frames.ground_truth",
    eval_key="eval_yolov3",
)

yolov5_results = dataset.evaluate_detections(
    "frames.yolov5",
    gt_field="frames.ground_truth",
    eval_key="eval_yolov5",
)

yolov8_results = dataset.evaluate_detections(
    "frames.yolov8",
    gt_field="frames.ground_truth",
    eval_key="eval_yolov8",
)

yolov11_results = dataset.evaluate_detections(
    "frames.yolov11",
    gt_field="frames.ground_truth",
    eval_key="eval_yolov11",
)

counts = dataset.count_values("frames.ground_truth.detections.label")
classes_top10 = sorted(counts, key=counts.get, reverse=True)[:10]

print("Yolov3:")
yolov3_results.print_report(classes_top10)

print("Yolov5:")
yolov5_results.print_report(classes_top10)

print("Yolov8:")
yolov8_results.print_report(classes_top10)

print("Yolov11:")
yolov11_results.print_report(classes_top10)

def bench_model(model, video_path):
    """
    Оценивает производительность модели на одном видео.

    Args:
        model: Загруженная модель YOLO.
        video_path (str): Путь к видеофайлу.

    Returns:
        dict: Словарь с результатами (FPS и время инференса).
    """
    cap = cv2.VideoCapture(video_path)
    inference_times = []

    total_frames = 0
    start_time = time.time()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Измерение времени инференса на кадр
        frame_start = time.time()
        _ = model.predict(source=frame, save=False) # Замеряется время обработки каждого кадра для вычисления производительности.
        frame_time = time.time() - frame_start
        inference_times.append(frame_time)
        total_frames += 1

    elapsed_time = time.time() - start_time
    cap.release()

    avg_inference_time = sum(inference_times) / len(inference_times) if inference_times else 0
    fps = total_frames / elapsed_time if elapsed_time > 0 else 0

    return {
        "avg_inference_time": avg_inference_time,
        "fps": fps
    }

def test_optimized(video_path, ckpt_path, batch_size=16):
    """
    Выполняет пакетную обработку видео с заданной моделью,
    а затем передаёт модель для оценки в bench_model.

    Args:
        video_path (str): Путь к видеофайлу.
        ckpt_path (str): Путь к контрольной точке модели.
        batch_size (int): Размер пакета для инференса.

    Returns:
        dict: Результаты тестирования с помощью bench_model.
    """
    model = YOLO(ckpt_path)
    cap = cv2.VideoCapture(video_path)
    frames = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frames.append(frame)

        # Кадры отправляются на обработку, как только достигается размер пакета.
        if len(frames) == batch_size:
            # Выполнение пакетного инференса
            _ = model.predict(source=frames, save=False)
            frames = []

    # Обработка оставшихся кадров
    if frames:
        _ = model.predict(source=frames, save=False)

    cap.release()

    # Вызываем bench_model для оценки
    return bench_model(model, video_path)

def test_fp16(video_path, ckpt_path):
    model = YOLO(ckpt_path)
    model.overrides["half"] = True  # Включение FP16 для ускорения на GPU
    return bench_model(model, video_path)

def collect_and_display_results(models, video_path, test_functions):
    results = []
    for model_name, ckpt_path in models.items():
        for test_name, test_func in test_functions.items():
            print(f"\n--- {test_name}: {model_name} ---")
            test_result = test_func(video_path, ckpt_path)
            results.append({
                "Model": model_name,
                "Test Type": test_name,
                "Average Inference Time (s)": test_result["avg_inference_time"],
                "FPS": test_result["fps"]
            })

    return results

def display_results_table(results):
    """
    Выводит результаты тестирования моделей в виде таблицы.

    Args:
        results (list): Список словарей с результатами тестирования.
    """
    print("\nРезультаты тестирования:")
    print(tabulate(results, headers="keys", tablefmt="grid"))

models = {
    "YOLOv3": "yolov3.pt",
    "YOLOv5": "yolov5n.pt",
    "YOLOv8": "yolov8n.pt",
    "YOLOv11": "yolo11n.pt"
}

test_functions = {
    "Optimized": lambda video_path, ckpt_path: test_optimized(video_path, ckpt_path, batch_size=16),
    "FP16": test_fp16,
}

video_path = "visdrone_h264.mp4"
results = collect_and_display_results(models, video_path, test_functions)

display_results_table(results)

display_results_table(results)
