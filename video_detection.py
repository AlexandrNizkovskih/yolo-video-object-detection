import os
import cv2
import time
import subprocess
from typing import Dict, Callable, List

import gdown
from ultralytics import YOLO
import fiftyone as fo
from PIL import Image
from tabulate import tabulate


def download_dataset(file_id: str, output_zip: str) -> str:
    """Download dataset from Google Drive using gdown and unzip it."""
    gdown.download(f"https://drive.google.com/uc?id={file_id}", output_zip, quiet=False)
    subprocess.run(["unzip", output_zip, "-d", "./"], check=True)
    return output_zip


def convert_folder_to_yolo(images_dir: str, annotation_file: str, labels_dir: str) -> None:
    """Convert VisDrone annotations to YOLO format."""
    os.makedirs(labels_dir, exist_ok=True)

    class_names = [
        "ignored_region", "person", "person", "bicycle", "car", "car",
        "truck", "tricycle", "awning_tricycle", "bus", "motor", "others"
    ]

    def convert_box(size, box):
        dw = 1.0 / size[0]
        dh = 1.0 / size[1]
        return (
            (box[0] + box[2] / 2) * dw,
            (box[1] + box[3] / 2) * dh,
            box[2] * dw,
            box[3] * dh,
        )

    with open(annotation_file, "r") as f:
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
            if int(row[0]) != frame_id or int(row[-1]) == 1:
                continue

            cls_idx = int(row[7])
            cls_name = class_names[cls_idx] if cls_idx < len(class_names) else f"class_{cls_idx}"
            xmin, ymin, width, height = map(int, row[2:6])
            box = convert_box(img_size, (xmin, ymin, width, height))
            lines.append(f"{cls_name} {' '.join(f'{x:.6f}' for x in box)}\n")

        if lines:
            with open(output_file, "w") as fl:
                fl.writelines(lines)


def create_video_in_h264(images_dir: str, output_video: str) -> None:
    """Create an H.264 video from image sequence."""
    image_files = sorted(f for f in os.listdir(images_dir) if f.endswith('.jpg'))
    if not image_files:
        raise FileNotFoundError("No images found in the directory")

    first_image = cv2.imread(os.path.join(images_dir, image_files[0]))
    height, width, _ = first_image.shape

    temp_video = "temp_output.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(temp_video, fourcc, 30, (width, height))

    for image_file in image_files:
        frame = cv2.imread(os.path.join(images_dir, image_file))
        out.write(frame)

    out.release()

    cmd = [
        "ffmpeg", "-y", "-i", temp_video,
        "-vcodec", "libx264", "-pix_fmt", "yuv420p", "-movflags", "faststart",
        "-acodec", "aac", output_video
    ]
    subprocess.run(cmd, check=True)
    os.remove(temp_video)


def bench_model(model: YOLO, video_path: str) -> Dict[str, float]:
    """Benchmark YOLO model on a video."""
    cap = cv2.VideoCapture(video_path)
    inference_times: List[float] = []
    total_frames = 0
    start_time = time.time()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_start = time.time()
        _ = model.predict(source=frame, save=False)
        inference_times.append(time.time() - frame_start)
        total_frames += 1

    elapsed_time = time.time() - start_time
    cap.release()

    avg_inference_time = sum(inference_times) / len(inference_times) if inference_times else 0
    fps = total_frames / elapsed_time if elapsed_time > 0 else 0
    return {"avg_inference_time": avg_inference_time, "fps": fps}


def test_optimized(video_path: str, ckpt_path: str, batch_size: int = 16) -> Dict[str, float]:
    model = YOLO(ckpt_path)
    cap = cv2.VideoCapture(video_path)
    frames: List = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
        if len(frames) == batch_size:
            _ = model.predict(source=frames, save=False)
            frames = []
    if frames:
        _ = model.predict(source=frames, save=False)
    cap.release()
    return bench_model(model, video_path)


def test_fp16(video_path: str, ckpt_path: str) -> Dict[str, float]:
    model = YOLO(ckpt_path)
    model.overrides["half"] = True
    return bench_model(model, video_path)


def collect_and_display_results(models: Dict[str, str], video_path: str, test_functions: Dict[str, Callable]) -> List[Dict[str, float]]:
    results: List[Dict[str, float]] = []
    for model_name, ckpt_path in models.items():
        for test_name, test_func in test_functions.items():
            print(f"\n--- {test_name}: {model_name} ---")
            test_result = test_func(video_path, ckpt_path)
            results.append({
                "Model": model_name,
                "Test Type": test_name,
                "Average Inference Time (s)": test_result["avg_inference_time"],
                "FPS": test_result["fps"],
            })
    return results


def display_results_table(results: List[Dict[str, float]]) -> None:
    print("\nРезультаты тестирования:")
    print(tabulate(results, headers="keys", tablefmt="grid"))


def main() -> None:
    dataset_id = "1230efb28xUv-C7uQC5Q6bW8DWYUcaJ8R"
    zip_name = "VisDrone2019-VID-test.zip"
    download_dataset(dataset_id, zip_name)

    images_dir = "./VisDrone2019-VID-test-dev/sequences/uav0000077_00720_v"
    annotation_file = "./VisDrone2019-VID-test-dev/annotations/uav0000077_00720_v.txt"
    labels_dir = "./VisDrone2019-VID-test-dev/annotations_yolo"

    convert_folder_to_yolo(images_dir, annotation_file, labels_dir)

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

    sample["frames"] = {idx + 1: {"ground_truth": frame_annotations[idx]} for idx in range(len(frame_annotations))}
    dataset.add_sample(sample)

    yolov3 = YOLO("yolov3.pt")
    yolov5 = YOLO("yolov5n.pt")
    yolov8 = YOLO("yolov8n.pt")
    yolov11 = YOLO("yolo11n.pt")

    dataset.apply_model(yolov3, label_field="yolov3")
    dataset.apply_model(yolov5, label_field="yolov5")
    dataset.apply_model(yolov8, label_field="yolov8")
    dataset.apply_model(yolov11, label_field="yolov11")

    test_functions = {
        "Optimized": lambda video, ckpt: test_optimized(video, ckpt, batch_size=16),
        "FP16": test_fp16,
    }
    models = {
        "YOLOv3": "yolov3.pt",
        "YOLOv5": "yolov5n.pt",
        "YOLOv8": "yolov8n.pt",
        "YOLOv11": "yolo11n.pt",
    }

    results = collect_and_display_results(models, output_video, test_functions)
    display_results_table(results)

    fo.launch_app(dataset)


if __name__ == "__main__":
    main()
