import cv2
import time
import torch
import numpy as np
import psutil
import threading
from collections import deque
import argparse
import json
from datetime import datetime
import sys
from pathlib import Path
try:
    from safetensors.torch import load_file as safetensors_load_file
except ImportError:
    safetensors_load_file = None
try:
    from torchvision.models.detection import detr_resnet50
except ImportError:
    detr_resnet50 = None
model_path = "models/detr-finetuned-epoch-15-20251225T080504Z-3-001/detr-finetuned-epoch-15/model.safetensors"

class DETRBenchmark:
    def __init__(self, model_path, device="cuda", warmup_frames=30):
        """
        DETR GPU Benchmark
        
        Args:
            model_path (str): Path to DETR model
            device (str): Device to run inference on ('cuda' for GPU)
            warmup_frames (int): Number of frames for GPU warmup
        """
        self.model_path = model_path
        self.device = device
        self.warmup_frames = warmup_frames

        self.fps_history = deque(maxlen=100)
        self.inference_times = deque(maxlen=100)
        self.preprocessing_times = deque(maxlen=100)
        self.postprocessing_times = deque(maxlen=100)
        self.gpu_memory_usage = deque(maxlen=100)
        self.cpu_usage = deque(maxlen=100)
        self.gpu_utilization = deque(maxlen=100)

        if device == "cuda" and not torch.cuda.is_available():
            print("Warning: CUDA not available, falling back to CPU")
            self.device = "cpu"
        else:
            self.device = device

        print(f"Loading DETR model: {model_path}")
        model_path = Path(model_path)
        if model_path.suffix == ".safetensors":
            if safetensors_load_file is None:
                raise ImportError("safetensors is not installed. Please install with 'pip install safetensors'.")
            if detr_resnet50 is None:
                raise ImportError("torchvision>=0.13 is required for detr_resnet50.")
            # Model yapısı oluşturuluyor
            self.model = detr_resnet50(pretrained=False, num_classes=91)
            state_dict = safetensors_load_file(str(model_path))
            self.model.load_state_dict(state_dict)
        else:
            self.model = torch.load(model_path, map_location=device)
        self.model.eval()
        self.model.to(device)

        # Dummy class names for visualization
        self.class_names = [str(i) for i in range(1000)]

        if self.device == "cuda" and torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name()}")
            print(f"CUDA Version: {torch.version.cuda}")
            print(f"Total GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
        else:
            print(f"Using device: {self.device.upper()}")

    def monitor_system_resources(self):
        while self.monitoring:
            try:
                cpu_percent = psutil.cpu_percent(interval=0.1)
                self.cpu_usage.append(cpu_percent)
                if torch.cuda.is_available():
                    gpu_memory = torch.cuda.memory_allocated() / 1024**3
                    self.gpu_memory_usage.append(gpu_memory)
                    gpu_util = 0  # Placeholder, as torch does not provide utilization directly
                    self.gpu_utilization.append(gpu_util)
                time.sleep(0.5)
            except Exception as e:
                print(f"Resource monitoring error: {e}")
                break

    def warmup_gpu(self, input_size=(640, 640)):
        print(f"Warming up GPU with {self.warmup_frames} frames...")
        dummy_image = np.random.randint(0, 255, (input_size[1], input_size[0], 3), dtype=np.uint8)
        for i in range(self.warmup_frames):
            with torch.no_grad():
                _ = self.run_inference(dummy_image)
            if i % 10 == 0:
                print(f"Warmup progress: {i}/{self.warmup_frames}")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print("GPU warmup completed!")

    def preprocess_frame(self, frame):
        start_time = time.time()
        # Example: resize and normalize as DETR expects
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (640, 640))
        img = img.astype(np.float32) / 255.0
        img = (img - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
        img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).to(self.device)
        preprocessing_time = (time.time() - start_time) * 1000
        self.preprocessing_times.append(preprocessing_time)
        return img

    def run_inference(self, img_tensor):
        start_time = time.time()
        with torch.no_grad():
            outputs = self.model(img_tensor)
        inference_time = (time.time() - start_time) * 1000
        self.inference_times.append(inference_time)
        return outputs

    def postprocess_results(self, outputs, frame, conf_thresh=0.5):
        start_time = time.time()
        annotated_frame = frame.copy()
        # Example: postprocess for DETR (assuming COCO classes)
        if isinstance(outputs, dict) and "pred_logits" in outputs and "pred_boxes" in outputs:
            logits = outputs["pred_logits"].softmax(-1)[0, :, :-1]
            boxes = outputs["pred_boxes"][0]
            scores, labels = logits.max(-1)
            h, w, _ = frame.shape
            for score, label, box in zip(scores, labels, boxes):
                if score > conf_thresh:
                    cx, cy, bw, bh = box.cpu().numpy()
                    x1 = int((cx - bw/2) * w)
                    y1 = int((cy - bh/2) * h)
                    x2 = int((cx + bw/2) * w)
                    y2 = int((cy + bh/2) * h)
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    label_text = f"{self.class_names[label]}: {score:.2f}"
                    cv2.putText(annotated_frame, label_text, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
        postprocessing_time = (time.time() - start_time) * 1000
        self.postprocessing_times.append(postprocessing_time)
        return annotated_frame

    def get_performance_stats(self):
        stats = {
            "fps": {
                "current": self.fps_history[-1] if self.fps_history else 0,
                "average": np.mean(self.fps_history) if self.fps_history else 0,
                "min": np.min(self.fps_history) if self.fps_history else 0,
                "max": np.max(self.fps_history) if self.fps_history else 0,
                "std": np.std(self.fps_history) if self.fps_history else 0
            },
            "inference_time_ms": {
                "current": self.inference_times[-1] if self.inference_times else 0,
                "average": np.mean(self.inference_times) if self.inference_times else 0,
                "min": np.min(self.inference_times) if self.inference_times else 0,
                "max": np.max(self.inference_times) if self.inference_times else 0,
                "std": np.std(self.inference_times) if self.inference_times else 0
            },
            "preprocessing_time_ms": {
                "average": np.mean(self.preprocessing_times) if self.preprocessing_times else 0
            },
            "postprocessing_time_ms": {
                "average": np.mean(self.postprocessing_times) if self.postprocessing_times else 0
            },
            "system": {
                "cpu_usage_percent": np.mean(self.cpu_usage) if self.cpu_usage else 0,
                "gpu_memory_gb": self.gpu_memory_usage[-1] if self.gpu_memory_usage else 0,
                "gpu_utilization_percent": np.mean(self.gpu_utilization) if self.gpu_utilization else 0
            }
        }
        return stats

    def draw_performance_overlay(self, frame):
        stats = self.get_performance_stats()
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (400, 200), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        y_offset = 30
        cv2.putText(frame, f"FPS: {stats['fps']['current']:.1f} (Avg: {stats['fps']['average']:.1f})", 
                   (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        y_offset += 25
        cv2.putText(frame, f"Inference: {stats['inference_time_ms']['current']:.1f}ms", 
                   (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        y_offset += 25
        cv2.putText(frame, f"CPU: {stats['system']['cpu_usage_percent']:.1f}%", 
                   (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        y_offset += 25
        cv2.putText(frame, f"GPU Memory: {stats['system']['gpu_memory_gb']:.2f}GB", 
                   (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
        y_offset += 25
        cv2.putText(frame, f"Model: {self.model_path}", 
                   (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        y_offset += 25
        cv2.putText(frame, f"Device: {self.device.upper()}", 
                   (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        return frame

    def run_webcam_benchmark(self, camera_id=0, display=True, save_results=True):
        print(f"Starting webcam benchmark (Camera ID: {camera_id})")
        cap = cv2.VideoCapture(camera_id)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        if not cap.isOpened():
            print("Error: Could not open webcam")
            return
        self.monitoring = True
        monitor_thread = threading.Thread(target=self.monitor_system_resources)
        monitor_thread.daemon = True
        monitor_thread.start()
        self.warmup_gpu()
        print("Starting real-time benchmark... Press 'q' to quit, 's' to save results")
        frame_count = 0
        start_time = time.time()
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frame_start = time.time()
                img_tensor = self.preprocess_frame(frame)
                outputs = self.run_inference(img_tensor)
                annotated_frame = self.postprocess_results(outputs, frame)
                frame_time = time.time() - frame_start
                fps = 1.0 / frame_time if frame_time > 0 else 0
                self.fps_history.append(fps)
                if display:
                    display_frame = self.draw_performance_overlay(annotated_frame)
                    cv2.imshow('DETR Benchmark', display_frame)
                frame_count += 1
                if frame_count % 100 == 0:
                    stats = self.get_performance_stats()
                    print(f"\nFrame {frame_count}:")
                    print(f"  FPS: {stats['fps']['average']:.2f} ± {stats['fps']['std']:.2f}")
                    print(f"  Inference: {stats['inference_time_ms']['average']:.2f}ms ± {stats['inference_time_ms']['std']:.2f}ms")
                    print(f"  CPU Usage: {stats['system']['cpu_usage_percent']:.1f}%")
                    print(f"  GPU Memory: {stats['system']['gpu_memory_gb']:.2f}GB")
                if display:
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        break
                    elif key == ord('s'):
                        self.save_benchmark_results()
        except KeyboardInterrupt:
            print("\nBenchmark interrupted by user")
        finally:
            self.monitoring = False
            cap.release()
            if display:
                cv2.destroyAllWindows()
            total_time = time.time() - start_time
            print(f"\n=== Benchmark Results ===")
            print(f"Total frames processed: {frame_count}")
            print(f"Total time: {total_time:.2f}s")
            print(f"Average FPS: {frame_count / total_time:.2f}")
            final_stats = self.get_performance_stats()
            self.print_detailed_stats(final_stats)
            if save_results:
                self.save_benchmark_results()

    def run_video_benchmark(self, video_path, display=True, save_results=True):
        print(f"Starting video benchmark: {video_path}")
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Could not open video file: {video_path}")
            return
        self.monitoring = True
        monitor_thread = threading.Thread(target=self.monitor_system_resources)
        monitor_thread.daemon = True
        monitor_thread.start()
        self.warmup_gpu()
        frame_count = 0
        start_time = time.time()
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frame_start = time.time()
                img_tensor = self.preprocess_frame(frame)
                outputs = self.run_inference(img_tensor)
                annotated_frame = self.postprocess_results(outputs, frame)
                frame_time = time.time() - frame_start
                fps = 1.0 / frame_time if frame_time > 0 else 0
                self.fps_history.append(fps)
                if display:
                    display_frame = self.draw_performance_overlay(annotated_frame)
                    cv2.imshow('DETR Video Benchmark', display_frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                frame_count += 1
                if frame_count % 100 == 0:
                    stats = self.get_performance_stats()
                    print(f"Processed {frame_count} frames - FPS: {stats['fps']['average']:.2f}")
        except KeyboardInterrupt:
            print("\nBenchmark interrupted")
        finally:
            self.monitoring = False
            cap.release()
            if display:
                cv2.destroyAllWindows()
            total_time = time.time() - start_time
            final_stats = self.get_performance_stats()
            print(f"\n=== Video Benchmark Results ===")
            print(f"Frames processed: {frame_count}")
            print(f"Processing time: {total_time:.2f}s")
            self.print_detailed_stats(final_stats)
            if save_results:
                self.save_benchmark_results()

    def print_detailed_stats(self, stats):
        print(f"\n=== Detailed Performance Stats ===")
        print(f"FPS Statistics:")
        print(f"  Current: {stats['fps']['current']:.2f}")
        print(f"  Average: {stats['fps']['average']:.2f}")
        print(f"  Min: {stats['fps']['min']:.2f}")
        print(f"  Max: {stats['fps']['max']:.2f}")
        print(f"  Std Dev: {stats['fps']['std']:.2f}")
        print(f"\nInference Time (ms):")
        print(f"  Current: {stats['inference_time_ms']['current']:.2f}")
        print(f"  Average: {stats['inference_time_ms']['average']:.2f}")
        print(f"  Min: {stats['inference_time_ms']['min']:.2f}")
        print(f"  Max: {stats['inference_time_ms']['max']:.2f}")
        print(f"  Std Dev: {stats['inference_time_ms']['std']:.2f}")
        print(f"\nSystem Resources:")
        print(f"  CPU Usage: {stats['system']['cpu_usage_percent']:.1f}%")
        print(f"  GPU Memory: {stats['system']['gpu_memory_gb']:.2f}GB")
        print(f"  GPU Utilization: {stats['system']['gpu_utilization_percent']:.1f}%")
        print(f"\nProcessing Times:")
        print(f"  Preprocessing: {stats['preprocessing_time_ms']['average']:.2f}ms")
        print(f"  Postprocessing: {stats['postprocessing_time_ms']['average']:.2f}ms")

    def save_benchmark_results(self):
        stats = self.get_performance_stats()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"detr_benchmark_{timestamp}.json"
        benchmark_data = {
            "timestamp": timestamp,
            "model": self.model_path,
            "device": self.device,
            "system_info": {
                "gpu_name": torch.cuda.get_device_name() if torch.cuda.is_available() else "N/A",
                "cuda_version": torch.version.cuda,
                "pytorch_version": torch.__version__,
            },
            "performance_stats": stats,
            "raw_data": {
                "fps_history": list(self.fps_history),
                "inference_times": list(self.inference_times),
                "preprocessing_times": list(self.preprocessing_times),
                "postprocessing_times": list(self.postprocessing_times),
            }
        }
        try:
            with open(filename, 'w') as f:
                json.dump(benchmark_data, f, indent=2)
            print(f"\nBenchmark results saved to: {filename}")
        except Exception as e:
            print(f"Error saving results: {e}")

def main():
    parser = argparse.ArgumentParser(description='DETR GPU Benchmark')
    parser.add_argument('--model', type=str, required=True, help='Path to DETR model')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use (cuda/cpu, default: cuda)')
    parser.add_argument('--source', type=str, default='webcam', help='Source: webcam, video file path, or camera ID')
    parser.add_argument('--camera-id', type=int, default=0, help='Camera ID for webcam (default: 0)')
    parser.add_argument('--no-display', action='store_true', help='Run without display (headless mode)')
    parser.add_argument('--warmup', type=int, default=30, help='Number of warmup frames (default: 30)')
    parser.add_argument('--save-results', action='store_true', default=True, help='Save benchmark results to JSON file')
    args = parser.parse_args()
    print("=== DETR GPU Benchmark ===")
    print(f"Model: {args.model}")
    print(f"Device: {args.device}")
    print(f"Source: {args.source}")
    benchmark = DETRBenchmark(
        model_path=args.model,
        device=args.device,
        warmup_frames=args.warmup
    )
    if args.source == 'webcam' or args.source.isdigit():
        camera_id = int(args.source) if args.source.isdigit() else args.camera_id
        benchmark.run_webcam_benchmark(
            camera_id=camera_id,
            display=not args.no_display,
            save_results=args.save_results
        )
    else:
        benchmark.run_video_benchmark(
            video_path=args.source,
            display=not args.no_display,
            save_results=args.save_results
        )

if __name__ == "__main__":
    main()
