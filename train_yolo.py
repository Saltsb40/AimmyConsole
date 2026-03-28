"""
Train YOLOv8 — Train a YOLOv8 model on your labeled data and export to ONNX.

Usage:
    python train_yolo.py --data ../train --epochs 100 --model yolov8n.pt

This will:
    1. Create a YOLO dataset YAML config
    2. Train YOLOv8 on your labeled data
    3. Export the best model to ONNX format
    4. Copy the ONNX model to ../models/
"""

import argparse
import os
import yaml
import shutil
from pathlib import Path


def create_dataset_yaml(data_dir, num_classes, class_names=None):
    """Create a YOLO dataset YAML configuration."""
    data_dir = os.path.abspath(data_dir)

    if class_names is None:
        class_names = [f"class_{i}" for i in range(num_classes)]

    dataset_config = {
        "path": data_dir,
        "train": "images",
        "val": "images",  # Same as train for small datasets
        "nc": num_classes,
        "names": class_names,
    }

    yaml_path = os.path.join(data_dir, "dataset.yaml")
    with open(yaml_path, "w") as f:
        yaml.dump(dataset_config, f, default_flow_style=False)

    print(f"[Train] Dataset config saved to: {yaml_path}")
    return yaml_path


def on_train_epoch_end(trainer):
    """Callback to report new best fitness during training."""
    current_fitness = trainer.fitness
    if current_fitness is not None and current_fitness > getattr(trainer, 'prev_best_fitness', -1.0):
        # Use a distinctive highlight for the "Best Epoch" message
        print(f"\n" + "*"*60)
        print(f" [Train] NEW BEST EPOCH Fitness Reached: {current_fitness:.4f} (Epoch {trainer.epoch + 1})")
        print("*"*60 + "\n")
        trainer.prev_best_fitness = current_fitness


def main():
    parser = argparse.ArgumentParser(description="Train YOLOv8 and export to ONNX")
    parser.add_argument("--data", default="bin", help="Path to training data (with images/ and labels/ subdirs)")
    parser.add_argument("--model", default="yolov8m.pt", help="Base YOLO model (e.g., yolov8m.pt, yolov8s.pt)")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size for training (640 or 1024)")
    parser.add_argument("--batch", type=int, default=16, help="Batch size (Optimized for RTX 4060)")
    parser.add_argument("--classes", type=int, default=1, help="Number of classes")
    parser.add_argument("--class-names", nargs="+", default=None, help="Class names (e.g., player enemy)")
    parser.add_argument("--output", default="bin/models", help="Output directory for ONNX model")
    parser.add_argument("--device", default="0", help="Device: 0 for GPU, cpu for CPU")
    parser.add_argument("--lr0", type=float, default=0.01, help="Initial learning rate")
    parser.add_argument("--resume", action="store_true", help="Resume training from last.pt")
    parser.add_argument("--fine-tune", action="store_true", help="Fine-tune from best.pt (sets lr0=0.001)")
    parser.add_argument("--export-only", action="store_true", help="Skip training and export the best model from a previous run")
    args = parser.parse_args()

    # Verify training data exists
    images_dir = os.path.join(args.data, "images")
    labels_dir = os.path.join(args.data, "labels")

    if not os.path.exists(images_dir):
        print(f"[Train] Error: Images directory not found: {images_dir}")
        print(f"[Train] Capture screenshots and label them first!")
        return

    num_images = len([f for f in os.listdir(images_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    num_labels = len([f for f in os.listdir(labels_dir) if f.endswith('.txt')]) if os.path.exists(labels_dir) else 0

    print(f"[Train] Found {num_images} images, {num_labels} labels")

    if num_images == 0:
        print("[Train] No training images found! Capture and label some screenshots first.")
        return

    if num_labels == 0:
        print("[Train] No labels found! Run label_tool.py to annotate your screenshots.")
        return

    # Create dataset config
    yaml_path = create_dataset_yaml(args.data, args.classes, args.class_names)

    # Import and train
    try:
        from ultralytics import YOLO
    except ImportError:
        print("[Train] Error: ultralytics not installed!")
        print("[Train] Run: pip install -r requirements.txt")
        return

    print(f"\n[Train] Starting YOLOv8 training...")
    print(f"  Base model: {args.model}")
    print(f"  Epochs:     {args.epochs}")
    print(f"  Image size: {args.imgsz}")
    print(f"  Batch size: {args.batch}")
    print(f"  Classes:    {args.classes}")
    print(f"  Device:     {args.device}")

    # Handle export-only mode
    if args.export_only:
        print("\n[Export] Export-only mode active. Skipping training...")
        best_pt = os.path.join("training_runs", "aimbot", "weights", "best.pt")
        if not os.path.exists(best_pt):
            # Try finding it in the ultralytics standard path if our project-specific one fails
            best_pt = os.path.join("runs", "detect", "aimbot", "weights", "best.pt")
            
        if os.path.exists(best_pt):
            export_model(best_pt, args)
            return
        else:
            print(f"[Export] Error: No best.pt found at {best_pt}")
            return

    # --- INTERACTIVE TRAINING MODES ---
    # Search for checkpoints in common Ultralytics locations
    possible_paths = [
        os.path.join("runs", "aimbot_train", "weights", "last.pt"), # Current script default
        os.path.join("runs", "detect", "runs", "aimbot_train", "weights", "last.pt"), # Common nested path
        os.path.join("AuraVision_App", "scripts", "runs", "detect", "runs", "aimbot_train", "weights", "last.pt"), # User's specific nested path
        os.path.join("training_runs", "aimbot", "weights", "last.pt"), # New simplified path
    ]
    
    last_pt = None
    for p in possible_paths:
        if os.path.exists(p):
            last_pt = p
            break
            
    best_pt = None
    if last_pt:
        best_pt = last_pt.replace("last.pt", "best.pt")
    
    mode = "fresh"
    if not args.resume and not args.fine_tune:
        if last_pt:
            print("\n" + "="*55)
            print(f" [Train] PREVIOUS TRAINING DETECTED: {os.path.basename(os.path.dirname(os.path.dirname(last_pt)))}")
            print(" [Path] " + last_pt)
            print("="*55)
            print(" [1] Fresh Start (Delete previous training and start over)")
            print(" [2] Resume (Pick up exactly where you left off - last.pt)")
            print(" [3] Fine-Tune (Load best weights, reduce LR to 0.001, disable Mosaic)")
            print("="*55)
            choice = input("\n [Input] Select Mode [1-3]: ").strip()
            
            if choice == "2":
                mode = "resume"
                args.model = last_pt
            elif choice == "3":
                mode = "fine-tune"
                args.model = best_pt if (best_pt and os.path.exists(best_pt)) else last_pt
                args.lr0 = 0.001
                
                # Allow user to choose how many epochs to add
                print(f"\n [Fine-Tune] Current target is {args.epochs} total epochs.")
                add_epochs = input(" [Input] Enter TOTAL epochs to reach (e.g., 200): ").strip()
                if add_epochs.isdigit():
                    args.epochs = int(add_epochs)
                
                print(f"\n [Train] Fine-tuning enabled. Model: {args.model}")
                print(f" [Train] LR: {args.lr0}, Target Epochs: {args.epochs}, Mosaic: Disabled")
            else:
                # Fresh start with existing data
                print("\n" + "-"*30)
                print(" [Fresh Start] Select Model Size:")
                print(" [1] yolov8n.pt (Nano - Fastest / Lowest accuracy)")
                print(" [2] yolov8s.pt (Small - Fast / Good accuracy)")
                print(" [3] yolov8m.pt (Medium - Balanced - Recommended)")
                print(" [4] yolov8l.pt (Large - Slow / High accuracy)")
                print(" [5] yolov8x.pt (X-Large - Slowest / Max accuracy)")
                print("-"*30)
                m_choice = input(" [Input] Choice [1-5, Default 3]: ").strip()
                
                model_map = {"1": "yolov8n.pt", "2": "yolov8s.pt", "3": "yolov8m.pt", "4": "yolov8l.pt", "5": "yolov8x.pt"}
                args.model = model_map.get(m_choice, "yolov8m.pt")
                if args.model == "yolov8n.pt":
                    args.batch = 48
                    print(" [Train] Nano model detected. Batch size set to 48 for optimization.")
                
                print(f" [Train] Fresh start selected. Model: {args.model}. Wiping old progress...")
                
                # Allow user to choose how many epochs for fresh start
                print(f"\n [Train] Default is {args.epochs} epochs.")
                add_epochs = input(" [Input] Enter TOTAL epochs to train (e.g., 100): ").strip()
                if add_epochs.isdigit():
                    args.epochs = int(add_epochs)
                    
                # Attempt to delete the parent directory of weights
                try:
                    shutil.rmtree(os.path.dirname(os.path.dirname(last_pt)), ignore_errors=True)
                except:
                    pass
        else:
            # First time training!
            print("\n" + "="*55)
            print(" [Train] NO PREVIOUS TRAINING DETECTED")
            print("="*55)
            print(" [Fresh Start] Select Model Size:")
            print(" [1] yolov8n.pt (Nano - Fastest / Lowest accuracy)")
            print(" [2] yolov8s.pt (Small - Fast / Good accuracy)")
            print(" [3] yolov8m.pt (Medium - Balanced - Recommended)")
            print(" [4] yolov8l.pt (Large - Slow / High accuracy)")
            print(" [5] yolov8x.pt (X-Large - Slowest / Max accuracy)")
            print("-"*30)
            m_choice = input(" [Input] Choice [1-5, Default 3]: ").strip()
            
            model_map = {"1": "yolov8n.pt", "2": "yolov8s.pt", "3": "yolov8m.pt", "4": "yolov8l.pt", "5": "yolov8x.pt"}
            args.model = model_map.get(m_choice, "yolov8m.pt")
            if args.model == "yolov8n.pt":
                args.batch = 48
                print(" [Train] Nano model detected. Batch size set to 48 for optimization.")
            
            # Epoch selection for first time
            print(f"\n [Train] Default is {args.epochs} epochs.")
            add_epochs = input(" [Input] Enter TOTAL epochs to train (e.g., 100): ").strip()
            if add_epochs.isdigit():
                args.epochs = int(add_epochs)
            
            print(f" [Train] Starting new training with {args.model} for {args.epochs} epochs.")

    if args.resume: 
        mode = "resume"
        if last_pt: args.model = last_pt
    if args.fine_tune: 
        mode = "fine-tune"
        if best_pt and os.path.exists(best_pt): args.model = best_pt
        args.lr0 = 0.001

    # Ensure CUDA is available
    try:
        import torch
        if not torch.cuda.is_available() and args.device != 'cpu':
            print("\n[Train] WARNING: CUDA not detected! Training will be extremely slow on CPU.")
            print("[Train] Ensure torch with CUDA is installed (e.g., pip install torch --index-url https://download.pytorch.org/whl/cu121)")
            
            confirm = input("[Train] Proceed with CPU training? (y/N): ").lower().strip()
            if confirm != 'y':
                print("[Train] Training aborted to prevent CPU lag.")
                return
    except ImportError:
        pass

    if torch.cuda.is_available():
        print(f"[Train] CUDA Detected: {torch.cuda.get_device_name(0)}")

    # Force batch size to 48 if training with Nano model (yolov8n.pt)
    if "yolov8n.pt" in str(args.model).lower():
        args.batch = 48
        print(" [Train] Nano model detected. Ensuring batch size is 48.")

    model = YOLO(args.model)
    
    # Add callback for live fitness reporting
    model.add_callback("on_train_epoch_end", on_train_epoch_end)

    results = model.train(
        data=yaml_path,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=8,     # Optimized for 32GB RAM
        project="training_runs",
        name="aimbot",
        exist_ok=True,
        resume=(mode == "resume"),
        lr0=args.lr0,
        cache='disk',        # Cache to disk as requested to save RAM
        amp=True,           # Use Automatic Mixed Precision (Fastest on RTX 4060)
        # --- GEMINI OPTIMIZATIONS ---
        close_mosaic=0 if mode == "fine-tune" else 10,  # Disable Mosaic for fine-tuning stability
    )

    # Report Fitness Metrics
    print("\n" + "="*40)
    print("      TRAINING RESULTS & FITNESS")
    print("="*40)
    try:
        # YOLOv8 results.fitness is (mAP50 * 0.1 + mAP50-95 * 0.9)
        fitness = results.fitness
        print(f"[Train] Overall Fitness: {fitness:.4f}")
        
        if hasattr(results, 'results_dict'):
            metrics = results.results_dict
            print(f"[Train] mAP50:           {metrics.get('metrics/mAP50(B)', 0):.4f}")
            print(f"[Train] mAP50-95:        {metrics.get('metrics/mAP50-95(B)', 0):.4f}")
            print(f"[Train] Precision:       {metrics.get('metrics/precision(B)', 0):.4f}")
            print(f"[Train] Recall:          {metrics.get('metrics/recall(B)', 0):.4f}")
    except Exception as e:
        print(f"[Train] Could not extract detailed fitness: {e}")
    print("="*40 + "\n")

    # Export to ONNX
    best_model_path = os.path.join(results.save_dir, "weights", "best.pt")
    export_model(best_model_path, args)

    # Log results to persistent file
    log_path = os.path.join("runs", "fitness_log.txt")
    os.makedirs("runs", exist_ok=True)
    try:
        with open(log_path, "a") as f:
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            f.write(f"[{timestamp}] Model: {args.model}, Epochs: {args.epochs}, Fitness: {fitness:.4f}, mAP50: {metrics.get('metrics/mAP50(B)', 0):.4f}\n")
        print(f"[Train] Performance logged to: {log_path}")
    except Exception as e:
        print(f"[Train] Could not log results: {e}")


def export_model(best_model_path, args):
    """Encapsulated export logic to reuse for export-only mode."""
    try:
        from ultralytics import YOLO
    except ImportError:
        print("[Train] Error: ultralytics not installed!")
        return

    if os.path.exists(best_model_path):
        best_model = YOLO(best_model_path)
        print(f"\n[Export] Loading model from: {best_model_path}")
        onnx_path = best_model.export(format="onnx", imgsz=args.imgsz, opset=12)
        print(f"[Export] ONNX model saved: {onnx_path}")

        # Try to export TensorRT engine
        try:
            print("[Export] Exporting to TensorRT Engine (may take several minutes)...")
            engine_path = best_model.export(format="engine", imgsz=args.imgsz, dynamic=False, half=True)
            print(f"[Export] TensorRT Engine saved: {engine_path}")
        except Exception as e:
            print(f"[Export] TensorRT export failed (non-critical): {e}")
            engine_path = None

        # Determine destination filename
        param_count = sum(p.numel() for p in best_model.model.parameters())
        print(f"[Export] Model parameters: {param_count:,}")
        
        dst_name = "Apex.onnx"
        dst = os.path.join(args.output, dst_name)
        
        # Ensure output directory exists
        os.makedirs(args.output, exist_ok=True)
        
        shutil.copy2(onnx_path, dst)
        print(f"[Export] ONNX Model copied to: {dst}")

        if engine_path and os.path.exists(engine_path):
            engine_dst = os.path.join(args.output, "Apex.engine")
            shutil.copy2(engine_path, engine_dst)
            print(f"[Export] TensorRT Engine copied to: {engine_dst}")
            
        print(f"\n[Export] Done! Model is ready for use in Aimmy V2.")
    else:
        print(f"[Export] Error: weights not found at {best_model_path}")


if __name__ == "__main__":
    main()
