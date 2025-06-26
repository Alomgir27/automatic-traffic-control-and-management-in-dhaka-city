#!/usr/bin/env python3
"""
Dhaka Traffic Detection - Optimized Training Script
Perfect configuration for best results
"""

import os
import sys
from pathlib import Path
import time

class DhakaTrafficTrainer:
    def __init__(self):
        self.project_name = "dhaka_traffic_detection"
        self.timestamp = time.strftime("%Y%m%d_%H%M%S")
        
        # Optimal configuration for best results
        self.config = {
            'model': 'yolo11m',  # Best balance of speed and accuracy
            'epochs': 100,       # Optimal for this dataset size
            'batch_size': 16,    # Good for most systems
            'imgsz': 640,        # Standard YOLO input size
            'lr0': 0.01,         # Initial learning rate
            'device': 'auto',    # Auto-detect GPU/CPU
            'patience': 30,      # Early stopping
            'save_period': 10,   # Save every 10 epochs
            'workers': 4         # Data loading workers
        }
        
        # Dataset classes
        self.classes = [
            "ambulance", "auto rickshaw", "bicycle", "bus", "car", 
            "garbagevan", "human hauler", "minibus", "minivan", 
            "motorbike", "Pickup", "army vehicle", "policecar", 
            "rickshaw", "scooter", "suv", "taxi", 
            "three wheelers (CNG)", "truck", "van", "wheelbarrow"
        ]
    
    def check_environment(self):
        """Check if all dependencies are available"""
        try:
            import ultralytics
            import torch
            import cv2
            import numpy as np
            print("✅ All dependencies are available!")
            return True
        except ImportError as e:
            print(f"❌ Missing dependency: {e}")
            return False
    
    def check_dataset(self):
        """Verify dataset structure and files"""
        dataset_path = Path("archive/yolov")
        
        if not dataset_path.exists():
            print("❌ Dataset not found!")
            return False
        
        # Check required folders
        required_folders = [
            "images/train", "images/val", 
            "labels/train", "labels/val"
        ]
        
        for folder in required_folders:
            folder_path = dataset_path / folder
            if not folder_path.exists():
                print(f"❌ Missing folder: {folder}")
                return False
        
        # Count files
        train_images = len(list((dataset_path / "images/train").glob("*.jpg")))
        val_images = len(list((dataset_path / "images/val").glob("*.jpg")))
        train_labels = len(list((dataset_path / "labels/train").glob("*.txt")))
        val_labels = len(list((dataset_path / "labels/val").glob("*.txt")))
        
        print(f"📊 Dataset Summary:")
        print(f"   Training Images: {train_images}")
        print(f"   Validation Images: {val_images}")
        print(f"   Training Labels: {train_labels}")
        print(f"   Validation Labels: {val_labels}")
        
        if train_images > 0 and val_images > 0:
            print("✅ Dataset is ready!")
            return True
        else:
            print("❌ Dataset is incomplete!")
            return False
    
    def create_dataset_yaml(self):
        """Create optimized dataset configuration"""
        yaml_content = f"""# Dhaka Traffic Detection Dataset
# Optimized for YOLOv11 training

path: {os.path.abspath('archive/yolov')}
train: images/train
val: images/val

# Classes
nc: {len(self.classes)}
names: {self.classes}
"""
        
        with open("dataset.yaml", "w") as f:
            f.write(yaml_content)
        
        print("✅ Dataset configuration created!")
        return "dataset.yaml"
    
    def get_optimal_config(self):
        """Get optimal training configuration based on system"""
        try:
            import torch
            
            # Check if CUDA is available
            if torch.cuda.is_available():
                gpu_name = torch.cuda.get_device_name(0)
                gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
                
                print(f"🚀 GPU Detected: {gpu_name}")
                print(f"💾 GPU Memory: {gpu_memory:.1f} GB")
                
                # Adjust batch size based on GPU memory
                if gpu_memory >= 8:
                    self.config['batch_size'] = 32
                    self.config['epochs'] = 150
                elif gpu_memory >= 6:
                    self.config['batch_size'] = 24
                    self.config['epochs'] = 120
                elif gpu_memory >= 4:
                    self.config['batch_size'] = 16
                    self.config['epochs'] = 100
                else:
                    self.config['batch_size'] = 8
                    self.config['epochs'] = 80
                
                self.config['device'] = 0
                print(f"⚙️ Optimized for GPU: batch_size={self.config['batch_size']}, epochs={self.config['epochs']}")
            
            else:
                print("💻 CPU Training Mode")
                self.config['batch_size'] = 4
                self.config['epochs'] = 50
                self.config['device'] = 'cpu'
                self.config['workers'] = 2
                print("⚙️ Optimized for CPU: batch_size=4, epochs=50")
                
        except ImportError:
            print("⚠️ PyTorch not available, using default config")
        
        return self.config
    
    def train_model(self):
        """Start the training process"""
        try:
            from ultralytics import YOLO
            
            print(f"🚀 Starting {self.config['model']} Training")
            print("=" * 50)
            
            # Load model
            model = YOLO(f"{self.config['model']}.pt")
            
            # Start training
            results = model.train(
                data='dataset.yaml',
                epochs=self.config['epochs'],
                batch=self.config['batch_size'],
                imgsz=self.config['imgsz'],
                lr0=self.config['lr0'],
                device=self.config['device'],
                project='runs/detect',
                name=f'{self.project_name}_{self.timestamp}',
                exist_ok=True,
                patience=self.config['patience'],
                save_period=self.config['save_period'],
                workers=self.config['workers'],
                verbose=True,
                plots=True,
                save=True,
                cache=True,
                cos_lr=True,
                close_mosaic=10,
                amp=True,
                
                # Augmentation (optimized for traffic data)
                hsv_h=0.015,
                hsv_s=0.7,
                hsv_v=0.4,
                degrees=0.0,
                translate=0.1,
                scale=0.5,
                shear=0.0,
                perspective=0.0,
                flipud=0.0,
                fliplr=0.5,
                mosaic=1.0,
                mixup=0.0
            )
            
            print("✅ Training Completed Successfully!")
            
            # Save training info
            results_dir = f"runs/detect/{self.project_name}_{self.timestamp}"
            print(f"📁 Results saved in: {results_dir}")
            
            return {
                'success': True,
                'results_dir': results_dir,
                'model_path': f"{results_dir}/weights/best.pt"
            }
            
        except Exception as e:
            print(f"❌ Training failed: {e}")
            return {'success': False, 'error': str(e)}
    
    def run_complete_training(self):
        """Complete training pipeline"""
        print("🇧🇩 Dhaka Traffic Detection - Optimized Training")
        print("=" * 60)
        
        # Step 1: Check environment
        print("\n1️⃣ Checking Environment...")
        if not self.check_environment():
            print("💡 Please install dependencies first:")
            print("pip install ultralytics torch torchvision opencv-python")
            return False
        
        # Step 2: Check dataset
        print("\n2️⃣ Checking Dataset...")
        if not self.check_dataset():
            return False
        
        # Step 3: Create dataset config
        print("\n3️⃣ Creating Dataset Configuration...")
        self.create_dataset_yaml()
        
        # Step 4: Optimize configuration
        print("\n4️⃣ Optimizing Configuration...")
        self.get_optimal_config()
        
        # Step 5: Start training
        print("\n5️⃣ Starting Training...")
        results = self.train_model()
        
        if results['success']:
            print("\n🎉 Training Pipeline Completed Successfully!")
            print(f"📁 Model saved: {results['model_path']}")
            
            # Create thesis guidelines
            self.create_thesis_guidelines(results)
            
            return True
        else:
            print(f"\n❌ Training failed: {results['error']}")
            return False
    
    def create_thesis_guidelines(self, results):
        """Create thesis documentation guidelines"""
        guidelines = f"""
# Dhaka Traffic Detection - Thesis Guidelines

## Project Overview
- **Dataset**: Dhaka Traffic Detection with {len(self.classes)} vehicle classes
- **Model**: YOLOv11 ({self.config['model']})
- **Training Images**: ~2400 images
- **Validation Images**: ~600 images

## Training Configuration
- **Epochs**: {self.config['epochs']}
- **Batch Size**: {self.config['batch_size']}
- **Image Size**: {self.config['imgsz']}
- **Learning Rate**: {self.config['lr0']}
- **Device**: {self.config['device']}

## Results
- **mAP50**: {results.get('map50', 'N/A'):.3f}
- **mAP50-95**: {results.get('map50_95', 'N/A'):.3f}
- **Model Path**: {results.get('model_path', 'N/A')}

## Vehicle Classes Detected
{', '.join(self.classes)}

## Files for Thesis
1. **Trained Model**: `{results.get('model_path', 'best.pt')}`
2. **Training Results**: `{results.get('results_dir', 'runs/detect/latest')}`
3. **Dataset Config**: `dataset.yaml`
4. **Training Script**: `dhaka_traffic_trainer.py`

## Usage Instructions
```python
from ultralytics import YOLO

# Load trained model
model = YOLO('{results.get('model_path', 'best.pt')}')

# Predict on new image
results = model('path/to/image.jpg')

# Show results
results[0].show()
```

## Thesis Sections to Include
1. **Introduction**: Traffic monitoring in Dhaka city
2. **Literature Review**: YOLO object detection methods
3. **Methodology**: YOLOv11 architecture and training process
4. **Dataset**: Description of {len(self.classes)} vehicle classes
5. **Results**: Performance metrics (mAP, precision, recall)
6. **Conclusion**: Applications and future work

## Performance Metrics to Report
- Mean Average Precision (mAP) at IoU 0.5
- Mean Average Precision (mAP) at IoU 0.5:0.95
- Precision and Recall for each vehicle class
- Training time and computational requirements
- Real-time inference speed (FPS)

Generated on: {time.strftime("%Y-%m-%d %H:%M:%S")}
"""
        
        with open("THESIS_GUIDELINES.md", "w") as f:
            f.write(guidelines)
        
        print("📝 Thesis guidelines created: THESIS_GUIDELINES.md")

def main():
    trainer = DhakaTrafficTrainer()
    trainer.run_complete_training()

if __name__ == "__main__":
    main() 