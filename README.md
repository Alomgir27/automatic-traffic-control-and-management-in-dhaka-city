# 🚗 Dhaka Traffic Detection with YOLOv11

একটি অপ্টিমাইজড YOLOv11 মডেল ব্যবহার করে ঢাকা শহরের ট্রাফিক ডিটেকশন প্রজেক্ট।

## 📊 Dataset Overview

- **Training Images**: 2,400+ images
- **Validation Images**: 600+ images  
- **Vehicle Classes**: 21 different types
- **Classes**: ambulance, auto rickshaw, bicycle, bus, car, garbagevan, human hauler, minibus, minivan, motorbike, pickup, army vehicle, policecar, rickshaw, scooter, suv, taxi, three wheelers (CNG), truck, van, wheelbarrow

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run Training
```bash
python dhaka_traffic_trainer.py
```

## 📁 Project Structure

```
thesis/
├── archive/
│   └── yolov/                  # Dataset folder
│       ├── images/
│       │   ├── train/          # Training images (2400+)
│       │   └── val/            # Validation images (600+)
│       └── labels/
│           ├── train/          # Training labels
│           └── val/            # Validation labels
├── dhaka_traffic_trainer.py    # Main training script
├── dataset.yaml               # Dataset configuration
├── requirements.txt           # Dependencies
├── traffic_config.yaml        # Training configuration
└── README.md                  # This file
```

## ⚙️ Configuration

The model automatically optimizes configuration based on your system:

- **GPU Available**: Higher batch size, more epochs
- **CPU Only**: Optimized for CPU training
- **Memory Detection**: Automatic batch size adjustment

## 🎯 Optimal Settings

### For GPU (8GB+)
- Batch Size: 32
- Epochs: 150
- Model: yolo11m

### For GPU (4-8GB)  
- Batch Size: 16
- Epochs: 100
- Model: yolo11m

### For CPU
- Batch Size: 4
- Epochs: 50
- Model: yolo11n

## 📈 Expected Results

- **mAP50**: 0.85+ (Expected)
- **mAP50-95**: 0.65+ (Expected)
- **Training Time**: 2-4 hours (GPU), 8-12 hours (CPU)

## 🎓 For Thesis

After training completion, you'll get:

1. **Trained Model**: `runs/detect/dhaka_traffic_detection_*/weights/best.pt`
2. **Training Plots**: Precision, Recall, mAP curves
3. **Validation Results**: Per-class performance metrics
4. **Confusion Matrix**: Class-wise detection accuracy

## 📋 Usage After Training

```python
from ultralytics import YOLO

# Load your trained model
model = YOLO('runs/detect/dhaka_traffic_detection_*/weights/best.pt')

# Predict on new image
results = model('path/to/your/image.jpg')

# Display results
results[0].show()

# Save results
results[0].save('output_image.jpg')
```

## 🔧 Troubleshooting

### Common Issues:

1. **CUDA Out of Memory**: Reduce batch size in config
2. **Slow Training**: Use GPU or reduce image size
3. **Low Accuracy**: Increase epochs or use larger model

### Dependencies Issues:
```bash
# For CPU-only installation
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# For GPU installation (if CUDA available)  
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

## 📄 Citation

If you use this dataset/code in your research, please cite:

```
@misc{dhaka_traffic_detection_2024,
  title={Dhaka Traffic Detection using YOLOv11},
  author={Your Name},
  year={2024},
  howpublished={Thesis Project}
}
```

## 📞 Support

For any issues or questions, check:
1. Requirements are properly installed
2. Dataset is in correct format
3. System has enough memory
4. GPU drivers are updated (if using GPU)

---
**Note**: This project is optimized for thesis research on traffic detection in Dhaka city. 

https://www.kaggle.com/code/taha07/dhaka-ai-traffic-detection-using-yolov8/notebook