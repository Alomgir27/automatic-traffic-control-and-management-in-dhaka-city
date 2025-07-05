# Dhaka Traffic Management System 🚦

**Machine Learning and IoT-enabled Traffic Management System: Prioritizing Emergency Vehicles and Reducing Congestion**

## 📋 Abstract

This project presents an IoT-enabled traffic management system that leverages YOLOv11 machine learning models for real-time detection and classification of vehicles, pedestrians, and emergency vehicles. The system dynamically adjusts traffic signals based on density and emergency vehicle presence, minimizing unnecessary wait times and preventing lane starvation.

## 👨‍🎓 Authors

- **Alomgir Hossain** - Computer Science and Engineering, SUST (2019331027)
- **M Saidur Rahman** - Computer Science and Engineering, SUST (2019331091)  
- **M Shahidur Rahman** - Computer Science and Engineering, SUST (rahmanms@sust.edu)

**Institution**: Shahjalal University of Science and Technology, Sylhet, Bangladesh

## 🎯 Key Features

- ✅ **Real-time Object Detection** using YOLOv11
- ✅ **Emergency Vehicle Prioritization** with automatic signal override
- ✅ **Adaptive Traffic Signal Control** based on traffic density
- ✅ **Multi-class Vehicle Classification** (21 categories)
- ✅ **Performance Visualization** with detailed analytics
- ✅ **IoT Integration** for connected traffic management

## 📊 Performance Metrics

- **Detection Accuracy**: 91%
- **Emergency Response Time**: < 5 seconds
- **Traffic Flow Improvement**: 25-30%
- **Congestion Reduction**: 20-25%
- **Dataset**: 3,784 images with 171,436 annotated objects

## 🚀 Quick Start

### Prerequisites

```bash
# Install Python 3.8 or higher
# Install pip package manager
```

### Installation

1. **Clone/Download** this repository
2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Ensure model file exists**:
   - Make sure `best.pt` (trained model) is in the root directory
   
4. **Add video files** (optional):
   - Place your traffic videos in the `videos/` folder
   - Supported formats: `.mp4`

### Running the System

#### Option 1: Complete Demo (Recommended)
```bash
python run_thesis_demo.py
```
This will run the complete thesis demonstration including:
- Dependency checking
- Video analysis (if videos available)
- Animation creation
- Report generation
- Results summary

#### Option 2: Individual Components

**Traffic Analysis Only:**
```bash
python traffic_management_system.py
```

**Animation Creation Only:**
```bash
python traffic_animation_creator.py
```

## 📁 Project Structure

```
thesis/
├── best.pt                           # Trained YOLOv11 model
├── dhaka_traffic_trainer.py          # Model training script
├── traffic_management_system.py      # Main traffic analysis system
├── traffic_animation_creator.py      # Animation creator
├── run_thesis_demo.py               # Complete demo runner
├── dataset.yaml                     # Dataset configuration
├── traffic_config.yaml              # Traffic system config
├── requirements.txt                 # Python dependencies
├── videos/                          # Input video files
│   ├── VID20240912134745.mp4
│   ├── VID20240912143934.mp4
│   ├── VID20240912144021.mp4
│   └── VID20223020234234.mp4
└── results/                         # Generated results
    └── analyzed_*.mp4               # Processed videos
```

## 🔧 System Components

### 1. Traffic Management System (`traffic_management_system.py`)

**Key Classes:**
- `TrafficSignalController`: Manages adaptive signal timing
- `TrafficAnalyzer`: Performs object detection and classification
- `TrafficManagementSystem`: Main system orchestrator

**Features:**
- Real-time video processing
- Emergency vehicle detection
- Adaptive signal control
- Performance metrics tracking

### 2. Animation Creator (`traffic_animation_creator.py`)

**Purpose**: Creates synthetic traffic scenarios for demonstration

**Features:**
- Realistic traffic simulation
- Emergency vehicle prioritization visualization
- Interactive traffic light control
- Statistical data collection

### 3. Demo Runner (`run_thesis_demo.py`)

**Purpose**: Comprehensive thesis demonstration

**Process:**
1. Dependency verification
2. File availability check
3. Traffic analysis execution
4. Animation generation
5. Report compilation
6. Results summarization

## 📈 Output Files

After running the system, you'll get:

- `traffic_analysis_results.png` - Performance visualization charts
- `traffic_management_report.json` - Detailed system metrics
- `traffic_simulation.mp4` - Animated traffic demonstration
- `emergency_detections.json` - Emergency vehicle detection log
- `THESIS_FINAL_REPORT.md` - Comprehensive thesis report
- `RESULTS_SUMMARY.txt` - Quick results overview
- `results/analyzed_*.mp4` - Processed video files

## 🎯 Vehicle Categories

The system can detect and classify 21 vehicle types:

**Emergency Vehicles:**
- Ambulance
- Police Car  
- Army Vehicle

**Regular Vehicles:**
- Auto Rickshaw, Bicycle, Bus, Car
- Garbage Van, Human Hauler, Minibus, Minivan
- Motorbike, Pickup, Rickshaw, Scooter
- SUV, Taxi, Three Wheeler (CNG)
- Truck, Van, Wheelbarrow

## 📊 Research Methodology

### Dataset Collection
- **Locations**: Shahbag, Polton, Motijheel, Science Lab, Panthapath, Bijoy Sarani, Gulistan
- **Images**: 3,784 high-resolution images
- **Annotations**: 171,436 objects manually annotated
- **Categories**: 3 main categories (Regular Vehicles, Emergency Vehicles, Pedestrians)

### Model Training
- **Architecture**: YOLOv11 (You Only Look Once)
- **Training Strategy**: Transfer learning with custom dataset
- **Optimization**: Adaptive learning rate, data augmentation
- **Validation**: Cross-validation with real-world scenarios

### System Integration
- **Real-time Processing**: Live CCTV feed analysis
- **IoT Connectivity**: Connected traffic management infrastructure
- **Emergency Priority**: Automatic signal override system
- **Performance Monitoring**: Continuous system evaluation

## 🔬 Technical Specifications

- **Input Resolution**: 640x640 pixels
- **Processing Speed**: Real-time (>30 FPS)
- **Model Size**: Optimized for edge deployment
- **Memory Usage**: GPU/CPU adaptive processing
- **Confidence Threshold**: 0.5 (adjustable)

## 🏆 Key Achievements

1. **91% Detection Accuracy** on custom Dhaka traffic dataset
2. **Emergency Response Time** reduced to < 5 seconds
3. **Traffic Flow Improvement** of 25-30%
4. **Congestion Reduction** of 20-25%
5. **Scalable Solution** for other developing urban areas

## 🚀 Future Enhancements

- **Weather Adaptation**: Algorithm adjustment for different weather conditions
- **Pedestrian Safety**: Enhanced pedestrian crossing management
- **Mobile Integration**: Real-time traffic updates for citizens
- **City-wide Coordination**: Multi-junction traffic optimization
- **AI Predictive Analysis**: Traffic pattern prediction and prevention

## 📞 Support & Contact

For questions, issues, or collaborations:

- **Email**: [Contact through SUST CSE Department]
- **Institution**: Shahjalal University of Science and Technology
- **Department**: Computer Science and Engineering

## 📄 License

This project is developed for academic research purposes at Shahjalal University of Science and Technology.

## 🙏 Acknowledgments

We thank:
- Shahjalal University of Science and Technology
- Department of Computer Science and Engineering
- All contributors and supporters of this research project
- The open-source community for tools and frameworks used

---

**📅 Project Timeline**: 2024  
**🎓 Academic Level**: Undergraduate Thesis  
**🏫 Institution**: SUST, Sylhet, Bangladesh  

*For the complete thesis document, refer to `thesis_paper.pdf`* 