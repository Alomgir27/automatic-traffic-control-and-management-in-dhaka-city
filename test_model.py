#!/usr/bin/env python3
"""
Dhaka Traffic Detection - Model Testing Script
Use this script to test your trained model on new images
"""

import os
import sys
from pathlib import Path

class DhakaTrafficTester:
    def __init__(self, model_path=None):
        """
        Initialize the tester with trained model
        
        Args:
            model_path: Path to the trained model (.pt file)
        """
        self.model_path = model_path
        self.model = None
        
        # Vehicle classes
        self.classes = [
            "ambulance", "auto rickshaw", "bicycle", "bus", "car", 
            "garbagevan", "human hauler", "minibus", "minivan", 
            "motorbike", "Pickup", "army vehicle", "policecar", 
            "rickshaw", "scooter", "suv", "taxi", 
            "three wheelers (CNG)", "truck", "van", "wheelbarrow"
        ]
    
    def find_best_model(self):
        """Automatically find the best trained model"""
        runs_dir = Path("runs/detect")
        
        if not runs_dir.exists():
            print("❌ No trained models found!")
            print("Please train the model first using: python dhaka_traffic_trainer.py")
            return None
        
        # Find all training runs
        training_runs = [d for d in runs_dir.iterdir() if d.is_dir() and "dhaka_traffic" in d.name]
        
        if not training_runs:
            print("❌ No Dhaka traffic models found!")
            return None
        
        # Get the latest training run
        latest_run = max(training_runs, key=lambda x: x.stat().st_mtime)
        best_model = latest_run / "weights" / "best.pt"
        
        if best_model.exists():
            print(f"✅ Found trained model: {best_model}")
            return str(best_model)
        else:
            print(f"❌ Model weights not found in {latest_run}")
            return None
    
    def load_model(self):
        """Load the trained model"""
        try:
            from ultralytics import YOLO
            
            # If no model path provided, find the best one
            if not self.model_path:
                self.model_path = self.find_best_model()
                
            if not self.model_path:
                return False
            
            # Load the model
            self.model = YOLO(self.model_path)
            print(f"✅ Model loaded successfully!")
            return True
            
        except ImportError:
            print("❌ Ultralytics not installed!")
            print("Install with: pip install ultralytics")
            return False
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            return False
    
    def predict_image(self, image_path, save_result=True, show_result=True):
        """
        Predict vehicles in an image
        
        Args:
            image_path: Path to the image file
            save_result: Whether to save the result image
            show_result: Whether to display the result
        """
        try:
            # Check if image exists
            if not os.path.exists(image_path):
                print(f"❌ Image not found: {image_path}")
                return None
            
            # Make prediction
            print(f"🔍 Analyzing image: {image_path}")
            results = self.model(image_path)
            
            # Get detection info
            detections = results[0]
            
            # Print detection summary
            if len(detections.boxes) > 0:
                print(f"✅ Found {len(detections.boxes)} vehicles:")
                
                for i, box in enumerate(detections.boxes):
                    class_id = int(box.cls)
                    confidence = float(box.conf)
                    class_name = self.classes[class_id]
                    
                    print(f"   {i+1}. {class_name} (Confidence: {confidence:.2f})")
                
                # Save result if requested
                if save_result:
                    output_path = f"result_{Path(image_path).stem}.jpg"
                    detections.save(output_path)
                    print(f"💾 Result saved: {output_path}")
                
                # Show result if requested
                if show_result:
                    print("🖼️ Displaying result...")
                    detections.show()
                    
            else:
                print("❌ No vehicles detected in the image")
            
            return results
            
        except Exception as e:
            print(f"❌ Error during prediction: {e}")
            return None
    
    def test_sample_images(self):
        """Test on sample images from validation set"""
        val_images_dir = Path("archive/yolov/images/val")
        
        if not val_images_dir.exists():
            print("❌ Validation images not found!")
            return
        
        # Get first 5 images for testing
        sample_images = list(val_images_dir.glob("*.jpg"))[:5]
        
        if not sample_images:
            print("❌ No sample images found!")
            return
        
        print(f"🧪 Testing on {len(sample_images)} sample images...")
        
        for i, image_path in enumerate(sample_images, 1):
            print(f"\n--- Testing Image {i}/{len(sample_images)} ---")
            self.predict_image(str(image_path), save_result=True, show_result=False)
    
    def run_interactive_test(self):
        """Interactive testing mode"""
        print("🎯 Interactive Testing Mode")
        print("Enter image paths to test, or 'sample' to test sample images, or 'quit' to exit")
        
        while True:
            user_input = input("\nEnter image path (or 'sample'/'quit'): ").strip()
            
            if user_input.lower() == 'quit':
                print("👋 Goodbye!")
                break
            elif user_input.lower() == 'sample':
                self.test_sample_images()
            elif user_input:
                self.predict_image(user_input)
            else:
                print("❌ Please enter a valid image path")

def main():
    print("🚗 Dhaka Traffic Detection - Model Testing")
    print("=" * 50)
    
    # Initialize tester
    tester = DhakaTrafficTester()
    
    # Load model
    if not tester.load_model():
        print("❌ Failed to load model!")
        return
    
    # Check command line arguments
    if len(sys.argv) > 1:
        # Test specific image from command line
        image_path = sys.argv[1]
        tester.predict_image(image_path)
    else:
        # Interactive mode
        tester.run_interactive_test()

if __name__ == "__main__":
    main() 