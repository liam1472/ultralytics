#!/usr/bin/env python3
"""
Helicopter Dataset Optimizer - Advanced configuration for large datasets
Handles ThreadPool optimization and mixed dataset format issues.
"""

import os
import threading
import time
from pathlib import Path
from ultralytics import YOLO

class HelicopterDatasetOptimizer:
    """Optimizer for training YOLO on large helicopter detection datasets."""
    
    def __init__(self, data_path, output_dir="helicopter_training_results"):
        self.data_path = data_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        self.base_config = {
            'cls_weights': True,        # Enable class balancing
            'cache': False,             # Critical: disable for large datasets
            'workers': 2,               # Reduced to prevent ThreadPool hang
            'verbose': True,
            'save': True,
            'plots': True,
            'task': 'detect',           # Force detection mode for mixed datasets
            'patience': 50,
            'save_period': 10,
        }
    
    def optimize_for_system(self, available_ram_gb=8, gpu_memory_gb=12):
        """Optimize configuration based on system resources."""
        config = self.base_config.copy()
        
        if gpu_memory_gb >= 12:
            config['batch'] = 8
            config['imgsz'] = 416
        elif gpu_memory_gb >= 8:
            config['batch'] = 6
            config['imgsz'] = 384
        else:
            config['batch'] = 4
            config['imgsz'] = 320
            config['device'] = 'cpu'  # Fallback to CPU
        
        if available_ram_gb >= 16:
            config['workers'] = 4
        elif available_ram_gb >= 8:
            config['workers'] = 2
        else:
            config['workers'] = 1
        
        return config
    
    def test_configuration(self, config, test_name="config_test"):
        """Test a configuration with minimal parameters."""
        test_config = config.copy()
        test_config.update({
            'epochs': 2,
            'fraction': 0.005,  # 0.5% of dataset
            'name': test_name,
            'save': False,
            'plots': False,
        })
        
        try:
            model = YOLO('yolo11n.yaml')
            start_time = time.time()
            
            results = model.train(data=self.data_path, **test_config)
            
            duration = time.time() - start_time
            print(f"✅ Configuration test '{test_name}' passed in {duration:.1f}s")
            return True, duration
            
        except Exception as e:
            print(f"❌ Configuration test '{test_name}' failed: {e}")
            return False, 0
    
    def find_optimal_fraction(self, config, max_fraction=0.1):
        """Find the largest dataset fraction that works reliably."""
        fractions = [0.005, 0.01, 0.02, 0.05, 0.1, 0.2]
        fractions = [f for f in fractions if f <= max_fraction]
        
        optimal_fraction = 0.005  # Safe default
        
        for fraction in fractions:
            test_config = config.copy()
            test_config['fraction'] = fraction
            
            print(f"Testing fraction {fraction} ({int(193526 * fraction)} images)...")
            
            success, duration = self.test_configuration(
                test_config, 
                f"fraction_test_{fraction}"
            )
            
            if success:
                optimal_fraction = fraction
                print(f"✅ Fraction {fraction} works (duration: {duration:.1f}s)")
            else:
                print(f"❌ Fraction {fraction} failed - stopping at {optimal_fraction}")
                break
        
        return optimal_fraction
    
    def create_production_config(self, fraction=None, epochs=50):
        """Create optimized production configuration."""
        config = self.optimize_for_system()
        
        if fraction is None:
            print("🔍 Finding optimal dataset fraction...")
            fraction = self.find_optimal_fraction(config)
        
        config.update({
            'data': self.data_path,
            'epochs': epochs,
            'fraction': fraction,
            'name': f'helicopter_production_f{fraction}_e{epochs}',
        })
        
        return config
    
    def run_production_training(self, config=None):
        """Run production training with monitoring."""
        if config is None:
            config = self.create_production_config()
        
        print(f"\n🚀 STARTING PRODUCTION TRAINING")
        print("=" * 50)
        print(f"Configuration:")
        for key, value in config.items():
            print(f"  {key}: {value}")
        print("=" * 50)
        
        monitoring = True
        def monitor_resources():
            import psutil
            while monitoring:
                cpu = psutil.cpu_percent(interval=5)
                memory = psutil.virtual_memory()
                print(f"📊 Resources: CPU {cpu:.1f}%, RAM {memory.percent:.1f}%")
                time.sleep(30)
        
        monitor_thread = threading.Thread(target=monitor_resources, daemon=True)
        monitor_thread.start()
        
        try:
            model = YOLO('yolo11n.yaml')
            start_time = time.time()
            
            results = model.train(**config)
            
            duration = time.time() - start_time
            monitoring = False  # Stop monitoring
            
            print(f"\n🎉 TRAINING COMPLETED!")
            print(f"Duration: {duration/3600:.1f} hours")
            
            summary_file = self.output_dir / f"training_summary_{config['name']}.txt"
            with open(summary_file, 'w') as f:
                f.write(f"Training Summary\n")
                f.write(f"================\n")
                f.write(f"Duration: {duration/3600:.1f} hours\n")
                f.write(f"Configuration: {config}\n")
                if hasattr(results, 'results_dict'):
                    f.write(f"Results: {results.results_dict}\n")
            
            print(f"📄 Summary saved to: {summary_file}")
            
            return results
            
        except Exception as e:
            monitoring = False
            print(f"❌ Production training failed: {e}")
            raise
    
    def handle_mixed_dataset_warning(self):
        """Provide guidance for mixed dataset format issues."""
        print(f"\n⚠️  MIXED DATASET FORMAT DETECTED")
        print("=" * 50)
        print("Your dataset has mixed detect-segment format:")
        print("- len(segments) = 34")
        print("- len(boxes) = 218,528")
        print("")
        print("Solutions:")
        print("1. Force detection-only mode (recommended):")
        print("   config['task'] = 'detect'")
        print("")
        print("2. Clean dataset to pure detection format:")
        print("   - Remove segment annotations")
        print("   - Keep only bounding box labels")
        print("")
        print("3. Convert to pure segmentation format:")
        print("   - Add segment annotations for all objects")
        print("   - Ensure segments match bounding boxes")

def main():
    """Main function to demonstrate usage."""
    print("🚁 HELICOPTER DATASET OPTIMIZER")
    print("=" * 60)
    
    data_path = "C:/Users/mexil/PyCharmMiscProject/Helicopter-Detection-3-1/data.yaml"
    optimizer = HelicopterDatasetOptimizer(data_path)
    
    optimizer.handle_mixed_dataset_warning()
    
    print(f"\n🔧 Creating optimized configuration...")
    config = optimizer.create_production_config(fraction=0.01, epochs=10)
    
    print(f"\n🧪 Testing configuration...")
    success, duration = optimizer.test_configuration(config, "final_test")
    
    if success:
        print(f"\n✅ Configuration validated!")
        print(f"Ready for production training with:")
        print(f"  - Dataset fraction: {config['fraction']} (~{int(193526 * config['fraction'])} images)")
        print(f"  - Batch size: {config['batch']}")
        print(f"  - Image size: {config['imgsz']}")
        print(f"  - Workers: {config['workers']}")
        print(f"  - cls_weights: {config['cls_weights']}")
        
        user_input = input(f"\nRun production training? (y/n): ").lower().strip()
        if user_input == 'y':
            optimizer.run_production_training(config)
    else:
        print(f"\n❌ Configuration needs further optimization")
        print(f"Try reducing fraction or batch size")

if __name__ == "__main__":
    main()
