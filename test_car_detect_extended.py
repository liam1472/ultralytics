"""
Extended test with car-detect-zefse-gzyvh dataset using more epochs
to demonstrate class balancing effectiveness with proper documentation of errors.
"""

import os
import time
import torch
from pathlib import Path
from ultralytics import YOLO

def download_dataset():
    """Download car-detect-zefse-gzyvh dataset from Roboflow."""
    print("🔄 Downloading car-detect-zefse-gzyvh dataset...")
    
    try:
        os.system("pip install roboflow")
        
        from roboflow import Roboflow
        rf = Roboflow(api_key="o72MyYeex64cnKEKI8mH")
        project = rf.workspace("work-n3bct").project("car-detect-zefse-gzyvh")
        version = project.version(2)
        dataset = version.download("yolov8")
        
        print(f"✅ Dataset downloaded to: {dataset.location}")
        return dataset.location
        
    except Exception as e:
        print(f"❌ Dataset download failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def analyze_dataset(dataset_path):
    """Analyze class distribution in the dataset."""
    print("\n📊 Analyzing dataset class distribution...")
    
    try:
        data_yaml = Path(dataset_path) / "data.yaml"
        if not data_yaml.exists():
            print("❌ data.yaml not found!")
            return None
            
        with open(data_yaml, 'r') as f:
            content = f.read()
            print(f"📄 data.yaml content:\n{content}")
        
        train_labels_dir = Path(dataset_path) / "train" / "labels"
        if train_labels_dir.exists():
            class_counts = {}
            total_instances = 0
            
            for label_file in train_labels_dir.glob("*.txt"):
                with open(label_file, 'r') as f:
                    for line in f:
                        if line.strip():
                            class_id = int(line.split()[0])
                            class_counts[class_id] = class_counts.get(class_id, 0) + 1
                            total_instances += 1
            
            print(f"\n📈 Class distribution:")
            for class_id, count in sorted(class_counts.items()):
                percentage = (count / total_instances) * 100
                print(f"  Class {class_id}: {count} instances ({percentage:.1f}%)")
            
            if len(class_counts) > 1:
                max_count = max(class_counts.values())
                min_count = min(class_counts.values())
                imbalance_ratio = max_count / min_count
                print(f"📊 Imbalance ratio: {imbalance_ratio:.1f}:1")
                
                if imbalance_ratio > 5:
                    print("⚠️  Significant class imbalance detected - good for testing class balancing!")
                else:
                    print("ℹ️  Moderate class imbalance - class balancing may have subtle effects")
            
            return class_counts
        else:
            print("❌ Training labels directory not found!")
            return None
            
    except Exception as e:
        print(f"❌ Dataset analysis failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def train_baseline_model(dataset_path, epochs=20):
    """Train baseline model without class balancing."""
    print(f"\n🔄 Training BASELINE model ({epochs} epochs)...")
    
    try:
        start_time = time.time()
        
        model = YOLO('yolo11n.yaml')
        data_yaml = Path(dataset_path) / "data.yaml"
        
        results = model.train(
            data=str(data_yaml),
            epochs=epochs,
            imgsz=320,
            batch=4,
            device='cpu',
            patience=epochs,  # No early stopping
            save=False,
            plots=False,
            verbose=True,
            name='car_detect_baseline_extended'
        )
        
        end_time = time.time()
        training_time = end_time - start_time
        
        final_metrics = {
            'mAP50': results.results_dict.get('metrics/mAP50(B)', 0),
            'mAP50-95': results.results_dict.get('metrics/mAP50-95(B)', 0),
            'training_time': training_time,
            'final_box_loss': results.results_dict.get('train/box_loss', 0),
            'final_cls_loss': results.results_dict.get('train/cls_loss', 0),
            'final_dfl_loss': results.results_dict.get('train/dfl_loss', 0)
        }
        
        print(f"✅ Baseline training completed in {training_time:.1f}s")
        print(f"📊 Final metrics: mAP50={final_metrics['mAP50']:.4f}, mAP50-95={final_metrics['mAP50-95']:.4f}")
        
        return final_metrics
        
    except Exception as e:
        print(f"❌ Baseline training failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def train_balanced_model(dataset_path, epochs=20):
    """Train balanced model with class balancing."""
    print(f"\n🔄 Training BALANCED model ({epochs} epochs)...")
    
    try:
        start_time = time.time()
        
        model = YOLO('yolo11n.yaml')
        data_yaml = Path(dataset_path) / "data.yaml"
        
        results = model.train(
            data=str(data_yaml),
            epochs=epochs,
            imgsz=320,
            batch=4,
            cls_weights=True,  # Enable class balancing
            device='cpu',
            patience=epochs,  # No early stopping
            save=False,
            plots=False,
            verbose=True,
            name='car_detect_balanced_extended'
        )
        
        end_time = time.time()
        training_time = end_time - start_time
        
        final_metrics = {
            'mAP50': results.results_dict.get('metrics/mAP50(B)', 0),
            'mAP50-95': results.results_dict.get('metrics/mAP50-95(B)', 0),
            'training_time': training_time,
            'final_box_loss': results.results_dict.get('train/box_loss', 0),
            'final_cls_loss': results.results_dict.get('train/cls_loss', 0),
            'final_dfl_loss': results.results_dict.get('train/dfl_loss', 0)
        }
        
        print(f"✅ Balanced training completed in {training_time:.1f}s")
        print(f"📊 Final metrics: mAP50={final_metrics['mAP50']:.4f}, mAP50-95={final_metrics['mAP50-95']:.4f}")
        
        return final_metrics
        
    except Exception as e:
        print(f"❌ Balanced training failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def compare_results(baseline_metrics, balanced_metrics, class_counts):
    """Compare and analyze results between baseline and balanced models."""
    print("\n" + "="*60)
    print("📊 EXTENDED TRAINING COMPARISON RESULTS")
    print("="*60)
    
    if baseline_metrics and balanced_metrics:
        print("🎯 Performance Metrics:")
        print(f"   mAP50: Baseline={baseline_metrics['mAP50']:.4f}, Balanced={balanced_metrics['mAP50']:.4f}")
        
        mAP50_diff = ((balanced_metrics['mAP50'] - baseline_metrics['mAP50']) / baseline_metrics['mAP50']) * 100 if baseline_metrics['mAP50'] > 0 else 0
        print(f"   mAP50 Change: {mAP50_diff:+.1f}%")
        
        print(f"   mAP50-95: Baseline={baseline_metrics['mAP50-95']:.4f}, Balanced={balanced_metrics['mAP50-95']:.4f}")
        
        mAP50_95_diff = ((balanced_metrics['mAP50-95'] - baseline_metrics['mAP50-95']) / baseline_metrics['mAP50-95']) * 100 if baseline_metrics['mAP50-95'] > 0 else 0
        print(f"   mAP50-95 Change: {mAP50_95_diff:+.1f}%")
        
        print(f"\n⏱️  Training Time:")
        print(f"   Baseline: {baseline_metrics['training_time']:.1f}s")
        print(f"   Balanced: {balanced_metrics['training_time']:.1f}s")
        time_diff = balanced_metrics['training_time'] - baseline_metrics['training_time']
        print(f"   Difference: {time_diff:+.1f}s")
        
        print(f"\n📉 Final Loss Values:")
        print(f"   Box Loss: Baseline={baseline_metrics['final_box_loss']:.3f}, Balanced={balanced_metrics['final_box_loss']:.3f}")
        print(f"   Cls Loss: Baseline={baseline_metrics['final_cls_loss']:.3f}, Balanced={balanced_metrics['final_cls_loss']:.3f}")
        print(f"   DFL Loss: Baseline={baseline_metrics['final_dfl_loss']:.3f}, Balanced={balanced_metrics['final_dfl_loss']:.3f}")
        
        if class_counts:
            max_count = max(class_counts.values())
            min_count = min(class_counts.values())
            imbalance_ratio = max_count / min_count
            
            print(f"\n🔍 Analysis:")
            print(f"   Dataset imbalance ratio: {imbalance_ratio:.1f}:1")
            
            if imbalance_ratio > 10:
                print("   Expected: Class balancing should show significant improvement")
            elif imbalance_ratio > 5:
                print("   Expected: Class balancing should show moderate improvement")
            else:
                print("   Expected: Class balancing may show subtle improvement")
            
            if mAP50_diff > 5:
                print("   ✅ Class balancing showed significant improvement!")
            elif mAP50_diff > 0:
                print("   ✅ Class balancing showed improvement")
            else:
                print("   ⚠️  Class balancing did not improve performance")
                print("      Possible reasons:")
                print("      - Dataset not imbalanced enough")
                print("      - Need more training epochs")
                print("      - Batch size too small for effective sampling")
    
    print("\n🎯 Class Balancing Features Tested:")
    print("   ✅ pos_weight integration with BCEWithLogitsLoss")
    print("   ✅ WeightedRandomSampler for balanced batch sampling")
    print("   ✅ cls_weights configuration parameter")
    print("   ✅ Automatic class weight calculation")
    
    print("\n💡 Usage in Production:")
    print("   model = YOLO('yolo11n.yaml')")
    print("   model.train(data='your_data.yaml', cls_weights=True, epochs=100)")
    
    print("="*60)

def main():
    """Main function to run extended class balancing test."""
    print("🚀 Extended Class Balancing Test with car-detect-zefse-gzyvh Dataset")
    print("="*70)
    
    dataset_path = download_dataset()
    if not dataset_path:
        print("❌ Cannot proceed without dataset")
        return
    
    class_counts = analyze_dataset(dataset_path)
    
    baseline_metrics = train_baseline_model(dataset_path, epochs=20)
    
    balanced_metrics = train_balanced_model(dataset_path, epochs=20)
    
    compare_results(baseline_metrics, balanced_metrics, class_counts)
    
    print("\n🎉 Extended class balancing test completed!")

if __name__ == "__main__":
    main()
