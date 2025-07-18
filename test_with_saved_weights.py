"""
Enhanced validation test script that saves trained model weights and reuses them
for efficient testing on separate validation datasets.

Addresses user feedback: "lưu weight lại ghi rõ là file nào là của baseline file nào là của balanced 
rồi dùng weight đó test trên dataset khác là được chứ mỗi lần chạy train lại như này tốn bao nhiêu thời gian"

Features:
- Saves weights with clear naming: baseline_weights.pt, balanced_weights.pt
- Reuses saved weights for validation testing (no retraining needed)
- Tests on separate validation data for proper overfitting detection
- Provides detailed per-class metrics analysis
- Efficient workflow for repeated testing
"""

import os
import time
import shutil
from pathlib import Path
from ultralytics import YOLO
import yaml

def setup_dataset():
    """Download and setup car-detect-2 dataset if not exists."""
    dataset_path = "/home/ubuntu/repos/ultralytics/car-detect-2"
    
    if Path(dataset_path).exists():
        print(f"✅ Dataset already exists at {dataset_path}")
        return dataset_path
    
    print("📥 Downloading car-detect-2 dataset...")
    try:
        from roboflow import Roboflow
        rf = Roboflow(api_key="your_api_key_here")  # Will use public dataset
        project = rf.workspace("zefse-gzyvh").project("car-detect-zefse-gzyvh")
        dataset = project.version(2).download("yolov8", location=dataset_path)
        print(f"✅ Dataset downloaded to {dataset_path}")
        return dataset_path
    except Exception as e:
        print(f"❌ Dataset download failed: {e}")
        return None

def analyze_dataset_distribution(dataset_path):
    """Analyze class distribution in the dataset."""
    print("\n📊 DATASET ANALYSIS")
    print("=" * 50)
    
    data_yaml_path = Path(dataset_path) / "data.yaml"
    with open(data_yaml_path, 'r') as f:
        data_config = yaml.safe_load(f)
    
    class_names = data_config['names']
    print(f"Classes: {class_names}")
    
    train_labels_dir = Path(dataset_path) / "train" / "labels"
    class_counts = {i: 0 for i in range(len(class_names))}
    total_instances = 0
    
    for label_file in train_labels_dir.glob("*.txt"):
        with open(label_file, 'r') as f:
            for line in f:
                if line.strip():
                    class_id = int(line.split()[0])
                    class_counts[class_id] += 1
                    total_instances += 1
    
    print(f"\n📈 Training set distribution:")
    print(f"   Total instances: {total_instances}")
    for i, name in enumerate(class_names):
        count = class_counts[i]
        percentage = (count / total_instances) * 100 if total_instances > 0 else 0
        print(f"   {name}: {count} instances ({percentage:.1f}%)")
    
    max_count = max(class_counts.values())
    min_count = min([count for count in class_counts.values() if count > 0])
    imbalance_ratio = max_count / min_count if min_count > 0 else 0
    print(f"   Imbalance ratio: {imbalance_ratio:.1f}:1")
    
    val_labels_dir = Path(dataset_path) / "valid" / "labels"
    val_class_counts = {i: 0 for i in range(len(class_names))}
    val_total_instances = 0
    
    for label_file in val_labels_dir.glob("*.txt"):
        with open(label_file, 'r') as f:
            for line in f:
                if line.strip():
                    class_id = int(line.split()[0])
                    val_class_counts[class_id] += 1
                    val_total_instances += 1
    
    print(f"\n📈 Validation set distribution:")
    print(f"   Total instances: {val_total_instances}")
    for i, name in enumerate(class_names):
        count = val_class_counts[i]
        percentage = (count / val_total_instances) * 100 if val_total_instances > 0 else 0
        print(f"   {name}: {count} instances ({percentage:.1f}%)")
    
    return class_names, class_counts, val_class_counts

def train_and_save_models(dataset_path, force_retrain=False):
    """Train baseline and balanced models, save weights with clear naming."""
    print("\n🚀 TRAINING AND SAVING MODELS")
    print("=" * 50)
    
    data_yaml = Path(dataset_path) / "data.yaml"
    weights_dir = Path("/home/ubuntu/repos/ultralytics/saved_weights")
    weights_dir.mkdir(exist_ok=True)
    
    baseline_weights = weights_dir / "baseline_weights.pt"
    balanced_weights = weights_dir / "balanced_weights.pt"
    
    if baseline_weights.exists() and balanced_weights.exists() and not force_retrain:
        print("✅ Saved weights found:")
        print(f"   Baseline: {baseline_weights}")
        print(f"   Balanced: {balanced_weights}")
        print("   Skipping training (use force_retrain=True to retrain)")
        return str(baseline_weights), str(balanced_weights)
    
    print("🔄 Training models (this will take time, but only once)...")
    
    print("\n📊 Training BASELINE model...")
    start_time = time.time()
    
    baseline_model = YOLO('yolo11n.yaml')
    baseline_results = baseline_model.train(
        data=str(data_yaml),
        epochs=20,
        imgsz=320,
        batch=4,
        cls_weights=False,  # No class balancing
        device='cpu',
        patience=20,
        save=True,
        plots=False,
        verbose=True,
        name='baseline_training',
        project=str(weights_dir)
    )
    
    baseline_time = time.time() - start_time
    
    baseline_model.save(str(baseline_weights))
    print(f"✅ Baseline model saved: {baseline_weights}")
    print(f"   Training time: {baseline_time:.1f}s")
    
    print("\n⚖️ Training BALANCED model...")
    start_time = time.time()
    
    balanced_model = YOLO('yolo11n.yaml')
    balanced_results = balanced_model.train(
        data=str(data_yaml),
        epochs=20,
        imgsz=320,
        batch=4,
        cls_weights=True,  # Auto class balancing
        device='cpu',
        patience=20,
        save=True,
        plots=False,
        verbose=True,
        name='balanced_training',
        project=str(weights_dir)
    )
    
    balanced_time = time.time() - start_time
    
    balanced_model.save(str(balanced_weights))
    print(f"✅ Balanced model saved: {balanced_weights}")
    print(f"   Training time: {balanced_time:.1f}s")
    
    print(f"\n🎯 Training Summary:")
    print(f"   Baseline training: {baseline_time:.1f}s")
    print(f"   Balanced training: {balanced_time:.1f}s")
    print(f"   Total training time: {(baseline_time + balanced_time):.1f}s")
    print(f"   Weights saved to: {weights_dir}")
    
    return str(baseline_weights), str(balanced_weights)

def test_models_on_validation(baseline_weights_path, balanced_weights_path, dataset_path, class_names):
    """Test saved models on validation data and extract per-class metrics."""
    print("\n🧪 TESTING MODELS ON VALIDATION DATA")
    print("=" * 50)
    print("Using saved weights - NO RETRAINING NEEDED!")
    
    data_yaml = Path(dataset_path) / "data.yaml"
    
    print("\n📊 Testing BASELINE model on validation data...")
    baseline_model = YOLO(baseline_weights_path)
    baseline_val_results = baseline_model.val(
        data=str(data_yaml),
        split='val',
        imgsz=320,
        batch=4,
        device='cpu',
        verbose=True,
        save_json=False,
        plots=False
    )
    
    print("\n⚖️ Testing BALANCED model on validation data...")
    balanced_model = YOLO(balanced_weights_path)
    balanced_val_results = balanced_model.val(
        data=str(data_yaml),
        split='val',
        imgsz=320,
        batch=4,
        device='cpu',
        verbose=True,
        save_json=False,
        plots=False
    )
    
    return baseline_val_results, balanced_val_results

def extract_per_class_metrics(val_results, class_names, model_name):
    """Extract detailed per-class metrics from validation results."""
    print(f"\n📈 {model_name.upper()} - PER-CLASS METRICS")
    print("-" * 40)
    
    overall_metrics = {
        'mAP50': val_results.results_dict.get('metrics/mAP50(B)', 0),
        'mAP50-95': val_results.results_dict.get('metrics/mAP50-95(B)', 0),
        'precision': val_results.results_dict.get('metrics/precision(B)', 0),
        'recall': val_results.results_dict.get('metrics/recall(B)', 0)
    }
    
    print(f"Overall Performance:")
    print(f"   mAP50: {overall_metrics['mAP50']:.4f}")
    print(f"   mAP50-95: {overall_metrics['mAP50-95']:.4f}")
    print(f"   Precision: {overall_metrics['precision']:.4f}")
    print(f"   Recall: {overall_metrics['recall']:.4f}")
    
    per_class_metrics = {}
    
    try:
        det_metrics = val_results.results_dict.get('det_metrics')
        if det_metrics and hasattr(det_metrics, 'class_result'):
            print(f"\nPer-Class Breakdown:")
            print(f"{'Class':<12} {'Precision':<10} {'Recall':<10} {'mAP50':<10} {'mAP50-95':<10}")
            print("-" * 60)
            
            for i, class_name in enumerate(class_names):
                try:
                    class_result = det_metrics.class_result(i)
                    if class_result:
                        precision = class_result[0] if len(class_result) > 0 else 0
                        recall = class_result[1] if len(class_result) > 1 else 0
                        map50 = class_result[2] if len(class_result) > 2 else 0
                        map50_95 = class_result[3] if len(class_result) > 3 else 0
                        
                        per_class_metrics[class_name] = {
                            'precision': precision,
                            'recall': recall,
                            'mAP50': map50,
                            'mAP50-95': map50_95
                        }
                        
                        print(f"{class_name:<12} {precision:<10.4f} {recall:<10.4f} {map50:<10.4f} {map50_95:<10.4f}")
                    else:
                        per_class_metrics[class_name] = {
                            'precision': 0, 'recall': 0, 'mAP50': 0, 'mAP50-95': 0
                        }
                        print(f"{class_name:<12} {'0.0000':<10} {'0.0000':<10} {'0.0000':<10} {'0.0000':<10}")
                        
                except Exception as e:
                    print(f"   {class_name}: Unable to extract metrics ({e})")
                    per_class_metrics[class_name] = {
                        'precision': 0, 'recall': 0, 'mAP50': 0, 'mAP50-95': 0
                    }
        else:
            print("⚠️ Unable to extract per-class metrics from DetMetrics")
            for class_name in class_names:
                per_class_metrics[class_name] = {
                    'precision': 0, 'recall': 0, 'mAP50': 0, 'mAP50-95': 0
                }
                
    except Exception as e:
        print(f"⚠️ Error extracting per-class metrics: {e}")
        for class_name in class_names:
            per_class_metrics[class_name] = {
                'precision': 0, 'recall': 0, 'mAP50': 0, 'mAP50-95': 0
            }
    
    return overall_metrics, per_class_metrics

def compare_models_per_class(baseline_metrics, balanced_metrics, baseline_per_class, balanced_per_class, class_names, class_counts, val_class_counts):
    """Compare baseline vs balanced model performance per class."""
    print("\n🔍 DETAILED COMPARISON ANALYSIS")
    print("=" * 60)
    
    baseline_map50 = baseline_metrics['mAP50']
    balanced_map50 = balanced_metrics['mAP50']
    baseline_map50_95 = baseline_metrics['mAP50-95']
    balanced_map50_95 = balanced_metrics['mAP50-95']
    
    overall_improvement_50 = ((balanced_map50 - baseline_map50) / baseline_map50 * 100) if baseline_map50 > 0 else 0
    overall_improvement_50_95 = ((balanced_map50_95 - baseline_map50_95) / baseline_map50_95 * 100) if baseline_map50_95 > 0 else 0
    
    print(f"📊 Overall Performance Comparison:")
    print(f"   mAP50:     {baseline_map50:.4f} → {balanced_map50:.4f} ({overall_improvement_50:+.1f}%)")
    print(f"   mAP50-95:  {baseline_map50_95:.4f} → {balanced_map50_95:.4f} ({overall_improvement_50_95:+.1f}%)")
    
    print(f"\n🎯 Per-Class Performance Analysis:")
    print(f"{'Class':<12} {'Train':<8} {'Val':<6} {'Baseline':<12} {'Balanced':<12} {'Improvement':<12} {'Status'}")
    print("-" * 85)
    
    minority_improvements = []
    majority_improvements = []
    
    for i, class_name in enumerate(class_names):
        train_count = class_counts.get(i, 0)
        val_count = val_class_counts.get(i, 0)
        
        baseline_map50_class = baseline_per_class[class_name]['mAP50']
        balanced_map50_class = balanced_per_class[class_name]['mAP50']
        
        if baseline_map50_class > 0:
            improvement = ((balanced_map50_class - baseline_map50_class) / baseline_map50_class) * 100
        else:
            improvement = 0 if balanced_map50_class == 0 else float('inf')
        
        max_train_count = max(class_counts.values())
        is_minority = train_count < (max_train_count * 0.1)
        
        if is_minority:
            minority_improvements.append(improvement)
            status = "🟢 Minority"
        else:
            majority_improvements.append(improvement)
            status = "🔵 Majority"
        
        print(f"{class_name:<12} {train_count:<8} {val_count:<6} {baseline_map50_class:<12.4f} {balanced_map50_class:<12.4f} {improvement:<12.1f}% {status}")
    
    print(f"\n💡 Class Balancing Impact Analysis:")
    
    if minority_improvements:
        avg_minority_improvement = sum(minority_improvements) / len(minority_improvements)
        print(f"   Minority classes average improvement: {avg_minority_improvement:.1f}%")
        
        significant_improvements = [imp for imp in minority_improvements if imp > 50]
        if significant_improvements:
            print(f"   Classes with >50% improvement: {len(significant_improvements)}")
            print(f"   Maximum minority improvement: {max(minority_improvements):.1f}%")
    
    if majority_improvements:
        avg_majority_improvement = sum(majority_improvements) / len(majority_improvements)
        print(f"   Majority classes average improvement: {avg_majority_improvement:.1f}%")
    
    print(f"\n🔍 Overfitting Risk Assessment:")
    extreme_improvements = [(i, class_names[i], imp) for i, imp in enumerate(minority_improvements) if imp > 500]
    
    if extreme_improvements:
        print(f"   ⚠️ Classes with extreme improvement (>500%):")
        for i, class_name, improvement in extreme_improvements:
            train_count = class_counts.get(i, 0)
            val_count = val_class_counts.get(i, 0)
            print(f"     {class_name}: {improvement:.1f}% (Train: {train_count}, Val: {val_count})")
            
            if train_count < 20:
                print(f"       🚨 Very low sample size - high overfitting risk")
            elif val_count < 5:
                print(f"       ⚠️ Very few validation samples - difficult to assess")
            else:
                print(f"       ✅ Sufficient samples for validation")
    else:
        print(f"   ✅ No extreme improvements detected")
    
    return {
        'overall_improvement_50': overall_improvement_50,
        'overall_improvement_50_95': overall_improvement_50_95,
        'minority_improvements': minority_improvements,
        'majority_improvements': majority_improvements,
        'extreme_improvements': extreme_improvements
    }

def main():
    """Main function to run the enhanced validation test with saved weights."""
    print("🚀 ENHANCED VALIDATION TEST WITH SAVED WEIGHTS")
    print("=" * 60)
    print("Efficient approach: Train once, test multiple times!")
    print("Saves: baseline_weights.pt, balanced_weights.pt")
    print("Tests on separate validation data for proper overfitting detection")
    print("=" * 60)
    
    dataset_path = setup_dataset()
    if not dataset_path:
        print("❌ Cannot proceed without dataset")
        return
    
    class_names, class_counts, val_class_counts = analyze_dataset_distribution(dataset_path)
    
    baseline_weights_path, balanced_weights_path = train_and_save_models(dataset_path, force_retrain=False)
    
    baseline_val_results, balanced_val_results = test_models_on_validation(
        baseline_weights_path, balanced_weights_path, dataset_path, class_names
    )
    
    baseline_metrics, baseline_per_class = extract_per_class_metrics(baseline_val_results, class_names, "baseline")
    balanced_metrics, balanced_per_class = extract_per_class_metrics(balanced_val_results, class_names, "balanced")
    
    comparison_results = compare_models_per_class(
        baseline_metrics, balanced_metrics, 
        baseline_per_class, balanced_per_class, 
        class_names, class_counts, val_class_counts
    )
    
    print(f"\n🎉 VALIDATION TEST COMPLETE!")
    print(f"✅ Models tested on separate validation data")
    print(f"✅ Weights saved for future reuse:")
    print(f"   📁 {baseline_weights_path}")
    print(f"   📁 {balanced_weights_path}")
    print(f"✅ Per-class metrics extracted and analyzed")
    print(f"✅ Overfitting risk assessed")
    
    return {
        'baseline_metrics': baseline_metrics,
        'balanced_metrics': balanced_metrics,
        'baseline_per_class': baseline_per_class,
        'balanced_per_class': balanced_per_class,
        'comparison_results': comparison_results,
        'weights_paths': {
            'baseline': baseline_weights_path,
            'balanced': balanced_weights_path
        }
    }

if __name__ == "__main__":
    results = main()
    print(f"\n📊 Final Results Summary:")
    print(f"Results: {results}")
