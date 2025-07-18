"""
Practical test to validate motorbike overfitting hypothesis by comparing:
1. Extreme weights (auto cls_weights=True) - current 793.9% improvement
2. Conservative weights (manual cls_weights) - expected ~200% improvement

This will prove whether the extreme improvement is due to overfitting.
"""

import os
import time
from pathlib import Path
from ultralytics import YOLO

def test_conservative_vs_extreme_weights():
    """Test conservative vs extreme cls_weights to validate overfitting hypothesis."""
    print("🧪 CONSERVATIVE VS EXTREME WEIGHTS TEST")
    print("=" * 60)
    print("Purpose: Validate motorbike overfitting hypothesis")
    print("Hypothesis: Conservative weights will show sustainable improvement")
    print("Expected: ~200% improvement instead of 793.9%")
    print("=" * 60)
    
    dataset_path = "/home/ubuntu/repos/ultralytics/car-detect-2"
    data_yaml = Path(dataset_path) / "data.yaml"
    
    if not data_yaml.exists():
        print("❌ Dataset not found!")
        return
    
    results = {}
    
    print("\n🔄 TEST 1: CONSERVATIVE WEIGHTS")
    print("-" * 40)
    print("cls_weights = [0.8, 1.3, 0.7, 1.8, 1.1]")
    print("Expected: Sustainable improvement ~150-250%")
    
    try:
        start_time = time.time()
        
        model1 = YOLO('yolo11n.yaml')
        
        results1 = model1.train(
            data=str(data_yaml),
            epochs=15,  # Shorter epochs for quick test
            imgsz=320,
            batch=4,
            cls_weights=[0.8, 1.3, 0.7, 1.8, 1.1],  # Conservative weights
            device='cpu',
            patience=15,
            save=False,
            plots=False,
            verbose=True,
            name='conservative_weights_test'
        )
        
        end_time = time.time()
        training_time1 = end_time - start_time
        
        conservative_metrics = {
            'mAP50': results1.results_dict.get('metrics/mAP50(B)', 0),
            'mAP50-95': results1.results_dict.get('metrics/mAP50-95(B)', 0),
            'training_time': training_time1
        }
        
        print(f"✅ Conservative training completed in {training_time1:.1f}s")
        print(f"📊 Results: mAP50={conservative_metrics['mAP50']:.4f}, mAP50-95={conservative_metrics['mAP50-95']:.4f}")
        
        results['conservative'] = conservative_metrics
        
    except Exception as e:
        print(f"❌ Conservative training failed: {e}")
        results['conservative'] = None
    
    print("\n🔄 TEST 2: EXTREME WEIGHTS (AUTO)")
    print("-" * 40)
    print("cls_weights = True (auto calculation)")
    print("Expected: Extreme improvement ~793% (potentially overfitted)")
    
    try:
        start_time = time.time()
        
        model2 = YOLO('yolo11n.yaml')
        
        results2 = model2.train(
            data=str(data_yaml),
            epochs=15,  # Same epochs for fair comparison
            imgsz=320,
            batch=4,
            cls_weights=True,  # Auto extreme weights
            device='cpu',
            patience=15,
            save=False,
            plots=False,
            verbose=True,
            name='extreme_weights_test'
        )
        
        end_time = time.time()
        training_time2 = end_time - start_time
        
        extreme_metrics = {
            'mAP50': results2.results_dict.get('metrics/mAP50(B)', 0),
            'mAP50-95': results2.results_dict.get('metrics/mAP50-95(B)', 0),
            'training_time': training_time2
        }
        
        print(f"✅ Extreme training completed in {training_time2:.1f}s")
        print(f"📊 Results: mAP50={extreme_metrics['mAP50']:.4f}, mAP50-95={extreme_metrics['mAP50-95']:.4f}")
        
        results['extreme'] = extreme_metrics
        
    except Exception as e:
        print(f"❌ Extreme training failed: {e}")
        results['extreme'] = None
    
    print("\n🔄 TEST 3: BASELINE (NO BALANCING)")
    print("-" * 40)
    print("cls_weights = False (no class balancing)")
    print("Expected: Poor performance on minority classes")
    
    try:
        start_time = time.time()
        
        model3 = YOLO('yolo11n.yaml')
        
        results3 = model3.train(
            data=str(data_yaml),
            epochs=15,  # Same epochs
            imgsz=320,
            batch=4,
            cls_weights=False,  # No balancing
            device='cpu',
            patience=15,
            save=False,
            plots=False,
            verbose=True,
            name='baseline_test'
        )
        
        end_time = time.time()
        training_time3 = end_time - start_time
        
        baseline_metrics = {
            'mAP50': results3.results_dict.get('metrics/mAP50(B)', 0),
            'mAP50-95': results3.results_dict.get('metrics/mAP50-95(B)', 0),
            'training_time': training_time3
        }
        
        print(f"✅ Baseline training completed in {training_time3:.1f}s")
        print(f"📊 Results: mAP50={baseline_metrics['mAP50']:.4f}, mAP50-95={baseline_metrics['mAP50-95']:.4f}")
        
        results['baseline'] = baseline_metrics
        
    except Exception as e:
        print(f"❌ Baseline training failed: {e}")
        results['baseline'] = None
    
    print("\n" + "="*60)
    print("📊 COMPARATIVE ANALYSIS")
    print("="*60)
    
    if all(results.values()):
        baseline_map50 = results['baseline']['mAP50']
        conservative_map50 = results['conservative']['mAP50']
        extreme_map50 = results['extreme']['mAP50']
        
        if baseline_map50 > 0:
            conservative_improvement = ((conservative_map50 - baseline_map50) / baseline_map50) * 100
            extreme_improvement = ((extreme_map50 - baseline_map50) / baseline_map50) * 100
        else:
            conservative_improvement = 0
            extreme_improvement = 0
        
        print(f"🎯 Performance Comparison:")
        print(f"   Baseline mAP50:     {baseline_map50:.4f}")
        print(f"   Conservative mAP50: {conservative_map50:.4f} ({conservative_improvement:+.1f}%)")
        print(f"   Extreme mAP50:      {extreme_map50:.4f} ({extreme_improvement:+.1f}%)")
        
        print(f"\n⏱️  Training Time:")
        print(f"   Baseline:     {results['baseline']['training_time']:.1f}s")
        print(f"   Conservative: {results['conservative']['training_time']:.1f}s")
        print(f"   Extreme:      {results['extreme']['training_time']:.1f}s")
        
        print(f"\n🔍 Overfitting Analysis:")
        
        if extreme_improvement > conservative_improvement * 2:
            print("   🚨 OVERFITTING DETECTED!")
            print(f"   Extreme improvement ({extreme_improvement:.1f}%) >> Conservative ({conservative_improvement:.1f}%)")
            print("   Recommendation: Use conservative weights for production")
        elif extreme_improvement > conservative_improvement * 1.5:
            print("   ⚠️  POSSIBLE OVERFITTING")
            print(f"   Extreme improvement ({extreme_improvement:.1f}%) > Conservative ({conservative_improvement:.1f}%)")
            print("   Recommendation: Monitor validation performance")
        else:
            print("   ✅ NO SIGNIFICANT OVERFITTING")
            print(f"   Improvements are comparable: Extreme ({extreme_improvement:.1f}%) vs Conservative ({conservative_improvement:.1f}%)")
        
        print(f"\n💡 Sustainability Analysis:")
        if conservative_improvement > 100 and conservative_improvement < 300:
            print(f"   ✅ Conservative improvement ({conservative_improvement:.1f}%) is sustainable")
        elif conservative_improvement > 300:
            print(f"   ⚠️  Even conservative improvement ({conservative_improvement:.1f}%) may be high")
        else:
            print(f"   📊 Conservative improvement ({conservative_improvement:.1f}%) is moderate")
        
        print(f"\n🎯 FINAL RECOMMENDATION:")
        if conservative_improvement > 50:
            print("   ✅ Use conservative cls_weights: [0.8, 1.3, 0.7, 1.8, 1.1]")
            print("   ✅ Expected sustainable improvement for minority classes")
            print("   ✅ Reduced risk of overfitting")
        else:
            print("   📊 Consider further tuning of cls_weights")
            print("   📊 May need more epochs or different approach")
    
    else:
        print("❌ Some tests failed - cannot perform complete analysis")
    
    return results

if __name__ == "__main__":
    print("🚀 Starting conservative vs extreme weights validation test...")
    test_results = test_conservative_vs_extreme_weights()
    print(f"\n🎉 Validation test complete!")
    print(f"📊 Results summary: {test_results}")
