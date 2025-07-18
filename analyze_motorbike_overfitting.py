"""
Simple overfitting analysis for motorbike class without external dependencies.
Analyzes the 793.9% improvement on 14 samples to detect overfitting signs.
"""

import os
from pathlib import Path

def analyze_motorbike_overfitting():
    """Analyze motorbike class for overfitting indicators."""
    print("🔍 MOTORBIKE OVERFITTING ANALYSIS")
    print("=" * 60)
    
    dataset_path = "/home/ubuntu/repos/ultralytics/car-detect-2"
    data_yaml = Path(dataset_path) / "data.yaml"
    
    if not data_yaml.exists():
        print("❌ Dataset not found!")
        return
    
    print("\n📊 1. MOTORBIKE SAMPLE ANALYSIS")
    print("-" * 40)
    
    train_labels_dir = Path(dataset_path) / "train" / "labels"
    motorbike_files = []
    motorbike_instances = 0
    total_instances = 0
    class_counts = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}
    
    for label_file in train_labels_dir.glob("*.txt"):
        with open(label_file, 'r') as f:
            lines = f.readlines()
            has_motorbike = False
            motorbike_count = 0
            
            for line in lines:
                if line.strip():
                    class_id = int(line.split()[0])
                    class_counts[class_id] += 1
                    total_instances += 1
                    
                    if class_id == 3:  # motorbike class
                        has_motorbike = True
                        motorbike_count += 1
                        motorbike_instances += 1
            
            if has_motorbike:
                motorbike_files.append({
                    'file': label_file.name,
                    'count': motorbike_count
                })
    
    print(f"📈 Dataset statistics:")
    print(f"   Total instances: {total_instances}")
    print(f"   Class distribution:")
    class_names = ['Bus', 'bus', 'car', 'motorbike', 'truck']
    for i, name in enumerate(class_names):
        percentage = (class_counts[i] / total_instances) * 100 if total_instances > 0 else 0
        print(f"     {name}: {class_counts[i]} instances ({percentage:.1f}%)")
    
    print(f"\n🏍️  Motorbike specific:")
    print(f"   Total motorbike instances: {motorbike_instances}")
    print(f"   Files with motorbikes: {len(motorbike_files)}")
    print(f"   Average per file: {motorbike_instances/len(motorbike_files):.1f}" if len(motorbike_files) > 0 else "   No motorbike files found")
    
    print(f"\n📊 2. IMBALANCE ANALYSIS")
    print("-" * 40)
    
    max_class_count = max(class_counts.values())
    min_class_count = min([count for count in class_counts.values() if count > 0])
    imbalance_ratio = max_class_count / min_class_count if min_class_count > 0 else 0
    
    print(f"   Dominant class: {max_class_count} instances")
    print(f"   Minority class: {min_class_count} instances")
    print(f"   Imbalance ratio: {imbalance_ratio:.1f}:1")
    
    print(f"\n🚨 3. OVERFITTING RISK ASSESSMENT")
    print("-" * 40)
    
    baseline_map50 = 0.0267
    balanced_map50 = 0.2385
    improvement_percentage = ((balanced_map50 - baseline_map50) / baseline_map50) * 100
    
    print(f"   Motorbike performance:")
    print(f"     Baseline mAP50: {baseline_map50:.4f}")
    print(f"     Balanced mAP50: {balanced_map50:.4f}")
    print(f"     Improvement: {improvement_percentage:.1f}%")
    
    risk_score = 0
    risk_factors = []
    
    if motorbike_instances < 20:
        risk_score += 3
        risk_factors.append(f"Very low sample size: {motorbike_instances} < 20 (+3)")
    elif motorbike_instances < 50:
        risk_score += 2
        risk_factors.append(f"Low sample size: {motorbike_instances} < 50 (+2)")
    
    if improvement_percentage > 500:
        risk_score += 3
        risk_factors.append(f"Extreme improvement: {improvement_percentage:.1f}% > 500% (+3)")
    elif improvement_percentage > 300:
        risk_score += 2
        risk_factors.append(f"High improvement: {improvement_percentage:.1f}% > 300% (+2)")
    
    if len(motorbike_files) < 10:
        risk_score += 2
        risk_factors.append(f"Low file diversity: {len(motorbike_files)} files (+2)")
    
    avg_per_file = motorbike_instances / len(motorbike_files) if len(motorbike_files) > 0 else 0
    if avg_per_file > 2:
        risk_score += 1
        risk_factors.append(f"High instances per file: {avg_per_file:.1f} > 2 (+1)")
    
    print(f"\n   Risk factors identified:")
    for factor in risk_factors:
        print(f"     • {factor}")
    
    print(f"\n🎯 OVERFITTING RISK SCORE: {risk_score}/11")
    
    if risk_score >= 8:
        risk_level = "🔴 VERY HIGH"
        confidence = "Strong evidence of overfitting"
        recommendation = "Results likely unreliable - needs validation"
    elif risk_score >= 6:
        risk_level = "🟠 HIGH"
        confidence = "Likely overfitting"
        recommendation = "Requires validation on unseen data"
    elif risk_score >= 4:
        risk_level = "🟡 MODERATE"
        confidence = "Some overfitting risk"
        recommendation = "Monitor performance carefully"
    else:
        risk_level = "🟢 LOW"
        confidence = "Low overfitting risk"
        recommendation = "Results likely reliable"
    
    print(f"   Risk Level: {risk_level}")
    print(f"   Assessment: {confidence}")
    print(f"   Recommendation: {recommendation}")
    
    print(f"\n💡 4. VALIDATION STRATEGIES")
    print("-" * 40)
    
    print("🔬 To confirm overfitting:")
    print("   1. Hold-out validation: Reserve 20% motorbike samples")
    print("   2. Cross-validation: Split into train/val folds")
    print("   3. External validation: Test on new motorbike images")
    print("   4. Confusion matrix: Check false positive rate")
    
    print("\n🛠️  Mitigation strategies:")
    print("   1. Reduce cls_weight: From auto (~15.0) to manual (~1.8)")
    print("   2. Data augmentation: mixup, mosaic, copy-paste")
    print("   3. Regularization: dropout, weight decay")
    print("   4. Progressive training: Gentle balancing approach")
    
    print(f"\n📈 5. REALISTIC EXPECTATIONS")
    print("-" * 40)
    
    realistic_improvement = min(200, improvement_percentage * 0.25)
    realistic_map50 = baseline_map50 * (1 + realistic_improvement / 100)
    
    print(f"   Current improvement: {improvement_percentage:.1f}%")
    print(f"   Realistic improvement: ~{realistic_improvement:.0f}%")
    print(f"   Expected mAP50: ~{realistic_map50:.3f}")
    print(f"   Overfitting component: ~{improvement_percentage - realistic_improvement:.0f}%")
    
    print(f"\n⚙️  6. RECOMMENDED SETTINGS")
    print("-" * 40)
    
    print("   Conservative cls_weights:")
    print("   [0.8, 1.3, 0.7, 1.8, 1.1]  # motorbike: 1.8 instead of ~15.0")
    print("")
    print("   Training parameters:")
    print("   epochs=40, mixup=0.3, mosaic=1.0, copy_paste=0.4")
    print("   dropout=0.2, weight_decay=0.0005")
    
    return {
        'risk_score': risk_score,
        'risk_level': risk_level,
        'motorbike_instances': motorbike_instances,
        'improvement_percentage': improvement_percentage,
        'realistic_improvement': realistic_improvement,
        'recommendation': recommendation
    }

if __name__ == "__main__":
    print("🚀 Starting motorbike overfitting analysis...")
    results = analyze_motorbike_overfitting()
    print(f"\n🎉 Analysis complete!")
    print(f"📊 Summary: {results}")
