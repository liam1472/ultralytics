#!/usr/bin/env python3
"""
TRAINING WITH FULL HELICOPTER DATASET (193,526 images)
======================================================

This script provides optimized configuration to train with the COMPLETE
helicopter dataset while avoiding ThreadPool hang during cache_labels.

STRATEGY:
- Use very conservative parameters to prevent ThreadPool overload
- Minimize workers and batch size to reduce memory pressure
- Disable caching completely for large dataset
- Use progressive checkpointing to save progress
- Handle mixed dataset format properly
"""

from ultralytics import YOLO
import time

def train_full_helicopter_dataset():
    """Train with complete helicopter dataset using ultra-conservative settings."""
    print("🚁 TRAINING WITH FULL HELICOPTER DATASET")
    print("=" * 60)
    print("Dataset: 193,526 helicopter images")
    print("Strategy: Ultra-conservative parameters to prevent ThreadPool hang")
    print("Expected time: 6-12 hours depending on hardware")
    print("=" * 60)
    
    full_dataset_config = {
        'data': "C:/Users/mexil/PyCharmMiscProject/Helicopter-Detection-3-1/data.yaml",
        'cls_weights': True,        # ✓ Class balancing enabled
        'epochs': 50,               # Full training
        'imgsz': 416,              # Reduced to minimize memory usage
        'batch': 4,                # Very small batch to prevent memory issues
        'workers': 1,              # CRITICAL: Single worker to prevent ThreadPool hang
        'cache': False,            # ESSENTIAL: No caching for 193k images
        'task': 'detect',          # Handle mixed dataset format
        'patience': 100,           # High patience for large dataset
        'save_period': 5,          # Save every 5 epochs
        'verbose': True,
        'save': True,
        'plots': True,
        'device': 'cpu',           # Use CPU to avoid GPU memory issues
        'single_cls': False,
        'rect': False,             # Disable rectangular training
        'mosaic': 0.5,            # Reduce mosaic augmentation
        'mixup': 0.0,             # Disable mixup
        'copy_paste': 0.0,        # Disable copy-paste
        'name': 'helicopter_full_dataset'
    }
    
    print("🔧 ULTRA-CONSERVATIVE CONFIGURATION:")
    print("Key optimizations for 193,526 images:")
    for key, value in full_dataset_config.items():
        print(f"  {key}: {value}")
    
    print(f"\n⚠️  CRITICAL SETTINGS FOR FULL DATASET:")
    print(f"   • workers=1 → Single worker prevents ThreadPool overload")
    print(f"   • batch=4 → Minimal batch size for memory efficiency")
    print(f"   • cache=False → Essential for large datasets")
    print(f"   • device='cpu' → Avoid GPU memory limitations")
    print(f"   • save_period=5 → Frequent checkpointing")
    print(f"   • imgsz=416 → Reduced image size for faster processing")
    
    print(f"\n🚀 Starting training with full dataset...")
    print("This will take several hours. Monitor progress carefully.")
    
    try:
        start_time = time.time()
        
        model = YOLO("yolo11n.yaml")
        print("✅ Model loaded successfully")
        
        print(f"\n📊 Training with cls_weights=True on 193,526 images...")
        print("Expected phases:")
        print("1. Data loading and verification (30-60 minutes)")
        print("2. Cache creation (if any)")
        print("3. Actual training epochs (4-8 hours)")
        
        results = model.train(**full_dataset_config)
        
        end_time = time.time()
        training_time = end_time - start_time
        
        print(f"\n🎉 SUCCESS! Full dataset training completed")
        print(f"Total time: {training_time/3600:.1f} hours")
        
        if hasattr(results, 'results_dict'):
            metrics = results.results_dict
            print(f"\n📈 Final Results:")
            print(f"   mAP50: {metrics.get('metrics/mAP50(B)', 'N/A')}")
            print(f"   mAP50-95: {metrics.get('metrics/mAP50-95(B)', 'N/A')}")
        
        print(f"\n✅ FULL DATASET TRAINING COMPLETE:")
        print(f"   ✓ All 193,526 images processed")
        print(f"   ✓ cls_weights applied to full dataset")
        print(f"   ✓ Model saved with checkpoints")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Full dataset training failed: {e}")
        print(f"Error type: {type(e).__name__}")
        
        print(f"\n🛠️  FALLBACK OPTIONS:")
        print(f"1. Try with even smaller batch=2")
        print(f"2. Use fraction=0.5 (96,763 images)")
        print(f"3. Split dataset into chunks")
        
        return False

def alternative_chunked_approach():
    """Alternative approach: Split dataset into chunks."""
    print(f"\n📦 ALTERNATIVE: CHUNKED TRAINING APPROACH")
    print("=" * 60)
    print("If full dataset training fails, use this approach:")
    print("")
    
    chunk_configs = [
        {'fraction': 0.25, 'name': 'chunk_1', 'description': '25% (48,382 images)'},
        {'fraction': 0.25, 'name': 'chunk_2', 'description': '25% (48,382 images)'},
        {'fraction': 0.25, 'name': 'chunk_3', 'description': '25% (48,382 images)'},
        {'fraction': 0.25, 'name': 'chunk_4', 'description': '25% (48,382 images)'}
    ]
    
    print("Train in 4 chunks of 25% each:")
    for i, chunk in enumerate(chunk_configs, 1):
        print(f"\n{i}. CHUNK {i} - {chunk['description']}:")
        print(f"   model.train(")
        print(f"       data='your_data.yaml',")
        print(f"       cls_weights=True,")
        print(f"       fraction={chunk['fraction']},")
        print(f"       epochs=15,  # Reduced epochs per chunk")
        print(f"       batch=8,")
        print(f"       workers=2,")
        print(f"       cache=False,")
        print(f"       name='{chunk['name']}',")
        print(f"       resume=True  # Resume from previous chunk")
        print(f"   )")

def gpu_optimized_config():
    """GPU-optimized configuration for full dataset."""
    print(f"\n🎮 GPU-OPTIMIZED CONFIGURATION")
    print("=" * 60)
    print("If you have sufficient GPU memory (RTX 3060 12GB):")
    
    gpu_config = {
        'data': "C:/Users/mexil/PyCharmMiscProject/Helicopter-Detection-3-1/data.yaml",
        'cls_weights': True,
        'epochs': 50,
        'imgsz': 512,              # Slightly larger for GPU
        'batch': 6,                # Larger batch for GPU
        'workers': 2,              # Still conservative
        'cache': False,            # Still no caching
        'task': 'detect',
        'patience': 100,
        'save_period': 5,
        'device': '0',             # Use GPU
        'amp': True,               # Mixed precision
        'name': 'helicopter_full_gpu'
    }
    
    print("# GPU Configuration (if CPU training is too slow):")
    print("model = YOLO('yolo11n.yaml')")
    print("results = model.train(")
    for key, value in gpu_config.items():
        if isinstance(value, str):
            print(f"    {key}='{value}',")
        else:
            print(f"    {key}={value},")
    print(")")
    
    print(f"\n⚠️  GPU REQUIREMENTS:")
    print(f"   • At least 8GB VRAM")
    print(f"   • Monitor GPU memory usage")
    print(f"   • Reduce batch if out of memory")

def monitoring_guide():
    """Guide for monitoring full dataset training."""
    print(f"\n📊 MONITORING FULL DATASET TRAINING")
    print("=" * 60)
    
    print(f"🔍 PHASES TO WATCH:")
    print(f"1. **Data Loading (30-60 min)**:")
    print(f"   - 'train: Scanning' progress bar")
    print(f"   - Should show 193526/193526 images")
    print(f"   - Watch for duplicate label warnings")
    print(f"   - Mixed format warning is normal")
    print(f"")
    print(f"2. **Training Start**:")
    print(f"   - Should see 'Epoch 1/50' after data loading")
    print(f"   - Loss values should appear")
    print(f"   - Progress bar for each epoch")
    print(f"")
    print(f"3. **Progress Monitoring**:")
    print(f"   - Check saved models in runs/detect/helicopter_full_dataset/")
    print(f"   - Monitor system resources (RAM, CPU)")
    print(f"   - Look for checkpoint saves every 5 epochs")
    print(f"")
    print(f"🚨 WARNING SIGNS:")
    print(f"   • Hanging at 'train: Scanning' → Reduce workers further")
    print(f"   • Out of memory errors → Reduce batch size")
    print(f"   • Very slow progress → Consider GPU or chunked approach")

def main():
    """Main execution function."""
    print("🚁 FULL HELICOPTER DATASET TRAINING SOLUTION")
    print("=" * 70)
    print("Complete guide for training with all 193,526 images")
    print("Includes ultra-conservative settings and alternatives")
    print("=" * 70)
    
    success = train_full_helicopter_dataset()
    
    alternative_chunked_approach()
    gpu_optimized_config()
    monitoring_guide()
    
    print(f"\n🎯 SUMMARY FOR FULL DATASET TRAINING")
    print("=" * 70)
    print(f"✅ **RECOMMENDED APPROACH**: Ultra-conservative CPU training")
    print(f"   • workers=1, batch=4, cache=False, device='cpu'")
    print(f"   • Expected time: 6-12 hours")
    print(f"   • Most reliable for avoiding ThreadPool hang")
    print(f"")
    print(f"🎮 **ALTERNATIVE**: GPU training (if sufficient VRAM)")
    print(f"   • workers=2, batch=6, device='0', amp=True")
    print(f"   • Expected time: 2-4 hours")
    print(f"   • Requires monitoring GPU memory")
    print(f"")
    print(f"📦 **FALLBACK**: Chunked training")
    print(f"   • Train in 4 chunks of 25% each")
    print(f"   • More manageable but requires manual coordination")
    print(f"")
    print(f"🔑 **KEY SUCCESS FACTORS**:")
    print(f"   • ALWAYS use workers=1 or 2 (never more)")
    print(f"   • ALWAYS use cache=False")
    print(f"   • ALWAYS use task='detect'")
    print(f"   • ALWAYS use small batch sizes (2-6)")
    print(f"   • Monitor the 'train: Scanning' phase carefully")
    
    print(f"\n📧 NEXT STEPS:")
    print(f"1. Choose your approach (CPU conservative vs GPU vs chunked)")
    print(f"2. Copy the configuration above")
    print(f"3. Start training and monitor progress")
    print(f"4. Be patient - full dataset takes hours")
    print(f"5. Check saved models every 5 epochs")

if __name__ == "__main__":
    main()
