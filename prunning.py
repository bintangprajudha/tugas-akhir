"""
Weight Pruning - Multi-Sparsity Comparison
Magnitude-based pruning untuk model compression
"""
import os, time, gzip, shutil, json
from datetime import datetime
import numpy as np
import pandas as pd
import joblib
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Layer, Conv2D, MaxPooling2D, Flatten, Dense,
                                     GlobalAveragePooling2D, Reshape, Multiply, Input)
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.metrics import (accuracy_score, confusion_matrix, roc_auc_score, classification_report)
import matplotlib.pyplot as plt
import seaborn as sns
import cv2

# Try to import tfmot
try:
    import tensorflow_model_optimization as tfmot
    TFMOT_AVAILABLE = True
    print("✅ tensorflow_model_optimization available")
except:
    TFMOT_AVAILABLE = False
    print("⚠️  tfmot not available - using manual pruning")

# ============================================================================
# SETUP
# ============================================================================
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

print(f"TensorFlow: {tf.__version__}")
print(f"GPU: {len(tf.config.list_physical_devices('GPU'))} device(s)\n")

# ============================================================================
# CUSTOM LAYER
# ============================================================================
class ChannelAttention(Layer):
    def __init__(self, ratio=8, **kwargs):
        super().__init__(**kwargs)
        self.ratio = ratio

    def build(self, input_shape):
        channels = input_shape[-1]
        self.gap = GlobalAveragePooling2D()
        self.dense1 = Dense(max(1, channels // self.ratio), activation="relu")
        self.dense2 = Dense(channels, activation="sigmoid")
        self.reshape = Reshape((1, 1, channels))
        super().build(input_shape)

    def call(self, x):
        attn = self.gap(x)
        attn = self.dense1(attn)
        attn = self.dense2(attn)
        attn = self.reshape(attn)
        return Multiply()([x, attn])

    def get_config(self):
        config = super().get_config()
        config.update({"ratio": self.ratio})
        return config

CUSTOM_OBJECTS = {"ChannelAttention": ChannelAttention}

# ============================================================================
# HELPERS
# ============================================================================
def get_size_kb(path):
    return os.path.getsize(path) / 1024 if os.path.exists(path) else 0

def evaluate_model(extractor, svm, scaler, X, y_true, class_names, label):
    # Feature extraction
    t0 = time.time()
    feats = extractor.predict(X, verbose=0, batch_size=16)
    feat_time = time.time() - t0
    
    # SVM inference
    t0 = time.time()
    feats_scaled = scaler.transform(feats)
    y_pred = svm.predict(feats_scaled)
    y_proba = svm.predict_proba(feats_scaled)
    inf_time = time.time() - t0
    
    # Metrics
    accuracy = accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)
    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
    
    sensitivity = report['macro avg']['recall']
    precision = report['macro avg']['precision']
    f1 = report['macro avg']['f1-score']
    
    # Specificity
    specs = []
    for i in range(len(class_names)):
        tn = cm.sum() - cm[i,:].sum() - cm[:,i].sum() + cm[i,i]
        fp = cm[:,i].sum() - cm[i,i]
        specs.append(tn/(tn+fp) if (tn+fp)>0 else 0)
    specificity = np.mean(specs)
    
    # AUC
    y_bin = label_binarize(y_true, classes=range(len(class_names)))
    auc = roc_auc_score(y_bin, y_proba, average='macro', multi_class='ovr')
    
    return {
        'label': label,
        'accuracy': accuracy,
        'sensitivity': sensitivity,
        'specificity': specificity,
        'precision': precision,
        'f1': f1,
        'auc': auc,
        'cm': cm,
        'feat_time': feat_time,
        'inf_time': inf_time
    }

def calculate_sparsity(model):
    """Calculate actual sparsity (% of zero weights)"""
    total, zeros = 0, 0
    for layer in model.layers:
        for w in layer.get_weights():
            arr = np.asarray(w).flatten()
            total += len(arr)
            zeros += np.sum(np.abs(arr) < 1e-7)
    return (zeros / total * 100) if total > 0 else 0.0

def create_pruned_model(original_model, target_sparsity, X_train, y_train, X_valid, y_valid, epochs=5):
    """Create and fine-tune pruned model"""
    
    use_tfmot = TFMOT_AVAILABLE
    
    if use_tfmot:
        try:
            # Clone model to avoid Keras 3.x compatibility issues
            model_to_prune = tf.keras.models.Model(
                inputs=original_model.input,
                outputs=original_model.output
            )
            model_to_prune.set_weights(original_model.get_weights())
            
            # Use tfmot pruning
            end_step = int(np.ceil(len(X_train) / 32) * epochs)
            pruning_schedule = tfmot.sparsity.keras.PolynomialDecay(
                initial_sparsity=0.0,
                final_sparsity=target_sparsity,
                begin_step=0,
                end_step=end_step
            )
            
            model_pruned = tfmot.sparsity.keras.prune_low_magnitude(
                model_to_prune,
                pruning_schedule=pruning_schedule
            )
            
            callbacks = [
                tfmot.sparsity.keras.UpdatePruningStep(),
                EarlyStopping(patience=2, restore_best_weights=True)
            ]
        except Exception as e:
            print(f"  ⚠️  tfmot failed: {e}")
            print(f"  Falling back to manual pruning...")
            use_tfmot = False
    
    if not use_tfmot:
        # Manual magnitude pruning
        model_pruned = tf.keras.models.clone_model(original_model)
        model_pruned.set_weights(original_model.get_weights())
        
        # Create masks
        masks = []
        for w in model_pruned.get_weights():
            if w.ndim >= 2:  # Only prune Conv/Dense layers
                threshold = np.percentile(np.abs(w).flatten(), target_sparsity * 100)
                mask = (np.abs(w) >= threshold).astype(np.float32)
            else:
                mask = np.ones_like(w, dtype=np.float32)
            masks.append(mask)
        
        # Apply masks callback
        class MaskCallback(tf.keras.callbacks.Callback):
            def __init__(self, masks):
                super().__init__()
                self.masks = masks
            def on_train_batch_end(self, batch, logs=None):
                self.model.set_weights([w * m for w, m in zip(self.model.get_weights(), self.masks)])
        
        callbacks = [MaskCallback(masks), EarlyStopping(patience=2, restore_best_weights=True)]
    
    # Compile
    model_pruned.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Fine-tune
    model_pruned.fit(
        X_train, y_train,
        validation_data=(X_valid, y_valid),
        epochs=epochs,
        batch_size=32,
        callbacks=callbacks,
        verbose=0
    )
    
    # Strip pruning wrappers if using tfmot
    if use_tfmot:
        model_pruned = tfmot.sparsity.keras.strip_pruning(model_pruned)
    
    return model_pruned

# ============================================================================
# LOAD DATASET
# ============================================================================
print("="*70)
print("LOADING DATASET")
print("="*70)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
dataset_base_path = os.path.join(os.path.dirname(SCRIPT_DIR), "dataset_processed2")
if not os.path.exists(dataset_base_path):
    dataset_base_path = "../dataset_processed2"
if not os.path.exists(dataset_base_path):
    dataset_base_path = "dataset_processed2"

categories = ["Bengin cases", "Malignant cases", "Normal cases"]
class_names = ["Bengin", "Malignant", "Normal"]

def load_data(split):
    X, y = [], []
    split_path = os.path.join(dataset_base_path, split)
    for idx, cat in enumerate(categories):
        cat_path = os.path.join(split_path, cat)
        if not os.path.isdir(cat_path):
            continue
        for fn in sorted(os.listdir(cat_path)):
            if not fn.lower().endswith(('.jpg', '.jpeg', '.png')):
                continue
            img = cv2.imread(os.path.join(cat_path, fn))
            if img is None:
                continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (224, 224))
            X.append(img.astype(np.float32)/255.0)
            y.append(idx)
    return np.array(X), np.array(y)

print("Loading dataset...")
X_train, y_train_int = load_data("train")
X_valid, y_valid_int = load_data("valid")
X_test, y_test_int = load_data("test")

y_train = to_categorical(y_train_int, 3)
y_valid = to_categorical(y_valid_int, 3)

print(f"Train: {X_train.shape}")
print(f"Valid: {X_valid.shape}")
print(f"Test:  {X_test.shape}\n")

# ============================================================================
# LOAD BASELINE MODEL
# ============================================================================
print("="*70)
print("LOADING BASELINE MODEL")
print("="*70)

ORIG_FOLDER = os.path.join(SCRIPT_DIR, "saved_models_original")

if not os.path.isdir(ORIG_FOLDER):
    print(f"❌ Folder '{ORIG_FOLDER}' not found!")
    exit(1)

cnn_path = os.path.join(ORIG_FOLDER, "cnn_attention_model.keras")
extractor_path = os.path.join(ORIG_FOLDER, "feature_extractor.keras")
svm_path = os.path.join(ORIG_FOLDER, "svm_classifier.pkl")
scaler_path = os.path.join(ORIG_FOLDER, "feature_scaler.pkl")

print(f"Loading from {ORIG_FOLDER}/...")
model_orig = tf.keras.models.load_model(cnn_path, custom_objects=CUSTOM_OBJECTS)
extractor_orig = tf.keras.models.load_model(extractor_path, custom_objects=CUSTOM_OBJECTS)
svm_orig = joblib.load(svm_path)
scaler_orig = joblib.load(scaler_path)

print(f"✅ Models loaded")
print(f"   Params: {model_orig.count_params():,}\n")

# Evaluate baseline
print("Evaluating baseline...")
metrics_orig = evaluate_model(extractor_orig, svm_orig, scaler_orig,
                              X_test, y_test_int, class_names, "Original")
print(f"✅ Baseline accuracy: {metrics_orig['accuracy']:.4f}\n")

# ============================================================================
# MULTI-SPARSITY PRUNING
# ============================================================================
print("="*70)
print("MULTI-SPARSITY PRUNING")
print("="*70)

sparsity_levels = [0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95]
results = {}
os.makedirs("artifacts", exist_ok=True)

for target_sparsity in sparsity_levels:
    sp_pct = int(target_sparsity * 100)
    print(f"\n>>> Processing sparsity = {sp_pct}%")
    
    # Create pruned model
    print(f"  Pruning and fine-tuning...")
    model_pruned = create_pruned_model(
        model_orig, target_sparsity,
        X_train, y_train, X_valid, y_valid,
        epochs=5
    )
    
    # Calculate actual sparsity
    actual_sparsity = calculate_sparsity(model_pruned)
    print(f"  Actual sparsity: {actual_sparsity:.2f}%")
    
    # Extract features
    extractor_pruned = Model(
        inputs=model_pruned.input,
        outputs=model_pruned.get_layer('feature_layer').output
    )
    
    # Save models
    model_path = f"artifacts/pruned_s{sp_pct}.keras"
    extractor_path = f"artifacts/extractor_s{sp_pct}.keras"
    
    model_pruned.save(model_path)
    extractor_pruned.save(extractor_path)
    
    # Calculate sizes
    size_original = get_size_kb(model_path)
    
    # GZIP compression (zeros compress very well!)
    model_path_gz = model_path + ".gz"
    with open(model_path, 'rb') as f_in:
        with gzip.open(model_path_gz, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)
    
    size_compressed = get_size_kb(model_path_gz)
    
    print(f"  Size: {size_original:.2f} KB (uncompressed), {size_compressed:.2f} KB (gzip)")
    
    # Train SVM
    print(f"  Training SVM...")
    X_train_pruned = extractor_pruned.predict(X_train, verbose=0, batch_size=16)
    X_valid_pruned = extractor_pruned.predict(X_valid, verbose=0, batch_size=16)
    
    scaler_pruned = StandardScaler()
    X_combined = np.vstack([X_train_pruned, X_valid_pruned])
    y_combined = np.concatenate([y_train_int, y_valid_int])
    X_scaled = scaler_pruned.fit_transform(X_combined)
    
    # Use same hyperparameters as baseline
    svm_pruned = SVC(kernel='rbf', C=1.0, gamma='scale', probability=True, random_state=42)
    svm_pruned.fit(X_scaled, y_combined)
    
    # Save SVM
    joblib.dump(svm_pruned, f"artifacts/svm_s{sp_pct}.pkl")
    joblib.dump(scaler_pruned, f"artifacts/scaler_s{sp_pct}.pkl")
    
    # Evaluate
    metrics_pruned = evaluate_model(extractor_pruned, svm_pruned, scaler_pruned,
                                    X_test, y_test_int, class_names, f"Pruned {sp_pct}%")
    
    # Store results
    results[sp_pct] = {
        'model': model_pruned,
        'extractor': extractor_pruned,
        'svm': svm_pruned,
        'scaler': scaler_pruned,
        'metrics': metrics_pruned,
        'actual_sparsity': actual_sparsity,
        'size_kb': size_original,
        'size_compressed_kb': size_compressed
    }
    
    print(f"  ✅ Sparsity {sp_pct}%: Accuracy={metrics_pruned['accuracy']:.4f}, "
          f"Actual sparsity={actual_sparsity:.2f}%")

# ============================================================================
# COMPREHENSIVE COMPARISON TABLE
# ============================================================================
print("\n" + "="*160)
print("COMPREHENSIVE COMPARISON TABLE - MAGNITUDE-BASED PRUNING")
print("="*160)

# Calculate baseline size
temp_path = "temp.keras"
model_orig.save(temp_path)
size_baseline = os.path.getsize(temp_path) / 1024

# GZIP baseline for fair comparison
temp_gz = temp_path + ".gz"
with open(temp_path, 'rb') as f_in:
    with gzip.open(temp_gz, 'wb') as f_out:
        shutil.copyfileobj(f_in, f_out)
size_baseline_gz = os.path.getsize(temp_gz) / 1024

os.remove(temp_path)
os.remove(temp_gz)

comp_data = []
comp_data.append({
    "Sparsity": "0%",
    "Actual": "0.00%",
    "Params": f"{model_orig.count_params():,}",
    "Size (KB)": f"{size_baseline:.2f}",
    "Gzip (KB)": f"{size_baseline_gz:.2f}",
    "Compression": "1.00x",
    "Accuracy": f"{metrics_orig['accuracy']:.4f}",
    "Sensitivity": f"{metrics_orig['sensitivity']:.4f}",
    "Specificity": f"{metrics_orig['specificity']:.4f}",
    "Precision": f"{metrics_orig['precision']:.4f}",
    "F1-Score": f"{metrics_orig['f1']:.4f}",
    "AUC": f"{metrics_orig['auc']:.4f}",
    "Feat Ext (ms)": f"{metrics_orig['feat_time']*1000:.2f}",
    "SVM Inf (ms)": f"{metrics_orig['inf_time']*1000:.2f}",
    "Total (ms)": f"{(metrics_orig['feat_time']+metrics_orig['inf_time'])*1000:.2f}"
})

for sp_pct in sorted(results.keys()):
    res = results[sp_pct]
    metrics_pr = res['metrics']
    comp_data.append({
        "Sparsity": f"{sp_pct}%",
        "Actual": f"{res['actual_sparsity']:.2f}%",
        "Params": f"{res['model'].count_params():,}",
        "Size (KB)": f"{res['size_kb']:.2f}",
        "Gzip (KB)": f"{res['size_compressed_kb']:.2f}",
        "Compression": f"{size_baseline_gz/res['size_compressed_kb']:.2f}x",
        "Accuracy": f"{metrics_pr['accuracy']:.4f}",
        "Sensitivity": f"{metrics_pr['sensitivity']:.4f}",
        "Specificity": f"{metrics_pr['specificity']:.4f}",
        "Precision": f"{metrics_pr['precision']:.4f}",
        "F1-Score": f"{metrics_pr['f1']:.4f}",
        "AUC": f"{metrics_pr['auc']:.4f}",
        "Feat Ext (ms)": f"{metrics_pr['feat_time']*1000:.2f}",
        "SVM Inf (ms)": f"{metrics_pr['inf_time']*1000:.2f}",
        "Total (ms)": f"{(metrics_pr['feat_time']+metrics_pr['inf_time'])*1000:.2f}"
    })

df_comp = pd.DataFrame(comp_data)
print("\n" + df_comp.to_string(index=False))
print("\nNote: Compression ratio based on GZIP compressed size (zeros compress very well)")
print("="*160)

# ============================================================================
# VISUALIZATIONS
# ============================================================================
print("\n" + "="*70)
print("CREATING VISUALIZATIONS")
print("="*70)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 1. Sparsity vs Accuracy
sp_vals = [0] + list(sorted(results.keys()))
acc_vals = [metrics_orig['accuracy']] + [results[sp]['metrics']['accuracy'] for sp in sorted(results.keys())]

axes[0,0].plot(sp_vals, acc_vals, 'o-', linewidth=2, markersize=8, color='steelblue')
axes[0,0].set_xlabel('Target Sparsity (%)')
axes[0,0].set_ylabel('Accuracy')
axes[0,0].set_title('Accuracy vs Sparsity', fontweight='bold')
axes[0,0].grid(alpha=0.3)
axes[0,0].set_ylim([min(acc_vals)-0.02, 1.0])

# 2. Model Size (Uncompressed vs GZIP)
sp_vals_plot = [0] + list(sorted(results.keys()))
sizes_orig = [size_baseline] + [results[sp]['size_kb'] for sp in sorted(results.keys())]
sizes_gzip = [size_baseline_gz] + [results[sp]['size_compressed_kb'] for sp in sorted(results.keys())]
labels = ['0%'] + [f'{sp}%' for sp in sorted(results.keys())]

x = np.arange(len(labels))
width = 0.35

bars1 = axes[0,1].bar(x - width/2, sizes_orig, width, label='Uncompressed', color='lightcoral', alpha=0.7)
bars2 = axes[0,1].bar(x + width/2, sizes_gzip, width, label='GZIP', color='steelblue', alpha=0.7)

axes[0,1].set_ylabel('Model Size (KB)')
axes[0,1].set_title('Model Size: Uncompressed vs GZIP', fontweight='bold')
axes[0,1].set_xticks(x)
axes[0,1].set_xticklabels(labels)
axes[0,1].legend()
axes[0,1].grid(axis='y', alpha=0.3)

# Add value labels
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        axes[0,1].text(bar.get_x() + bar.get_width()/2., height,
                      f'{height:.0f}', ha='center', va='bottom', fontsize=7)

# 3. Actual vs Target Sparsity
target_sp = list(sorted(results.keys()))
actual_sp = [results[sp]['actual_sparsity'] for sp in target_sp]

axes[1,0].plot(target_sp, target_sp, '--', label='Target', color='gray', alpha=0.5)
axes[1,0].plot(target_sp, actual_sp, 'o-', label='Actual', linewidth=2, markersize=8, color='green')
axes[1,0].set_xlabel('Target Sparsity (%)')
axes[1,0].set_ylabel('Actual Sparsity (%)')
axes[1,0].set_title('Target vs Actual Sparsity', fontweight='bold')
axes[1,0].legend()
axes[1,0].grid(alpha=0.3)

# 4. Accuracy Degradation
degradations = [(metrics_orig['accuracy'] - results[sp]['metrics']['accuracy']) / metrics_orig['accuracy'] * 100 
                for sp in sorted(results.keys())]

bars = axes[1,1].bar([f'{sp}%' for sp in sorted(results.keys())], degradations, color='red', alpha=0.7)
axes[1,1].set_ylabel('Accuracy Degradation (%)')
axes[1,1].set_xlabel('Sparsity Level')
axes[1,1].set_title('Accuracy Degradation', fontweight='bold')
axes[1,1].grid(axis='y', alpha=0.3)
axes[1,1].axhline(y=0, color='black', linestyle='-', linewidth=0.5)

plt.tight_layout()
plt.savefig('artifacts/pruning_analysis.png', dpi=150, bbox_inches='tight')
print("✅ Visualization saved")
plt.close()

# ============================================================================
# SAVE RESULTS
# ============================================================================
print("\n" + "="*70)
print("SAVING RESULTS")
print("="*70)

timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
save_folder = f"saved_models_pruning_{timestamp}"
os.makedirs(save_folder, exist_ok=True)

# Save CSV
df_comp.to_csv(os.path.join(save_folder, "comparison.csv"), index=False)

# Save each sparsity level
for sp_pct in sorted(results.keys()):
    sp_folder = os.path.join(save_folder, f"sparsity_{sp_pct}")
    os.makedirs(sp_folder, exist_ok=True)
    
    res = results[sp_pct]
    
    # Copy models
    shutil.copy2(f"artifacts/pruned_s{sp_pct}.keras",
                os.path.join(sp_folder, f"model_s{sp_pct}.keras"))
    shutil.copy2(f"artifacts/extractor_s{sp_pct}.keras",
                os.path.join(sp_folder, f"extractor_s{sp_pct}.keras"))
    shutil.copy2(f"artifacts/svm_s{sp_pct}.pkl",
                os.path.join(sp_folder, f"svm_s{sp_pct}.pkl"))
    shutil.copy2(f"artifacts/scaler_s{sp_pct}.pkl",
                os.path.join(sp_folder, f"scaler_s{sp_pct}.pkl"))
    
    # Save metrics
    with open(os.path.join(sp_folder, f"metrics_s{sp_pct}.json"), 'w') as f:
        json.dump({
            'target_sparsity': float(sp_pct),
            'actual_sparsity': float(res['actual_sparsity']),
            'accuracy': float(res['metrics']['accuracy']),
            'auc': float(res['metrics']['auc'])
        }, f, indent=2)
    
    print(f"✅ Sparsity {sp_pct}% saved")

# Copy visualization
shutil.copy2('artifacts/pruning_analysis.png', save_folder)

print("\n" + "="*70)
print("PRUNING COMPLETE!")
print("="*70)
print(f"\n📁 Saved to: {save_folder}/")
print("\n📊 Summary:")
for sp_pct in sorted(results.keys()):
    res = results[sp_pct]
    degradation = (metrics_orig['accuracy'] - res['metrics']['accuracy']) / metrics_orig['accuracy'] * 100
    compression_gzip = size_baseline_gz / res['size_compressed_kb']
    print(f"\nSparsity {sp_pct}%:")
    print(f"  Actual sparsity: {res['actual_sparsity']:.2f}%")
    print(f"  Accuracy: {res['metrics']['accuracy']:.4f} ({degradation:+.2f}% degradation)")
    print(f"  Size (uncompressed): {res['size_kb']:.2f} KB")
    print(f"  Size (GZIP): {res['size_compressed_kb']:.2f} KB ({compression_gzip:.2f}x compression)")

print("\n" + "="*70)