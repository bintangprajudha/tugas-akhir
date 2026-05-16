"""
Low-Rank Decomposition (SVD) - Multi-Rank Comparison
Decompose Dense feature layer menggunakan SVD untuk model compression
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
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             roc_auc_score, roc_curve, auc)
import matplotlib.pyplot as plt
import seaborn as sns
import cv2

# ============================================================================
# SETUP
# ============================================================================
SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
os.environ["TF_DETERMINISTIC_OPS"] = "1"
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
def get_file_size_kb(path):
    return os.path.getsize(path) / 1024.0 if os.path.exists(path) else 0.0

def _gzip_file(src, dst):
    with open(src, "rb") as fi, gzip.open(dst, "wb") as fo:
        shutil.copyfileobj(fi, fo)

def evaluate_pipeline(extractor, svm_clf, scaler, X_test, y_test_int, class_names, label="Model"):
    # Feature extraction
    t0 = time.time()
    X_feat = extractor.predict(X_test, verbose=0, batch_size=16)
    feat_time = time.time() - t0
    
    # SVM inference
    t0 = time.time()
    X_scaled = scaler.transform(X_feat)
    y_pred = svm_clf.predict(X_scaled)
    y_proba = svm_clf.predict_proba(X_scaled)
    inf_time = time.time() - t0
    
    # Metrics
    accuracy = accuracy_score(y_test_int, y_pred)
    cm = confusion_matrix(y_test_int, y_pred)
    
    # Per-class
    report = classification_report(y_test_int, y_pred, target_names=class_names, output_dict=True)
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
    y_bin = label_binarize(y_test_int, classes=range(len(class_names)))
    auc_macro = roc_auc_score(y_bin, y_proba, average='macro', multi_class='ovr')
    
    return {
        'label': label,
        'accuracy': accuracy,
        'sensitivity': sensitivity,
        'specificity': specificity,
        'precision': precision,
        'f1': f1,
        'auc_macro': auc_macro,
        'cm': cm,
        'feat_time': feat_time,
        'inf_time': inf_time
    }

# ============================================================================
# LOW-RANK DECOMPOSITION
# ============================================================================
def svd_lowrank_dense(W, b, rank):
    """SVD-decompose Dense weight W into two matrices"""
    U, S, Vt = np.linalg.svd(W, full_matrices=False)
    r = max(1, min(rank, len(S)))
    W1 = U[:, :r] * np.sqrt(S[:r])
    W2 = (np.sqrt(S[:r])[:, None]) * Vt[:r, :]
    return W1.astype(np.float32), W2.astype(np.float32), b.astype(np.float32)

def build_lowrank_model(orig_model, num_classes, feature_rank=4):
    """Build model with low-rank feature layer"""
    inputs = orig_model.input
    
    # Copy layers up to feature_layer
    x = inputs
    for layer in orig_model.layers[1:]:
        if layer.name == "feature_layer":
            break
        x = layer(x)
    
    # Low-rank replacement
    x_low = Dense(feature_rank, activation=None, use_bias=False, name="feature_lowrank")(x)
    features = Dense(16, activation="relu", name="feature_layer")(x_low)
    output = Dense(num_classes, activation="softmax", name="softmax_output")(features)
    
    return Model(inputs, output, name=f"CNN_LowRank_r{feature_rank}")

def transfer_weights_with_lowrank(orig_model, lowrank_model, feature_rank=4):
    """Transfer weights and decompose feature_layer"""
    name_to_orig = {l.name: l for l in orig_model.layers}
    name_to_low = {l.name: l for l in lowrank_model.layers}
    
    # Copy unchanged layers
    for name, lr_layer in name_to_low.items():
        if name in name_to_orig and name not in ("feature_layer", "feature_lowrank", "softmax_output"):
            try:
                orig_weights = name_to_orig[name].get_weights()
                if orig_weights:
                    lr_layer.set_weights(orig_weights)
            except Exception as e:
                print(f"[WARN] Could not copy {name}: {e}")
    
    # SVD decomposition
    W, b = name_to_orig["feature_layer"].get_weights()
    print(f"  Decomposing feature_layer: {W.shape} → rank={feature_rank}")
    
    W1, W2, b_new = svd_lowrank_dense(W, b, feature_rank)
    name_to_low["feature_lowrank"].set_weights([W1])
    name_to_low["feature_layer"].set_weights([W2, b_new])
    
    return lowrank_model

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

img_size = 224
categories = ["Bengin cases", "Malignant cases", "Normal cases"]
class_names = ["Bengin", "Malignant", "Normal"]
num_classes = len(categories)

def load_split_data(split_path, categories):
    X, y = [], []
    for class_idx, cat in enumerate(categories):
        cat_path = os.path.join(split_path, cat)
        if not os.path.isdir(cat_path):
            continue
        for fn in sorted(os.listdir(cat_path)):
            if not fn.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            img = cv2.imread(os.path.join(cat_path, fn))
            if img is None:
                continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (img_size, img_size))
            img = img.astype(np.float32) / 255.0
            X.append(img)
            y.append(class_idx)
    return np.array(X, dtype=np.float32), np.array(y)

print("Loading dataset...")
X_train, y_train_int = load_split_data(os.path.join(dataset_base_path, "train"), categories)
X_valid, y_valid_int = load_split_data(os.path.join(dataset_base_path, "valid"), categories)
X_test, y_test_int = load_split_data(os.path.join(dataset_base_path, "test"), categories)

y_train = to_categorical(y_train_int, num_classes)
y_valid = to_categorical(y_valid_int, num_classes)
y_test = to_categorical(y_test_int, num_classes)

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
extractor_orig = tf.keras.models.load_model(extractor_path, custom_objects=CUSTOM_OBJECTS)  # ✅ FIXED!
svm_orig = joblib.load(svm_path)
scaler_orig = joblib.load(scaler_path)

print(f"✅ Models loaded")
print(f"   CNN params: {model_orig.count_params():,}")
print(f"   Extractor params: {extractor_orig.count_params():,}\n")

# Evaluate baseline
print("Evaluating baseline...")
metrics_orig = evaluate_pipeline(extractor_orig, svm_orig, scaler_orig,
                                X_test, y_test_int, class_names, "Original")
print(f"✅ Baseline accuracy: {metrics_orig['accuracy']:.4f}\n")

# ============================================================================
# SVD ANALYSIS
# ============================================================================
print("="*70)
print("SVD ANALYSIS")
print("="*70)

W_dense, b_dense = model_orig.get_layer("feature_layer").get_weights()
print(f"Feature layer shape: {W_dense.shape} ({np.prod(W_dense.shape):,} params)")

U, S, Vt = np.linalg.svd(W_dense, full_matrices=False)
S_norm = S / S[0]
cum_energy = np.cumsum(S**2) / np.sum(S**2)

print(f"\nTop 10 singular values:")
for i in range(min(10, len(S))):
    print(f"  σ_{i+1}: {S[i]:10.4f}  (normalized: {S_norm[i]:8.4f}, energy: {cum_energy[i]*100:6.2f}%)")

# ============================================================================
# MULTI-RANK LOW-RANK COMPRESSION
# ============================================================================
print("\n" + "="*70)
print("MULTI-RANK LOW-RANK COMPRESSION")
print("="*70)

ranks_to_test = [2, 4, 6, 8, 12]
results = {}
os.makedirs("artifacts", exist_ok=True)

for rank in ranks_to_test:
    print(f"\n>>> Processing rank = {rank}")
    
    # Build low-rank model
    model_lr = build_lowrank_model(model_orig, num_classes, feature_rank=rank)
    model_lr = transfer_weights_with_lowrank(model_orig, model_lr, feature_rank=rank)
    model_lr.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    # Fine-tune
    print(f"  Fine-tuning...")
    hist = model_lr.fit(
        X_train, y_train,
        validation_data=(X_valid, y_valid),
        epochs=5,
        batch_size=32,
        callbacks=[EarlyStopping(patience=2, restore_best_weights=True)],
        verbose=0
    )
    
    # Extract features
    extractor_lr = Model(inputs=model_lr.input,
                        outputs=model_lr.get_layer("feature_layer").output)
    
    # Save models
    cnn_path_lr = f"artifacts/cnn_lowrank_r{rank}.keras"
    ext_path_lr = f"artifacts/extractor_lowrank_r{rank}.keras"
    model_lr.save(cnn_path_lr)
    extractor_lr.save(ext_path_lr)
    
    # Train SVM with consistent hyperparameters
    print(f"  Training SVM...")
    X_train_lr = extractor_lr.predict(X_train, verbose=0, batch_size=16)
    X_valid_lr = extractor_lr.predict(X_valid, verbose=0, batch_size=16)
    
    scaler_lr = StandardScaler()
    X_combined = np.vstack([X_train_lr, X_valid_lr])
    y_combined = np.concatenate([y_train_int, y_valid_int])
    X_scaled = scaler_lr.fit_transform(X_combined)
    
    # Use same hyperparameters as baseline (from training)
    svm_lr = SVC(kernel='rbf', C=1.0, gamma='scale', probability=True, random_state=42)
    svm_lr.fit(X_scaled, y_combined)
    
    # Save SVM & scaler
    joblib.dump(svm_lr, f"artifacts/svm_lowrank_r{rank}.pkl")
    joblib.dump(scaler_lr, f"artifacts/scaler_lowrank_r{rank}.pkl")
    
    # Evaluate
    metrics_lr = evaluate_pipeline(extractor_lr, svm_lr, scaler_lr,
                                  X_test, y_test_int, class_names, f"Low-Rank r={rank}")
    
    # Store results
    results[rank] = {
        'model': model_lr,
        'extractor': extractor_lr,
        'svm': svm_lr,
        'scaler': scaler_lr,
        'metrics': metrics_lr,
        'size_kb': get_file_size_kb(cnn_path_lr)
    }
    
    print(f"  ✅ Rank {rank}: Accuracy={metrics_lr['accuracy']:.4f}, "
          f"Size={get_file_size_kb(cnn_path_lr):.2f} KB")

# ============================================================================
# COMPREHENSIVE COMPARISON TABLE
# ============================================================================
print("\n" + "="*140)
print("COMPREHENSIVE COMPARISON TABLE - LOW-RANK DECOMPOSITION (SVD)")
print("="*140)

# Calculate baseline size
temp_path = "temp.keras"
model_orig.save(temp_path)
size_baseline = os.path.getsize(temp_path) / 1024
os.remove(temp_path)

comp_data = []
comp_data.append({
    "Model": "Original",
    "Rank": "-",
    "Params": f"{model_orig.count_params():,}",
    "Size (KB)": f"{size_baseline:.2f}",
    "Compression": "1.00x",
    "Accuracy": f"{metrics_orig['accuracy']:.4f}",
    "Sensitivity": f"{metrics_orig['sensitivity']:.4f}",
    "Specificity": f"{metrics_orig['specificity']:.4f}",
    "Precision": f"{metrics_orig['precision']:.4f}",
    "F1-Score": f"{metrics_orig['f1']:.4f}",
    "AUC": f"{metrics_orig['auc_macro']:.4f}",
    "Feat Ext (ms)": f"{metrics_orig['feat_time']*1000:.2f}",
    "SVM Inf (ms)": f"{metrics_orig['inf_time']*1000:.2f}",
    "Total (ms)": f"{(metrics_orig['feat_time']+metrics_orig['inf_time'])*1000:.2f}"
})

for rank in ranks_to_test:
    res = results[rank]
    metrics_lr = res['metrics']
    comp_ratio = model_orig.count_params() / res['model'].count_params()
    
    comp_data.append({
        "Model": "Low-Rank",
        "Rank": str(rank),
        "Params": f"{res['model'].count_params():,}",
        "Size (KB)": f"{res['size_kb']:.2f}",
        "Compression": f"{comp_ratio:.2f}x",
        "Accuracy": f"{metrics_lr['accuracy']:.4f}",
        "Sensitivity": f"{metrics_lr['sensitivity']:.4f}",
        "Specificity": f"{metrics_lr['specificity']:.4f}",
        "Precision": f"{metrics_lr['precision']:.4f}",
        "F1-Score": f"{metrics_lr['f1']:.4f}",
        "AUC": f"{metrics_lr['auc_macro']:.4f}",
        "Feat Ext (ms)": f"{metrics_lr['feat_time']*1000:.2f}",
        "SVM Inf (ms)": f"{metrics_lr['inf_time']*1000:.2f}",
        "Total (ms)": f"{(metrics_lr['feat_time']+metrics_lr['inf_time'])*1000:.2f}"
    })

df_comp = pd.DataFrame(comp_data)
print("\n" + df_comp.to_string(index=False))
print("\n" + "="*140)

# ============================================================================
# VISUALIZATIONS
# ============================================================================
print("\n" + "="*70)
print("CREATING VISUALIZATIONS")
print("="*70)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 1. Singular value spectrum
ax = axes[0, 0]
ax.bar(range(1, min(20, len(S))+1), S_norm[:20], color='steelblue', alpha=0.7)
for rank in ranks_to_test:
    ax.axvline(rank, color='red', linestyle='--', alpha=0.5, label=f'r={rank}' if rank==ranks_to_test[0] else '')
ax.set_xlabel('Singular Value Index')
ax.set_ylabel('Normalized Value')
ax.set_title('Singular Value Spectrum')
ax.legend()
ax.grid(alpha=0.3)

# 2. Cumulative energy
ax = axes[0, 1]
ax.plot(range(1, len(S)+1), cum_energy*100, 'g-', linewidth=2)
for rank in ranks_to_test:
    energy_at_rank = cum_energy[rank-1]*100
    ax.scatter([rank], [energy_at_rank], s=100, c='red', zorder=5)
    ax.text(rank, energy_at_rank+2, f'{energy_at_rank:.1f}%', ha='center', fontsize=8)
ax.set_xlabel('Rank')
ax.set_ylabel('Cumulative Energy (%)')
ax.set_title('Energy Captured by Rank')
ax.grid(alpha=0.3)
ax.set_ylim([0, 105])

# 3. Accuracy vs Compression
ax = axes[1, 0]
labels = ['Orig'] + [f'r={r}' for r in ranks_to_test]
acc_vals = [metrics_orig['accuracy']] + [results[r]['metrics']['accuracy'] for r in ranks_to_test]
comp_vals = [1.0] + [model_orig.count_params()/results[r]['model'].count_params() for r in ranks_to_test]

ax.plot(labels, acc_vals, 'o-', linewidth=2, markersize=8, label='Accuracy', color='steelblue')
ax.set_ylabel('Accuracy', color='steelblue')
ax.tick_params(axis='y', labelcolor='steelblue')
ax.set_ylim([min(acc_vals)-0.02, 1.0])

ax2 = ax.twinx()
ax2.bar(labels, comp_vals, alpha=0.3, color='orange', label='Compression')
ax2.set_ylabel('Compression Ratio', color='orange')
ax2.tick_params(axis='y', labelcolor='orange')

ax.set_title('Accuracy vs Compression Trade-off')
ax.grid(alpha=0.3)
lines1, labels1 = ax.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax.legend(lines1 + lines2, labels1 + labels2, loc='lower left')

# 4. Inference time
ax = axes[1, 1]
times = [metrics_orig['feat_time']*1000] + [results[r]['metrics']['feat_time']*1000 for r in ranks_to_test]
ax.bar(labels, times, color='green', alpha=0.7)
ax.set_ylabel('Feature Extraction Time (ms)')
ax.set_title('Inference Speed')
ax.grid(alpha=0.3, axis='y')
for i, v in enumerate(times):
    ax.text(i, v+1, f'{v:.1f}', ha='center', fontsize=9)

plt.tight_layout()
plt.savefig('artifacts/lowrank_analysis.png', dpi=150, bbox_inches='tight')
print("✅ Visualization saved")
plt.close()

# Confusion matrices
n_models = len(ranks_to_test) + 1
ncols = 3
nrows = (n_models + ncols - 1) // ncols
fig, axes = plt.subplots(nrows, ncols, figsize=(15, 5*nrows))
axes = axes.flatten()

# Original
sns.heatmap(metrics_orig['cm'], annot=True, fmt='d', cmap='Blues',
           xticklabels=class_names, yticklabels=class_names, ax=axes[0])
axes[0].set_title(f"Original\nAcc: {metrics_orig['accuracy']:.4f}")

# Low-rank models
for i, rank in enumerate(ranks_to_test):
    metrics_lr = results[rank]['metrics']
    sns.heatmap(metrics_lr['cm'], annot=True, fmt='d', cmap='Blues',
               xticklabels=class_names, yticklabels=class_names, ax=axes[i+1])
    axes[i+1].set_title(f"Rank {rank}\nAcc: {metrics_lr['accuracy']:.4f}")

# Hide unused subplots
for j in range(n_models, len(axes)):
    axes[j].axis('off')

plt.tight_layout()
plt.savefig('artifacts/lowrank_confusion_matrices.png', dpi=150, bbox_inches='tight')
print("✅ Confusion matrices saved")
plt.close()

# ============================================================================
# SAVE RESULTS
# ============================================================================
print("\n" + "="*70)
print("SAVING RESULTS")
print("="*70)

timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
out_folder = f"saved_models_lowrank_{timestamp}"
os.makedirs(out_folder, exist_ok=True)

# Save CSV
df_comp.to_csv(os.path.join(out_folder, "comparison.csv"), index=False)

# Save each rank
for rank in ranks_to_test:
    rank_folder = os.path.join(out_folder, f"rank_{rank}")
    os.makedirs(rank_folder, exist_ok=True)
    
    # Copy files
    shutil.copy2(f"artifacts/cnn_lowrank_r{rank}.keras", 
                os.path.join(rank_folder, f"cnn_model_r{rank}.keras"))
    shutil.copy2(f"artifacts/extractor_lowrank_r{rank}.keras",
                os.path.join(rank_folder, f"extractor_r{rank}.keras"))
    shutil.copy2(f"artifacts/svm_lowrank_r{rank}.pkl",
                os.path.join(rank_folder, f"svm_classifier_r{rank}.pkl"))
    shutil.copy2(f"artifacts/scaler_lowrank_r{rank}.pkl",
                os.path.join(rank_folder, f"scaler_r{rank}.pkl"))
    
    # Save metrics
    metrics = results[rank]['metrics']
    with open(os.path.join(rank_folder, f"metrics_r{rank}.json"), 'w') as f:
        json.dump({
            'accuracy': float(metrics['accuracy']),
            'sensitivity': float(metrics['sensitivity']),
            'specificity': float(metrics['specificity']),
            'auc_macro': float(metrics['auc_macro']),
            'feat_time_ms': float(metrics['feat_time'] * 1000),
            'inf_time_ms': float(metrics['inf_time'] * 1000)
        }, f, indent=2)
    
    print(f"✅ Rank {rank} saved to {rank_folder}/")

# Copy visualizations
shutil.copy2('artifacts/lowrank_analysis.png', out_folder)
shutil.copy2('artifacts/lowrank_confusion_matrices.png', out_folder)

print("\n" + "="*70)
print("LOW-RANK COMPRESSION COMPLETE!")
print("="*70)
print(f"\n📁 All artifacts saved to: {out_folder}/")
print("\n📊 Summary:")
for rank in ranks_to_test:
    metrics = results[rank]['metrics']
    comp_ratio = model_orig.count_params() / results[rank]['model'].count_params()
    print(f"\nRank {rank}:")
    print(f"  Compression: {comp_ratio:.2f}x")
    print(f"  Accuracy: {metrics['accuracy']:.4f}")
    print(f"  Size: {results[rank]['size_kb']:.2f} KB")

print("\n" + "="*70)