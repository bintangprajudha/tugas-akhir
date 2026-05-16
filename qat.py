"""
QAT (Quantization-Aware Training)
==================================
Fine-tune model dengan fake quantization untuk minimize accuracy loss
"""

# ============================================================================
# CRITICAL: Set environment variable BEFORE importing tensorflow!
# ============================================================================
import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"

import time, json, shutil
from datetime import datetime
import numpy as np
import pandas as pd
import joblib
import cv2
import tensorflow as tf
import tf_keras
from tf_keras.models import Model
from tf_keras.layers import (Layer, Conv2D, MaxPooling2D, Flatten, Dense,
                             GlobalAveragePooling2D, Reshape, Multiply, Input)
from tf_keras.callbacks import EarlyStopping
from tf_keras.utils import to_categorical
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix, roc_auc_score)
import tensorflow_model_optimization as tfmot

print(f"TensorFlow: {tf.__version__}")
print(f"tf_keras: {tf_keras.__version__}\n")

# ============================================================================
# SETTINGS
# ============================================================================
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

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
# ARCHITECTURE - UPDATE TO MATCH YOUR MODEL!
# ============================================================================
def _build_arch_tfkeras(input_shape, num_classes):
    """
    Simple 2-stage architecture
    UPDATE this if your original model has different architecture!
    """
    inputs = Input(shape=input_shape)
    
    # Stage 1
    x = Conv2D(64, 3, padding="same", activation="relu")(inputs)
    x = ChannelAttention(ratio=8)(x)
    x = MaxPooling2D(2)(x)
    
    # Stage 2
    x = Conv2D(32, 3, padding="same", activation="relu")(x)
    x = ChannelAttention(ratio=8)(x)
    x = MaxPooling2D(2)(x)
    
    # Features
    x = Flatten()(x)
    features = Dense(16, activation="relu", name="feature_layer")(x)
    output = Dense(num_classes, activation="softmax", name="softmax_output")(features)
    
    return Model(inputs, output, name="CNN_Attention_tfkeras")

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

def _make_concrete_fn(keras_model):
    in_shape = tuple(keras_model.input_shape)
    spec = tf.TensorSpec(shape=in_shape, dtype=tf.float32, name="input")
    class _Wrap(tf.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m
        @tf.function(input_signature=[spec])
        def __call__(self, x):
            return self.m(x, training=False)
    wrapper = _Wrap(keras_model)
    concrete = wrapper.__call__.get_concrete_function()
    return concrete, wrapper

class TFLiteExtractor:
    def __init__(self, tflite_path):
        self.interpreter = tf.lite.Interpreter(model_path=tflite_path)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
    
    def predict(self, X, verbose=0, batch_size=None):
        feats = []
        for img in X:
            self.interpreter.set_tensor(self.input_details[0]['index'], img[np.newaxis, ...])
            self.interpreter.invoke()
            feats.append(self.interpreter.get_tensor(self.output_details[0]['index'])[0])
        return np.array(feats)

def _load_weights_from_keras_zip(tfk_model, keras_path):
    """Load weights from Keras 3.x .keras file into tf_keras model"""
    import zipfile, tempfile, h5py
    with zipfile.ZipFile(keras_path, 'r') as z:
        with tempfile.TemporaryDirectory() as tmpdir:
            z.extract('model.weights.h5', tmpdir)
            h5_path = os.path.join(tmpdir, 'model.weights.h5')
            
            tfk_layers = [l for l in tfk_model.layers if l.weights]
            
            with h5py.File(h5_path, 'r') as f:
                if 'layers' not in f:
                    return
                
                k3_layer_names = sorted(f['layers'].keys())
                used_k3 = set()
                
                for tfk_layer in tfk_layers:
                    target_shapes = [w.shape for w in tfk_layer.get_weights()]
                    
                    for k3_name in k3_layer_names:
                        if k3_name in used_k3:
                            continue
                        k3_group = f['layers'][k3_name]
                        
                        def collect_vars(group):
                            res = []
                            if 'vars' in group:
                                for v in sorted(group['vars'].keys()):
                                    res.append(np.array(group['vars'][v]))
                            for k in sorted(group.keys()):
                                if k != 'vars' and isinstance(group[k], h5py.Group):
                                    res.extend(collect_vars(group[k]))
                            return res
                        
                        k3_weights = collect_vars(k3_group)
                        if [w.shape for w in k3_weights] == target_shapes:
                            tfk_layer.set_weights(k3_weights)
                            used_k3.add(k3_name)
                            break
    
    print(f"  ✅ Weights loaded from {os.path.basename(keras_path)}")

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
X_train, y_train_labels = load_data("train")
X_valid, y_valid_labels = load_data("valid")
X_test, y_test_labels = load_data("test")

y_train = to_categorical(y_train_labels, 3)
y_valid = to_categorical(y_valid_labels, 3)

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
svm_path = os.path.join(ORIG_FOLDER, "svm_classifier.pkl")
scaler_path = os.path.join(ORIG_FOLDER, "feature_scaler.pkl")

svm_orig = joblib.load(svm_path)
scaler_orig = joblib.load(scaler_path)

print("Building tf_keras model...")
model_orig = _build_arch_tfkeras((224, 224, 3), 3)
_ = model_orig(X_train[:1])
_load_weights_from_keras_zip(model_orig, cnn_path)

extractor_orig = Model(model_orig.input, model_orig.get_layer("feature_layer").output)

print("Evaluating baseline...")
metrics_orig = evaluate_model(extractor_orig, svm_orig, scaler_orig,
                              X_test, y_test_labels, class_names, "Baseline (FP32)")
print(f"✅ Baseline accuracy: {metrics_orig['accuracy']:.4f}\n")

# ============================================================================
# QAT INT8
# ============================================================================
print("="*70)
print("QAT INT8 PIPELINE")
print("="*70)

# Rebuild model
model_qat_int8 = _build_arch_tfkeras((224, 224, 3), 3)
_ = model_qat_int8(X_train[:1])
_load_weights_from_keras_zip(model_qat_int8, cnn_path)

# Apply quantization annotation
quantize_annotate_layer = tfmot.quantization.keras.quantize_annotate_layer
quantize_apply = tfmot.quantization.keras.quantize_apply
quantize_scope = tfmot.quantization.keras.quantize_scope

def annotate(layer):
    if isinstance(layer, (Conv2D, Dense, MaxPooling2D, Flatten)):
        return quantize_annotate_layer(layer)
    return layer

annotated = tf_keras.models.clone_model(model_qat_int8, clone_function=annotate)
with quantize_scope({"ChannelAttention": ChannelAttention}):
    model_qat_int8 = quantize_apply(annotated)

model_qat_int8.compile(
    optimizer=tf_keras.optimizers.Adam(1e-4),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

print("Fine-tuning with fake quantization (5 epochs)...")
model_qat_int8.fit(
    X_train, y_train,
    validation_data=(X_valid, y_valid),
    epochs=5,
    batch_size=8,
    verbose=1
)

# Convert to TFLite INT8
print("\nConverting to TFLite INT8...")
feat_layer_name = "quant_feature_layer"
extractor_qat_int8_keras = Model(
    model_qat_int8.input,
    model_qat_int8.get_layer(feat_layer_name).output
)

concrete, wrapper = _make_concrete_fn(extractor_qat_int8_keras)
converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete], wrapper)
converter.optimizations = [tf.lite.Optimize.DEFAULT]

def rep_data():
    for i in range(100):
        yield [X_train[i:i+1]]

converter.representative_dataset = rep_data
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.float32
converter.inference_output_type = tf.float32

tflite_int8 = converter.convert()
os.makedirs("artifacts", exist_ok=True)
path_int8 = "artifacts/qat_int8.tflite"
with open(path_int8, "wb") as f:
    f.write(tflite_int8)

size_int8 = len(tflite_int8) / 1024
print(f"✅ INT8 saved: {size_int8:.2f} KB\n")

# Train SVM on INT8 features
print("Training SVM on INT8 features...")
ex_int8 = TFLiteExtractor(path_int8)
X_train_int8 = ex_int8.predict(X_train)
X_valid_int8 = ex_int8.predict(X_valid)

scaler_int8 = StandardScaler()
X_combined = np.vstack([X_train_int8, X_valid_int8])
y_combined = np.concatenate([y_train_labels, y_valid_labels])
X_combined_scaled = scaler_int8.fit_transform(X_combined)

# Use same hyperparameters as baseline
svm_int8 = SVC(kernel='rbf', C=1.0, gamma='scale', probability=True, random_state=42)
svm_int8.fit(X_combined_scaled, y_combined)

joblib.dump(svm_int8, "artifacts/svm_int8.pkl")
joblib.dump(scaler_int8, "artifacts/scaler_int8.pkl")

metrics_int8 = evaluate_model(ex_int8, svm_int8, scaler_int8,
                              X_test, y_test_labels, class_names, "QAT INT8")
print(f"✅ QAT INT8 accuracy: {metrics_int8['accuracy']:.4f}\n")

# ============================================================================
# QAT FP16
# ============================================================================
print("="*70)
print("QAT FP16 PIPELINE")
print("="*70)

model_qat_fp16 = _build_arch_tfkeras((224, 224, 3), 3)
_ = model_qat_fp16(X_train[:1])
_load_weights_from_keras_zip(model_qat_fp16, cnn_path)

model_qat_fp16.compile(
    optimizer=tf_keras.optimizers.Adam(1e-4),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

print("Fine-tuning for FP16 (5 epochs)...")
model_qat_fp16.fit(
    X_train, y_train,
    validation_data=(X_valid, y_valid),
    epochs=5,
    batch_size=8,
    verbose=1
)

# Convert to TFLite FP16
print("\nConverting to TFLite FP16...")
extractor_fp16_keras = Model(
    model_qat_fp16.input,
    model_qat_fp16.get_layer("feature_layer").output
)

concrete, wrapper = _make_concrete_fn(extractor_fp16_keras)
converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete], wrapper)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_types = [tf.float16]

tflite_fp16 = converter.convert()
path_fp16 = "artifacts/qat_fp16.tflite"
with open(path_fp16, "wb") as f:
    f.write(tflite_fp16)

size_fp16 = len(tflite_fp16) / 1024
print(f"✅ FP16 saved: {size_fp16:.2f} KB\n")

# Train SVM on FP16 features
print("Training SVM on FP16 features...")
ex_fp16 = TFLiteExtractor(path_fp16)
X_train_fp16 = ex_fp16.predict(X_train)
X_valid_fp16 = ex_fp16.predict(X_valid)

scaler_fp16 = StandardScaler()
X_combined = np.vstack([X_train_fp16, X_valid_fp16])
X_combined_scaled = scaler_fp16.fit_transform(X_combined)

# Use same hyperparameters as baseline
svm_fp16 = SVC(kernel='rbf', C=1.0, gamma='scale', probability=True, random_state=42)
svm_fp16.fit(X_combined_scaled, y_combined)

joblib.dump(svm_fp16, "artifacts/svm_fp16.pkl")
joblib.dump(scaler_fp16, "artifacts/scaler_fp16.pkl")

metrics_fp16 = evaluate_model(ex_fp16, svm_fp16, scaler_fp16,
                              X_test, y_test_labels, class_names, "QAT FP16")
print(f"✅ QAT FP16 accuracy: {metrics_fp16['accuracy']:.4f}\n")

# ============================================================================
# SAVE & COMPARISON
# ============================================================================
print("="*70)
print("SAVING RESULTS")
print("="*70)

timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
save_folder = f"saved_models_qat_{timestamp}"
os.makedirs(save_folder, exist_ok=True)

# Save models
shutil.copy2(path_int8, os.path.join(save_folder, "qat_int8.tflite"))
shutil.copy2(path_fp16, os.path.join(save_folder, "qat_fp16.tflite"))
shutil.copy2("artifacts/svm_int8.pkl", os.path.join(save_folder, "svm_int8.pkl"))
shutil.copy2("artifacts/scaler_int8.pkl", os.path.join(save_folder, "scaler_int8.pkl"))
shutil.copy2("artifacts/svm_fp16.pkl", os.path.join(save_folder, "svm_fp16.pkl"))
shutil.copy2("artifacts/scaler_fp16.pkl", os.path.join(save_folder, "scaler_fp16.pkl"))

# Save baseline reference
model_orig.save(os.path.join(save_folder, "baseline.keras"))
joblib.dump(svm_orig, os.path.join(save_folder, "svm_baseline.pkl"))
joblib.dump(scaler_orig, os.path.join(save_folder, "scaler_baseline.pkl"))

print(f"✅ Saved to: {save_folder}/\n")

# Comparison table
print("="*70)
print("COMPARISON TABLE")
print("="*70)

results = [metrics_orig, metrics_int8, metrics_fp16]

temp_path = "temp.keras"
model_orig.save(temp_path)
size_baseline = os.path.getsize(temp_path) / 1024
os.remove(temp_path)

sizes = [size_baseline, size_int8, size_fp16]

comp_data = []
for i, m in enumerate(results):
    comp_data.append({
        "Model": m['label'],
        "Size (KB)": f"{sizes[i]:.2f}",
        "Compression": f"{sizes[0]/sizes[i]:.2f}x",
        "Accuracy": f"{m['accuracy']:.4f}",
        "Sensitivity": f"{m['sensitivity']:.4f}",
        "Specificity": f"{m['specificity']:.4f}",
        "F1-Score": f"{m['f1']:.4f}",
        "AUC": f"{m['auc']:.4f}",
        "Feat Ext (ms)": f"{m['feat_time']*1000:.2f}",
        "Total (ms)": f"{(m['feat_time']+m['inf_time'])*1000:.2f}"
    })

df_comp = pd.DataFrame(comp_data)
print("\n" + df_comp.to_string(index=False))

df_comp.to_csv(os.path.join(save_folder, "comparison.csv"), index=False)

# Summary
summary = {}
for i, m in enumerate(results):
    key = m['label'].lower().replace(' ', '_').replace('(', '').replace(')', '')
    summary[key] = {
        'accuracy': float(m['accuracy']),
        'size_kb': float(sizes[i]),
        'degradation_pct': float((results[0]['accuracy'] - m['accuracy']) / results[0]['accuracy'] * 100) if i > 0 else 0.0,
        'speedup': float((results[0]['feat_time'] + results[0]['inf_time']) / (m['feat_time'] + m['inf_time']))
    }

with open(os.path.join(save_folder, "summary.json"), 'w') as f:
    json.dump(summary, f, indent=2)

print("\n" + "="*70)
print("QAT COMPLETE!")
print("="*70)
print(f"\n📁 Saved to: {save_folder}/")
print("\n📊 Summary:")
for i, m in enumerate(results):
    deg = (results[0]['accuracy'] - m['accuracy']) / results[0]['accuracy'] * 100
    speedup = (results[0]['feat_time'] + results[0]['inf_time']) / (m['feat_time'] + m['inf_time'])
    print(f"\n{m['label']}:")
    print(f"  Accuracy: {m['accuracy']:.4f} ({deg:+.2f}% degradation)")
    print(f"  Size: {sizes[i]:.2f} KB ({sizes[0]/sizes[i]:.2f}x compression)")
    print(f"  Time: {(m['feat_time']+m['inf_time'])*1000:.2f} ms ({speedup:.2f}x speedup)")

print("\n" + "="*70)