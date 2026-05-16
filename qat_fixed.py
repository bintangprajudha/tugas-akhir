import os, time, multiprocessing, json, shutil
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
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             roc_auc_score, roc_curve, auc)
import tensorflow_model_optimization as tfmot

# ============================================================================
# SETTINGS & SEED
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
        channels      = input_shape[-1]
        self.gap      = GlobalAveragePooling2D()
        self.dense1   = Dense(max(1, channels // self.ratio), activation="relu")
        self.dense2   = Dense(channels, activation="sigmoid")
        self.reshape  = Reshape((1, 1, channels))
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

def _build_arch_tfkeras(input_shape, num_classes):
    inputs   = Input(shape=input_shape)
    x        = Conv2D(64, 3, padding="same", activation="relu")(inputs)
    x        = ChannelAttention(ratio=8)(x)
    x        = MaxPooling2D(2)(x)
    x        = Conv2D(32, 3, padding="same", activation="relu")(x)
    x        = ChannelAttention(ratio=8)(x)
    x        = MaxPooling2D(2)(x)
    x        = Flatten()(x)
    features = Dense(16, activation="relu", name="feature_layer")(x)
    output   = Dense(num_classes, activation="softmax", name="softmax_output")(features)
    return Model(inputs, output, name="CNN_Attention_tfkeras")

# ============================================================================
# HELPERS
# ============================================================================
def get_file_size_kb(path):
    return os.path.getsize(path) / 1024.0 if os.path.exists(path) else 0.0

def evaluate_pipeline(extractor, svm_clf, scaler, X_test, y_test_int, class_names, label="Model"):
    X_test_feat = extractor.predict(X_test, verbose=0)
    X_test_scaled = scaler.transform(X_test_feat)
    y_pred        = svm_clf.predict(X_test_scaled)
    y_pred_proba  = svm_clf.predict_proba(X_test_scaled)
    
    accuracy = accuracy_score(y_test_int, y_pred)
    cm = confusion_matrix(y_test_int, y_pred)
    
    # Calculate macro AUC
    n_classes = len(class_names)
    y_test_bin = label_binarize(y_test_int, classes=range(n_classes))
    auc_macro = roc_auc_score(y_test_bin, y_pred_proba, average="macro", multi_class="ovr")
    
    return {
        "label": label,
        "accuracy": accuracy,
        "auc_macro": auc_macro,
        "cm": cm,
    }

def _make_concrete_fn(keras_model):
    in_shape = tuple(keras_model.input_shape)
    spec = tf.TensorSpec(shape=in_shape, dtype=tf.float32, name="input")
    class _Wrap(tf.Module):
        def __init__(self, m):
            super().__init__(); self.m = m
        @tf.function(input_signature=[spec])
        def __call__(self, x):
            return self.m(x, training=False)
    wrapper  = _Wrap(keras_model)
    concrete = wrapper.__call__.get_concrete_function()
    return concrete, wrapper

class TFLiteFeatureExtractor:
    def __init__(self, tflite_path):
        self.interpreter = tf.lite.Interpreter(model_path=tflite_path)
        self.interpreter.allocate_tensors()
        in_d = self.interpreter.get_input_details()[0]
        out_d = self.interpreter.get_output_details()[0]
        self._in_idx = in_d["index"]; self._out_idx = out_d["index"]
        self._in_dtype = in_d["dtype"]; self._out_dtype = out_d["dtype"]
        self._in_scale, self._in_zero = in_d.get("quantization", (0.0, 0))
        self._out_scale, self._out_zero = out_d.get("quantization", (0.0, 0))

    def predict(self, X, verbose=0):
        outs = []
        for i in range(len(X)):
            x = X[i:i+1].astype(np.float32) # Always start from float32
            if self._in_dtype in (np.int8, np.uint8) and self._in_scale != 0:
                x = np.clip(np.round(x / self._in_scale + self._in_zero), -128, 127).astype(np.int8)
            else:
                x = x.astype(self._in_dtype)
            
            self.interpreter.set_tensor(self._in_idx, x)
            self.interpreter.invoke()
            o = self.interpreter.get_tensor(self._out_idx).copy()
            if self._out_dtype in (np.int8, np.uint8) and self._out_scale != 0:
                o = (o.astype(np.float32) - self._out_zero) * self._out_scale
            outs.append(o)
        return np.vstack(outs).astype(np.float32)

def _load_weights_from_keras_zip(tfk_model, keras_path):
    import zipfile, tempfile, h5py
    with zipfile.ZipFile(keras_path, 'r') as z:
        with tempfile.TemporaryDirectory() as tmpdir:
            z.extract('model.weights.h5', tmpdir)
            h5_path = os.path.join(tmpdir, 'model.weights.h5')
            
            tfk_layers = [l for l in tfk_model.layers if l.weights]
            
            with h5py.File(h5_path, 'r') as f:
                if 'layers' not in f: return
                k3_layer_names = sorted(f['layers'].keys())
                used_k3 = set()
                for tfk_layer in tfk_layers:
                    target_shapes = [w.shape for w in tfk_layer.get_weights()]
                    found = False
                    for k3_name in k3_layer_names:
                        if k3_name in used_k3: continue
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
                            found = True
                            break
    print(f"  [INFO] Weights loaded from {os.path.basename(keras_path)}")

# ============================================================================
# MAIN EXECUTION
# ============================================================================
def main():
    # --- Paths ---
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    DATASET_PATH = os.path.join(os.path.dirname(SCRIPT_DIR), "dataset_processed2")
    ORIG_FOLDER = os.path.join(SCRIPT_DIR, "saved_models_original")
    ARTIFACTS_DIR = os.path.join(SCRIPT_DIR, "artifacts")
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)

    categories = ["Bengin cases", "Malignant cases", "Normal cases"]
    class_names = ["Bengin", "Malignant", "Normal"]

    # --- Step 1: Load Data ---
    def load_data(split):
        X, y = [], []
        split_path = os.path.join(DATASET_PATH, split)
        for idx, cat in enumerate(categories):
            cat_path = os.path.join(split_path, cat)
            for fn in os.listdir(cat_path):
                img = cv2.imread(os.path.join(cat_path, fn))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, (224, 224))
                X.append(img.astype(np.float32)/255.0); y.append(idx)
        return np.array(X), np.array(y)

    print("Step 1: Loading dataset...")
    X_train, y_train_labels = load_data("train")
    X_valid, y_valid_labels = load_data("valid")
    X_test,  y_test_labels  = load_data("test")
    y_train = to_categorical(y_train_labels, 3)
    y_valid = to_categorical(y_valid_labels, 3)

    # --- Step 2: Load Baseline ---
    print("\nStep 2: Loading baseline models...")
    cnn_path = os.path.join(ORIG_FOLDER, "cnn_attention_model.keras")
    svm_orig = joblib.load(os.path.join(ORIG_FOLDER, "svm_classifier.pkl"))
    scaler_orig = joblib.load(os.path.join(ORIG_FOLDER, "feature_scaler.pkl"))
    
    # Need architecture for extraction from keras 3 file
    model_orig = _build_arch_tfkeras((224, 224, 3), 3)
    _ = model_orig(X_train[:1])
    _load_weights_from_keras_zip(model_orig, cnn_path)
    
    extractor_orig = Model(model_orig.input, model_orig.get_layer("feature_layer").output)
    metrics_orig = evaluate_pipeline(extractor_orig, svm_orig, scaler_orig, X_test, y_test_labels, class_names, "Baseline (FP32)")
    print(f"  Baseline Accuracy: {metrics_orig['accuracy']:.4f}")

    # --- Step 3: QAT INT8 Pipeline ---
    print("\nStep 3: Pipeline QAT INT8...")
    # Rebuild a fresh copy for QAT
    model_qat_int8 = _build_arch_tfkeras((224, 224, 3), 3)
    _ = model_qat_int8(X_train[:1])
    _load_weights_from_keras_zip(model_qat_int8, cnn_path)

    quantize_annotate_layer = tfmot.quantization.keras.quantize_annotate_layer
    quantize_apply          = tfmot.quantization.keras.quantize_apply
    quantize_scope          = tfmot.quantization.keras.quantize_scope

    def annotate(layer):
        if isinstance(layer, (Conv2D, Dense, MaxPooling2D, Flatten)):
            return quantize_annotate_layer(layer)
        return layer

    annotated = tf_keras.models.clone_model(model_qat_int8, clone_function=annotate)
    with quantize_scope({"ChannelAttention": ChannelAttention}):
        model_qat_int8 = quantize_apply(annotated)

    model_qat_int8.compile(optimizer=tf_keras.optimizers.Adam(1e-4), loss="categorical_crossentropy", metrics=["accuracy"])
    print("  Fine-tuning QAT INT8 (5 epochs)...")
    model_qat_int8.fit(X_train, y_train, validation_data=(X_valid, y_valid), epochs=5, batch_size=8, verbose=0)

    # Convert to TFLite INT8
    print("  Converting to TFLite INT8...")
    feat_layer_name = "quant_feature_layer"
    extractor_qat_int8_keras = Model(model_qat_int8.input, model_qat_int8.get_layer(feat_layer_name).output)
    
    concrete, wrapper = _make_concrete_fn(extractor_qat_int8_keras)
    converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete], wrapper)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    def rep_data():
        for i in range(20): yield [X_train[i:i+1]]
    converter.representative_dataset = rep_data
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.float32
    
    tflite_int8 = converter.convert()
    path_int8 = os.path.join(ARTIFACTS_DIR, "qat_int8.tflite")
    with open(path_int8, "wb") as f: f.write(tflite_int8)

    # Train SVM for INT8
    print("  Training SVM on INT8 features...")
    ex_int8 = TFLiteFeatureExtractor(path_int8)
    X_tr_f = ex_int8.predict(X_train); X_va_f = ex_int8.predict(X_valid)
    scaler_int8 = StandardScaler().fit(np.vstack([X_tr_f, X_va_f]))
    svm_int8 = SVC(kernel="rbf", probability=True, random_state=42)
    svm_int8.fit(scaler_int8.transform(np.vstack([X_tr_f, X_va_f])), np.concatenate([y_train_labels, y_valid_labels]))
    
    metrics_int8 = evaluate_pipeline(ex_int8, svm_int8, scaler_int8, X_test, y_test_labels, class_names, "QAT INT8")
    print(f"  QAT INT8 Accuracy: {metrics_int8['accuracy']:.4f}")

    # --- Step 4: QAT FP16 Pipeline ---
    print("\nStep 4: Pipeline QAT FP16 (Fine-tune + FP16)...")
    model_qat_fp16 = _build_arch_tfkeras((224, 224, 3), 3)
    _ = model_qat_fp16(X_train[:1])
    _load_weights_from_keras_zip(model_qat_fp16, cnn_path)
    
    model_qat_fp16.compile(optimizer=tf_keras.optimizers.Adam(1e-4), loss="categorical_crossentropy", metrics=["accuracy"])
    print("  Fine-tuning for FP16 (5 epochs)...")
    model_qat_fp16.fit(X_train, y_train, validation_data=(X_valid, y_valid), epochs=5, batch_size=8, verbose=0)

    # Convert to TFLite FP16
    print("  Converting to TFLite FP16...")
    extractor_fp16_keras = Model(model_qat_fp16.input, model_qat_fp16.get_layer("feature_layer").output)
    concrete, wrapper = _make_concrete_fn(extractor_fp16_keras)
    converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete], wrapper)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_types = [tf.float16]
    
    tflite_fp16 = converter.convert()
    path_fp16 = os.path.join(ARTIFACTS_DIR, "qat_fp16.tflite")
    with open(path_fp16, "wb") as f: f.write(tflite_fp16)

    # Train SVM for FP16
    print("  Training SVM on FP16 features...")
    ex_fp16 = TFLiteFeatureExtractor(path_fp16)
    X_tr_f = ex_fp16.predict(X_train); X_va_f = ex_fp16.predict(X_valid)
    scaler_fp16 = StandardScaler().fit(np.vstack([X_tr_f, X_va_f]))
    svm_fp16 = SVC(kernel="rbf", probability=True, random_state=42)
    svm_fp16.fit(scaler_fp16.transform(np.vstack([X_tr_f, X_va_f])), np.concatenate([y_train_labels, y_valid_labels]))
    
    metrics_fp16 = evaluate_pipeline(ex_fp16, svm_fp16, scaler_fp16, X_test, y_test_labels, class_names, "QAT FP16")
    print(f"  QAT FP16 Accuracy: {metrics_fp16['accuracy']:.4f}")

    # --- Step 5: Summary ---
    print("\n" + "="*50)
    print("FINAL COMPARISON")
    print("="*50)
    data = {
        "Variant": ["Baseline", "QAT INT8", "QAT FP16"],
        "Accuracy": [metrics_orig['accuracy'], metrics_int8['accuracy'], metrics_fp16['accuracy']],
        "Size (KB)": [get_file_size_kb(os.path.join(ORIG_FOLDER, "feature_extractor.keras")), 
                      get_file_size_kb(path_int8), get_file_size_kb(path_fp16)]
    }
    print(pd.DataFrame(data).to_string(index=False))
    print("="*50)

if __name__ == "__main__":
    main()
