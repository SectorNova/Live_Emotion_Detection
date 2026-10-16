"""
Fast, high-accuracy training script for a 48x48 grayscale emotion CNN (FER-style data).

Folder layout expected:
    data/train/<class_name>/*.png
    data/test/<class_name>/*.png      (optional, used for the final honest evaluation)

Highlights
  * Fixed RandomBrightness range bug, train/val split mismatch, and batch-only shuffling
  * Self-contained ResNet + Squeeze-Excitation network (no build_model.py needed)
  * GPU-side augmentation, per-sample MixUp, label smoothing, soft class weights
  * Mixed precision (GPU), AdamW, warmup + cosine LR
  * Flip test-time augmentation + evaluation on data/test + per-class accuracy
  * Saves a CLEAN inference model (48x48x1 input in [0,1]); load with compile=False

Robustness fixes in this version
  * Model export can never crash training (e.g. Windows file lock because app.py /
    realtime_detection.py has the model open) and is written atomically, so an
    interrupted run can't leave a corrupt .keras file
  * `model.save(include_optimizer=...)` falls back for Keras versions that reject it
  * --resume now keeps the previous best (no overwriting it with a worse epoch) and
    uses a low learning rate by default instead of re-warming to the full peak LR
  * Clear error messages when data folders are missing

Usage:
    python train_model.py
    python train_model.py --epochs 120 --width 48
    python train_model.py --resume                 # continue from best_emotion.weights.h5
"""
import os
import shutil
import argparse

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    CSVLogger,
    TerminateOnNaN,
)

# ---------------------------------------------------
# CLI hyper-parameters
# ---------------------------------------------------
parser = argparse.ArgumentParser(description="Train the emotion CNN (fast version)")
parser.add_argument("--epochs", type=int, default=80)
parser.add_argument("--batch-size", type=int, default=128)
parser.add_argument("--lr", type=float, default=None,
                    help="peak learning rate (default 2e-3, or 3e-4 with --resume)")
parser.add_argument("--min-lr", type=float, default=1e-6, help="cosine floor")
parser.add_argument("--warmup-frac", type=float, default=0.05)
parser.add_argument("--weight-decay", type=float, default=0.05, help="AdamW decoupled decay")
parser.add_argument("--width", type=int, default=32,
                    help="base channels; stages use w, 2w, 4w, 8w (32 = fast, 48/64 = stronger)")
parser.add_argument("--dropout-scale", type=float, default=1.0)
parser.add_argument("--label-smoothing", type=float, default=0.1)
parser.add_argument("--mixup-alpha", type=float, default=0.2, help="0 disables MixUp")
parser.add_argument("--erase-prob", type=float, default=0.25, help="random-erasing prob, 0 disables")
parser.add_argument("--cw-power", type=float, default=0.5,
                    help="class-weight strength: 0 = none, 1 = fully balanced, 0.5 = soft")
parser.add_argument("--patience", type=int, default=30)
parser.add_argument("--no-se", dest="use_se", action="store_false",
                    help="disable Squeeze-and-Excitation blocks")
parser.add_argument("--no-amp", action="store_true", help="disable mixed precision")
parser.add_argument("--resume", action="store_true",
                    help="start from existing best_emotion.weights.h5, if present "
                         "(use the same --width / --no-se as the original run)")
parser.set_defaults(use_se=True)
ARGS = parser.parse_args()

LR = ARGS.lr if ARGS.lr is not None else (3e-4 if ARGS.resume else 2e-3)

# ---------------------------------------------------
# Setup
# ---------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "data", "train")
TEST_DIR = os.path.join(BASE_DIR, "data", "test")
SEED = 42
IMG_SIZE = (48, 48)
BATCH_SIZE = ARGS.batch_size
EPOCHS = ARGS.epochs
VAL_SPLIT = 0.2

if not os.path.isdir(DATASET_DIR):
    raise SystemExit(f"Training folder not found: {DATASET_DIR}\n"
                     f"Expected layout: data/train/<class_name>/*.png next to this script.")

tf.keras.utils.set_random_seed(SEED)
np.random.seed(SEED)

gpus = tf.config.list_physical_devices("GPU")
for g in gpus:
    try:
        tf.config.experimental.set_memory_growth(g, True)
    except RuntimeError:
        pass
if gpus and not ARGS.no_amp:
    tf.keras.mixed_precision.set_global_policy("mixed_float16")
    print("Mixed precision: ON (float16 compute)")
else:
    print("Mixed precision: OFF")

# ---------------------------------------------------
# Classes + counts
# ---------------------------------------------------
class_names = sorted(
    d for d in os.listdir(DATASET_DIR) if os.path.isdir(os.path.join(DATASET_DIR, d))
)
if not class_names:
    raise SystemExit(f"No class folders found in: {DATASET_DIR}")
num_classes = len(class_names)
print("Class names:", class_names)

class_counts = np.array(
    [
        len([f for f in os.listdir(os.path.join(DATASET_DIR, c))
             if os.path.isfile(os.path.join(DATASET_DIR, c, f))])
        for c in class_names
    ],
    dtype=np.int64,
)
total_samples = int(class_counts.sum())
if total_samples == 0:
    raise SystemExit("Class folders are empty.")

# Soft class weights, normalised so the average sample weight is 1.
cw = (total_samples / (num_classes * np.maximum(class_counts, 1))) ** ARGS.cw_power
cw = cw / (np.sum(cw * class_counts) / total_samples)
cw_vec = tf.constant(cw, dtype=tf.float32)

print("\nClass counts / weights:")
for i, n in enumerate(class_names):
    print(f"  {n:<10s} {class_counts[i]:>6d}   w={cw[i]:.3f}")

# ---------------------------------------------------
# Data  (unbatched -> cache -> shuffle SAMPLES -> batch)
# ---------------------------------------------------
AUTOTUNE = tf.data.AUTOTUNE


def load_raw(directory, subset=None, shuffle=True):
    kw = dict(
        image_size=IMG_SIZE,
        color_mode="grayscale",
        batch_size=None,              # we batch ourselves, after shuffling samples
        label_mode="categorical",
        class_names=class_names,
        shuffle=shuffle,
    )
    if subset:
        # IMPORTANT: shuffle must be identical (True) for both subsets, otherwise the
        # file lists differ and train/val overlap.
        kw.update(validation_split=VAL_SPLIT, subset=subset, seed=SEED)
    return tf.keras.utils.image_dataset_from_directory(directory, **kw)


def to_u8(x, y):
    return tf.cast(x, tf.uint8), y


def normalize(x, y):
    return tf.cast(x, tf.float32) / 255.0, y


def mixup_batch(x, y):
    """Per-sample MixUp with lambda ~ Beta(alpha, alpha)."""
    a = ARGS.mixup_alpha
    b = tf.shape(x)[0]
    g1 = tf.random.gamma([b], a)
    g2 = tf.random.gamma([b], a)
    lam = g1 / (g1 + g2 + 1e-8)
    idx = tf.random.shuffle(tf.range(b))
    lx = tf.reshape(lam, [-1, 1, 1, 1])
    ly = tf.reshape(lam, [-1, 1])
    return lx * x + (1.0 - lx) * tf.gather(x, idx), ly * y + (1.0 - ly) * tf.gather(y, idx)


def add_sample_weight(x, y):
    return x, y, tf.reduce_sum(y * cw_vec, axis=-1)


train_raw = load_raw(DATASET_DIR, "training")
val_raw = load_raw(DATASET_DIR, "validation")

try:
    n_train = int(train_raw.cardinality().numpy())
except Exception:
    n_train = -1
if n_train <= 0:
    n_train = total_samples - int(VAL_SPLIT * total_samples)

train_ds = (
    train_raw.map(to_u8, num_parallel_calls=AUTOTUNE)
    .cache()
    .shuffle(n_train, seed=SEED, reshuffle_each_iteration=True)
    .batch(BATCH_SIZE, drop_remainder=True)
    .map(normalize, num_parallel_calls=AUTOTUNE)
)
if ARGS.mixup_alpha > 0:
    train_ds = train_ds.map(mixup_batch, num_parallel_calls=AUTOTUNE)
    print(f"MixUp enabled (alpha={ARGS.mixup_alpha})")
train_ds = train_ds.map(add_sample_weight, num_parallel_calls=AUTOTUNE).prefetch(AUTOTUNE)

val_ds = (
    val_raw.map(to_u8, num_parallel_calls=AUTOTUNE)
    .cache()
    .batch(256)
    .map(normalize, num_parallel_calls=AUTOTUNE)
    .prefetch(AUTOTUNE)
)

steps_per_epoch = max(1, n_train // BATCH_SIZE)
total_steps = steps_per_epoch * EPOCHS

# ---------------------------------------------------
# GPU-side augmentation (active only during training)
# ---------------------------------------------------
class RandomErasing(layers.Layer):
    """Cutout: paints a random rectangle with the image mean."""

    def __init__(self, prob=0.25, min_frac=0.1, max_frac=0.35, **kwargs):
        kwargs.setdefault("dtype", "float32")
        super().__init__(**kwargs)
        self.prob, self.min_frac, self.max_frac = prob, min_frac, max_frac

    def call(self, x, training=None):
        if not training or self.prob <= 0:
            return x
        b = tf.shape(x)[0]
        h = tf.shape(x)[1]
        w = tf.shape(x)[2]
        hf, wf = tf.cast(h, tf.float32), tf.cast(w, tf.float32)
        eh = tf.random.uniform([b], self.min_frac, self.max_frac) * hf
        ew = tf.random.uniform([b], self.min_frac, self.max_frac) * wf
        cy = tf.random.uniform([b]) * hf
        cx = tf.random.uniform([b]) * wf
        ys = tf.cast(tf.range(h), tf.float32)[None, :, None]
        xs = tf.cast(tf.range(w), tf.float32)[None, None, :]
        y0, y1 = (cy - eh / 2)[:, None, None], (cy + eh / 2)[:, None, None]
        x0, x1 = (cx - ew / 2)[:, None, None], (cx + ew / 2)[:, None, None]
        mask = (ys >= y0) & (ys < y1) & (xs >= x0) & (xs < x1)
        apply = (tf.random.uniform([b]) < self.prob)[:, None, None]
        mask = (mask & apply)[..., None]
        fill = tf.reduce_mean(x, axis=[1, 2, 3], keepdims=True)
        return tf.where(mask, fill, x)

    def get_config(self):
        cfg = super().get_config()
        cfg.update(prob=self.prob, min_frac=self.min_frac, max_frac=self.max_frac)
        return cfg


def build_augmentation():
    f32 = dict(dtype="float32")
    return tf.keras.Sequential(
        [
            layers.RandomFlip("horizontal", **f32),
            layers.RandomRotation(0.08, **f32),
            layers.RandomTranslation(0.08, 0.08, **f32),
            layers.RandomZoom(0.10, **f32),
            # value_range MUST match the [0,1] input scale (default is 0-255 -> destroys images)
            layers.RandomBrightness(0.15, value_range=(0.0, 1.0), **f32),
            layers.RandomContrast(0.20, **f32),
            RandomErasing(ARGS.erase_prob),
        ],
        name="augmentation",
    )


# ---------------------------------------------------
# Model: small ResNet + Squeeze-and-Excitation
# ---------------------------------------------------
def se_block(x, ratio=8):
    c = x.shape[-1]
    s = layers.GlobalAveragePooling2D()(x)
    s = layers.Dense(max(8, c // ratio), activation="relu")(s)
    s = layers.Dense(c, activation="sigmoid")(s)
    s = layers.Reshape((1, 1, c))(s)
    return layers.Multiply()([x, s])


def conv_bn(x, filters, k=3, stride=1, act=True):
    x = layers.Conv2D(filters, k, strides=stride, padding="same", use_bias=False,
                      kernel_initializer="he_normal")(x)
    x = layers.BatchNormalization()(x)
    return layers.Activation("relu")(x) if act else x


def res_block(x, filters, stride, use_se):
    y = conv_bn(x, filters, 3, stride)
    y = conv_bn(y, filters, 3, 1, act=False)
    if use_se:
        y = se_block(y)
    shortcut = x
    if stride != 1 or x.shape[-1] != filters:
        shortcut = conv_bn(x, filters, 1, stride, act=False)
    return layers.Activation("relu")(layers.Add()([y, shortcut]))


def build_backbone(num_classes, width, dropout_scale, use_se):
    inp = layers.Input((IMG_SIZE[0], IMG_SIZE[1], 1), name="image")  # expects [0,1]
    x = conv_bn(inp, width, 3, 1)
    for mult, stride in [(1, 1), (2, 2), (4, 2), (8, 2)]:
        x = res_block(x, width * mult, stride, use_se)
        x = res_block(x, width * mult, 1, use_se)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(min(0.9, 0.4 * dropout_scale))(x)
    x = layers.Dense(num_classes, dtype="float32")(x)
    out = layers.Activation("softmax", dtype="float32")(x)
    return tf.keras.Model(inp, out, name="emotion_net")


backbone = build_backbone(num_classes, ARGS.width, ARGS.dropout_scale, ARGS.use_se)

# Training model = augmentation + backbone (they share weights).
t_in = layers.Input((IMG_SIZE[0], IMG_SIZE[1], 1))
train_model = tf.keras.Model(t_in, backbone(build_augmentation()(t_in)), name="train_wrapper")
print(f"\nBackbone params: {backbone.count_params():,}   "
      f"SE: {'ON' if ARGS.use_se else 'OFF'}   width: {ARGS.width}")

# ---------------------------------------------------
# LR schedule + optimizer + loss
# ---------------------------------------------------
@tf.keras.utils.register_keras_serializable(package="Emotion")
class WarmupCosineDecay(tf.keras.optimizers.schedules.LearningRateSchedule):
    """Linear warmup, then cosine decay to min_lr."""

    def __init__(self, base_lr, warmup_steps, total_steps, min_lr):
        super().__init__()
        self.base_lr = float(base_lr)
        self.warmup_steps = float(max(1, warmup_steps))
        self.total_steps = float(max(2, total_steps))
        self.min_lr = float(min_lr)

    def __call__(self, step):
        step = tf.cast(step, tf.float32)
        warm = self.base_lr * (step + 1.0) / self.warmup_steps
        denom = tf.maximum(1.0, self.total_steps - self.warmup_steps)
        progress = tf.clip_by_value((step - self.warmup_steps) / denom, 0.0, 1.0)
        cos = self.min_lr + 0.5 * (self.base_lr - self.min_lr) * (1.0 + tf.cos(np.pi * progress))
        return tf.where(step < self.warmup_steps, warm, cos)

    def get_config(self):
        return {"base_lr": self.base_lr, "warmup_steps": self.warmup_steps,
                "total_steps": self.total_steps, "min_lr": self.min_lr}


warmup_steps = int(ARGS.warmup_frac * total_steps)
lr_schedule = WarmupCosineDecay(LR, warmup_steps, total_steps, ARGS.min_lr)
print(f"LR: warmup {warmup_steps} steps -> cosine over {total_steps} steps "
      f"(peak {LR}, floor {ARGS.min_lr}); {steps_per_epoch} steps/epoch")


def make_optimizer():
    for factory in (
        lambda: tf.keras.optimizers.AdamW,
        lambda: tf.keras.optimizers.experimental.AdamW,
    ):
        try:
            opt = factory()(learning_rate=lr_schedule, weight_decay=ARGS.weight_decay, clipnorm=1.0)
            try:  # no decay on BatchNorm / bias parameters
                opt.exclude_from_weight_decay(var_names=["gamma", "beta", "bias"])
            except Exception:
                pass
            return opt
        except (AttributeError, TypeError):
            continue
    print("AdamW not available in this TF version -> falling back to Adam")
    return tf.keras.optimizers.Adam(learning_rate=lr_schedule, clipnorm=1.0)


train_model.compile(
    optimizer=make_optimizer(),
    loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=ARGS.label_smoothing),
    metrics=["accuracy"],
    jit_compile=False,   # augmentation ops are not XLA-compatible
)

# ---------------------------------------------------
# Safe export helpers
# ---------------------------------------------------
BEST_WEIGHTS = os.path.join(BASE_DIR, "best_emotion.weights.h5")
BEST_KERAS = os.path.join(BASE_DIR, "best_emotion_model.keras")
FINAL_KERAS = os.path.join(BASE_DIR, "live_emotion_detection_model.keras")


def save_clean(model, path):
    """Atomically write a clean inference model (no optimizer) to `path` (.keras)."""
    tmp = path[: -len(".keras")] + ".tmp.keras"
    try:
        model.save(tmp, include_optimizer=False)
    except (TypeError, ValueError):      # some Keras versions reject include_optimizer
        model.save(tmp)
    os.replace(tmp, path)


def export_inference_model(model, announce=""):
    """Write best + live model files. Never raises: a locked file must not kill training."""
    try:
        save_clean(model, BEST_KERAS)
        tmp_live = FINAL_KERAS[: -len(".keras")] + ".tmp.keras"
        shutil.copyfile(BEST_KERAS, tmp_live)
        os.replace(tmp_live, FINAL_KERAS)
        if announce:
            print(announce)
        return True
    except Exception as e:  # PermissionError on Windows if the app has the file open, disk full, ...
        print(f"\n[warning] could not export model files ({type(e).__name__}: {e}).\n"
              f"          Training continues; best weights are still in {os.path.basename(BEST_WEIGHTS)}.")
        return False


class BestKerasExporter(tf.keras.callbacks.Callback):
    """Export the clean .keras model each time val_accuracy reaches a new best."""

    def __init__(self, export_model, initial_best=0.0):
        super().__init__()
        self.export_model = export_model   # inference backbone (shares weights with train_model)
        self.best = float(initial_best)

    def on_epoch_end(self, epoch, logs=None):
        acc = float((logs or {}).get("val_accuracy", 0.0))
        if acc > self.best:
            self.best = acc
            export_inference_model(
                self.export_model,
                f"\nExported clean inference model (val_accuracy={acc:.4f}) -> "
                f"{os.path.basename(BEST_KERAS)}",
            )


# ---------------------------------------------------
# Optional resume (keeps the previous best instead of overwriting it)
# ---------------------------------------------------
resumed_acc = None
if ARGS.resume:
    if os.path.exists(BEST_WEIGHTS):
        try:
            train_model.load_weights(BEST_WEIGHTS)
            resumed_acc = float(train_model.evaluate(val_ds, verbose=0, return_dict=True)["accuracy"])
            print(f"Resumed from {os.path.basename(BEST_WEIGHTS)} "
                  f"(val_accuracy={resumed_acc:.4f}, peak LR {LR})")
        except Exception as e:
            print(f"--resume requested but weights unusable ({type(e).__name__}: {e}).\n"
                  f"Use the same --width/--no-se as the original run. Starting fresh.")
            resumed_acc = None
    else:
        print("--resume requested but no best_emotion.weights.h5 found; starting fresh.")

# ---------------------------------------------------
# Callbacks
# ---------------------------------------------------
ckpt_kwargs = {}
if resumed_acc is not None:
    ckpt_kwargs["initial_value_threshold"] = resumed_acc

callbacks = [
    EarlyStopping(monitor="val_accuracy", mode="max", patience=ARGS.patience, verbose=1),
    ModelCheckpoint(BEST_WEIGHTS, monitor="val_accuracy", mode="max",
                    save_best_only=True, save_weights_only=True, verbose=1, **ckpt_kwargs),
    BestKerasExporter(backbone, initial_best=resumed_acc or 0.0),
    CSVLogger(os.path.join(BASE_DIR, "training_log.csv"), append=bool(resumed_acc)),
    TerminateOnNaN(),
]

# ---------------------------------------------------
# Train
# ---------------------------------------------------
print("\nStarting training...\n")
history = train_model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=callbacks,
    verbose=1,
)

# Always evaluate / export the BEST epoch, not the last one.
if os.path.exists(BEST_WEIGHTS):
    try:
        train_model.load_weights(BEST_WEIGHTS)
    except Exception as e:
        print(f"Warning: could not load best weights ({type(e).__name__}: {e}); using last-epoch weights")

best_val_acc = max(history.history["val_accuracy"])
print("\nTraining completed.")
print(f"Best Validation Accuracy (no TTA): {best_val_acc:.4f}")
print("(Training accuracy is measured on MixUp/augmented data, so it is not comparable.)")

# ---------------------------------------------------
# Final export (clean inference model: input 48x48x1, values in [0,1])
# ---------------------------------------------------
if export_inference_model(backbone):
    print(f"\nSaved inference model:\n  {BEST_KERAS}\n  {FINAL_KERAS}")
    print("Load with: tf.keras.models.load_model(path, compile=False)")

# ---------------------------------------------------
# Evaluation with flip test-time augmentation
# ---------------------------------------------------
def evaluate(ds, title):
    y_true, p_plain, p_tta = [], [], []
    for x, y in ds:
        p1 = backbone(x, training=False)
        p2 = backbone(tf.image.flip_left_right(x), training=False)
        y_true.append(np.argmax(y.numpy(), axis=1))
        p_plain.append(p1.numpy())
        p_tta.append(0.5 * (p1.numpy() + p2.numpy()))
    y_true = np.concatenate(y_true)
    pred_plain = np.argmax(np.concatenate(p_plain), axis=1)
    pred_tta = np.argmax(np.concatenate(p_tta), axis=1)
    print(f"\n[{title}] samples={len(y_true)}  "
          f"acc={np.mean(pred_plain == y_true):.4f}  acc+flipTTA={np.mean(pred_tta == y_true):.4f}")
    for i, name in enumerate(class_names):
        m = y_true == i
        if m.any():
            print(f"    {name:<10s} {np.mean(pred_tta[m] == i):.3f}  (n={int(m.sum())})")


evaluate(val_ds, "validation")

if os.path.isdir(TEST_DIR):
    test_ds = (
        load_raw(TEST_DIR, shuffle=False)
        .map(to_u8, num_parallel_calls=AUTOTUNE)
        .batch(256)
        .map(normalize, num_parallel_calls=AUTOTUNE)
        .prefetch(AUTOTUNE)
    )
    evaluate(test_ds, "TEST (held-out, the number to trust)")
else:
    print(f"\nNo test folder at {TEST_DIR} - skipping held-out evaluation.")