import os
import glob
import json
import random
import numpy as np
import librosa
import cv2
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from vibe_model import build_vibe_model

# Mapping GTZAN genres to Vibes
# 0: Happy, 1: Sad, 2: Energetic
GENRE_TO_VIBE = {
    'pop': 0,
    'disco': 0,
    'country': 0,
    'reggae': 0,
    'blues': 1,
    'classical': 1,
    'jazz': 1,
    'rock': 2,
    'metal': 2,
    'hiphop': 2,
}
VIBE_LABELS = ["Happy", "Sad", "Energetic"]

DATA_DIR = os.path.join("Data", "genres_original")

SEED = 42
SAMPLE_RATE = 22050
WINDOW_SECONDS = 5
N_MELS = 128
IMG_SIZE = (128, 128)
HOP_LENGTH = 512
N_FFT = 2048
STATS_FEATURE_DIM = 5
WINDOWS_PER_TRACK = 6
CACHE_PATH = "feature_cache.npz"
TRIM_TOP_DB = 20
FINAL_MODEL_PATH = "advanced_vibe_meter.keras"
BEST_MODEL_PATH = "advanced_vibe_meter_best.keras"


def set_seed(seed: int = SEED):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def collect_tracks():
    track_paths = []
    labels = []

    print("Indexing dataset...")
    for genre, vibe_label in GENRE_TO_VIBE.items():
        genre_dir = os.path.join(DATA_DIR, genre)
        if not os.path.exists(genre_dir):
            print(f"Warning: Directory {genre_dir} not found. Skipping.")
            continue

        files = sorted(glob.glob(os.path.join(genre_dir, "*.wav")))
        print(f"{genre}: {len(files)} files -> class {vibe_label}")

        for filepath in files:
            track_paths.append(filepath)
            labels.append(vibe_label)

    if not track_paths:
        raise ValueError("No valid audio files found. Please ensure the dataset path is correct.")

    return np.array(track_paths), np.array(labels, dtype=np.int64)


def mel_feature_from_window(window: np.ndarray, sr: int) -> np.ndarray:
    mel = librosa.feature.melspectrogram(
        y=window,
        sr=sr,
        n_mels=N_MELS,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        power=2.0,
    )
    mel_db = librosa.power_to_db(mel, ref=np.max).astype(np.float32)
    delta = librosa.feature.delta(mel_db).astype(np.float32)
    delta2 = librosa.feature.delta(mel_db, order=2).astype(np.float32)

    # Resize each channel and stack as an RGB-like tensor.
    mel_resized = cv2.resize(mel_db, IMG_SIZE).astype(np.float32)
    delta_resized = cv2.resize(delta, IMG_SIZE).astype(np.float32)
    delta2_resized = cv2.resize(delta2, IMG_SIZE).astype(np.float32)

    feature_stack = np.stack([mel_resized, delta_resized, delta2_resized], axis=-1)

    # Channel-wise normalization improves robustness to recording setup.
    channel_means = np.mean(feature_stack, axis=(0, 1), keepdims=True)
    channel_stds = np.std(feature_stack, axis=(0, 1), keepdims=True) + 1e-6
    return (feature_stack - channel_means) / channel_stds


def handcrafted_stats_from_window(window: np.ndarray, sr: int) -> np.ndarray:
    centroid = librosa.feature.spectral_centroid(y=window, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH)
    rolloff = librosa.feature.spectral_rolloff(
        y=window,
        sr=sr,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        roll_percent=0.85,
    )
    rms = librosa.feature.rms(y=window, frame_length=N_FFT, hop_length=HOP_LENGTH)
    zcr = librosa.feature.zero_crossing_rate(y=window, frame_length=N_FFT, hop_length=HOP_LENGTH)

    tempo, _ = librosa.beat.beat_track(y=window, sr=sr, hop_length=HOP_LENGTH)
    tempo_value = float(np.asarray(tempo).reshape(-1)[0]) if np.asarray(tempo).size else 0.0

    nyquist = (sr / 2.0) + 1e-6
    stats = np.array(
        [
            float(np.mean(centroid) / nyquist),
            float(np.mean(rolloff) / nyquist),
            float(np.mean(rms)),
            float(np.mean(zcr)),
            float(min(max(tempo_value, 0.0), 300.0) / 300.0),
        ],
        dtype=np.float32,
    )
    return stats


def get_candidate_starts(total_samples: int, sr: int):
    window_samples = int(sr * WINDOW_SECONDS)
    if total_samples <= window_samples:
        return [0], window_samples

    max_start = total_samples - window_samples
    hop_samples = int(sr)  # 1-second hop
    starts = list(range(0, max_start + 1, hop_samples))
    if starts[-1] != max_start:
        starts.append(max_start)
    return starts, window_samples


def extract_track_features(
    filepath: str,
    windows_per_track: int,
    random_sampling: bool,
    rng: np.random.Generator,
):
    try:
        y, sr = librosa.load(filepath, sr=SAMPLE_RATE, mono=True)
    except Exception:
        return []

    # Remove leading/trailing silence and normalize loudness.
    y_trimmed, _ = librosa.effects.trim(y, top_db=TRIM_TOP_DB)
    if len(y_trimmed) > 0:
        y = y_trimmed
    y = librosa.util.normalize(y)

    starts, window_samples = get_candidate_starts(len(y), sr)
    if not starts:
        return []

    energy_and_start = []
    for start in starts:
        window = y[start:start + window_samples]
        energy = float(np.mean(np.square(window)))
        energy_and_start.append((energy, start))

    energy_and_start.sort(key=lambda item: item[0], reverse=True)
    top_windows = energy_and_start[:windows_per_track]
    chosen_starts = sorted([start for _, start in top_windows])

    features = []
    for start in chosen_starts:
        window = y[start:start + window_samples]
        if len(window) < window_samples:
            window = np.pad(window, (0, window_samples - len(window)))

        spectrogram_feat = mel_feature_from_window(window, sr)
        stats_feat = handcrafted_stats_from_window(window, sr)
        features.append((spectrogram_feat, stats_feat))

    return features


def build_feature_dataset(
    track_paths: np.ndarray,
    track_labels: np.ndarray,
    windows_per_track: int,
    random_sampling: bool,
    seed: int,
):
    rng = np.random.default_rng(seed)
    X_img = []
    X_stats = []
    y = []

    total = len(track_paths)
    for i, (filepath, label) in enumerate(zip(track_paths, track_labels), start=1):
        features = extract_track_features(filepath, windows_per_track, random_sampling, rng)
        if not features:
            continue

        for spectrogram_feat, stats_feat in features:
            X_img.append(spectrogram_feat)
            X_stats.append(stats_feat)
            y.append(int(label))

        if i % 100 == 0 or i == total:
            print(f"Processed {i}/{total} tracks")

    if not X_img:
        raise ValueError("Feature extraction failed. No training samples were created.")

    return (
        np.array(X_img, dtype=np.float32),
        np.array(X_stats, dtype=np.float32),
        np.array(y, dtype=np.int64),
    )

if __name__ == "__main__":
    set_seed(SEED)

    rebuild_cache = False
    required_cache_keys = {"X_img_all", "X_stats_all", "y_all"}

    if os.path.exists(CACHE_PATH):
        print(f"Loading cached features from {CACHE_PATH}...")
        with np.load(CACHE_PATH) as cached:
            cache_keys = set(cached.files)
            if required_cache_keys.issubset(cache_keys):
                X_img_cached = cached["X_img_all"]
                X_stats_cached = cached["X_stats_all"]
                y_cached = cached["y_all"]

                if (
                    X_img_cached.ndim == 4
                    and X_img_cached.shape[-1] == 3
                    and X_stats_cached.ndim == 2
                    and X_stats_cached.shape[1] == STATS_FEATURE_DIM
                ):
                    X_img_all = X_img_cached
                    X_stats_all = X_stats_cached
                    y_all = y_cached
                else:
                    print("Cache shape mismatch detected. Rebuilding features...")
                    rebuild_cache = True
            else:
                print("Legacy cache detected. Rebuilding features...")
                rebuild_cache = True
    else:
        rebuild_cache = True

    if rebuild_cache:
        if os.path.exists(CACHE_PATH):
            os.remove(CACHE_PATH)
        track_paths, track_labels = collect_tracks()
        print("Extracting full feature set...")
        X_img_all, X_stats_all, y_all = build_feature_dataset(
            track_paths,
            track_labels,
            windows_per_track=WINDOWS_PER_TRACK,
            random_sampling=True,
            seed=SEED,
        )
        np.savez_compressed(CACHE_PATH, X_img_all=X_img_all, X_stats_all=X_stats_all, y_all=y_all)
        print(f"Saved feature cache to {CACHE_PATH}")

    (
        X_img_train,
        X_img_val,
        X_stats_train,
        X_stats_val,
        y_train,
        y_val,
    ) = train_test_split(
        X_img_all,
        X_stats_all,
        y_all,
        test_size=0.2,
        random_state=SEED,
        stratify=y_all,
    )

    print(f"Train samples: {X_img_train.shape[0]}, Validation samples: {X_img_val.shape[0]}")
    print(f"Spectrogram input shape: {X_img_train.shape[1:]}")
    print(f"Stats input shape: {X_stats_train.shape[1:]}")
    unique_labels, label_counts = np.unique(y_train, return_counts=True)
    print(f"Unique labels in y_train: {unique_labels}")
    class_count_map = {VIBE_LABELS[int(label)]: int(count) for label, count in zip(unique_labels, label_counts)}
    print(f"Class counts in y_train: {class_count_map}")

    classes = np.unique(y_train)
    class_weights_values = compute_class_weight(class_weight='balanced', classes=classes, y=y_train)
    class_weights = {int(cls): float(w) for cls, w in zip(classes, class_weights_values)}
    print(f"Class weights: {class_weights}")

    model = build_vibe_model(
        input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3),
        stats_dim=STATS_FEATURE_DIM,
        num_classes=3,
    )

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            BEST_MODEL_PATH,
            monitor="val_accuracy",
            mode="max",
            save_best_only=True,
            verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            mode="max",
            patience=6,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            min_lr=1e-6,
            verbose=1,
        ),
    ]

    print("Starting training...")
    history = model.fit(
        [X_img_train, X_stats_train],
        y_train,
        epochs=30,
        batch_size=32,
        validation_data=([X_img_val, X_stats_val], y_val),
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=1,
    )

    best_model = tf.keras.models.load_model(BEST_MODEL_PATH)
    val_loss, val_accuracy = best_model.evaluate([X_img_val, X_stats_val], y_val, verbose=0)

    val_probs = best_model.predict([X_img_val, X_stats_val], verbose=0)
    mean_probs = np.mean(val_probs, axis=0)
    mean_probs_map = {VIBE_LABELS[i]: float(mean_probs[i]) for i in range(len(VIBE_LABELS))}
    print(f"Mean prob per class: {mean_probs_map}")

    y_pred = np.argmax(val_probs, axis=1)
    cm = confusion_matrix(y_val, y_pred, labels=[0, 1, 2])
    print("Confusion matrix (rows=true, cols=pred):")
    print(cm)

    report_text = classification_report(
        y_val,
        y_pred,
        labels=[0, 1, 2],
        target_names=VIBE_LABELS,
        digits=4,
        zero_division=0,
    )
    print(report_text)
    with open("validation_classification_report.txt", "w", encoding="utf-8") as f:
        f.write(report_text)

    best_model.save(FINAL_MODEL_PATH)
    with open("label_map.json", "w", encoding="utf-8") as f:
        json.dump(VIBE_LABELS, f)

    report = {
        "val_accuracy": float(val_accuracy),
        "val_loss": float(val_loss),
        "train_samples": int(X_img_train.shape[0]),
        "val_samples": int(X_img_val.shape[0]),
        "epochs_ran": len(history.history.get("loss", [])),
        "class_counts": class_count_map,
        "mean_prob_per_class": mean_probs_map,
        "confusion_matrix": cm.tolist(),
    }
    with open("training_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"Final validation accuracy: {val_accuracy * 100:.2f}%")
    if val_accuracy < 0.90:
        print("Validation accuracy is below 90%. Consider longer training or stronger augmentation.")
    else:
        print("Reached target: validation accuracy >= 90%")
