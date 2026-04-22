from contextlib import asynccontextmanager
import json
import os
import tempfile
from typing import Dict, List

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
import cv2
import librosa
import numpy as np
import tensorflow as tf


SAMPLE_RATE = 22050
WINDOW_SECONDS = 5
WINDOW_SAMPLES = SAMPLE_RATE * WINDOW_SECONDS
HOP_SECONDS = 1
HOP_SAMPLES = SAMPLE_RATE * HOP_SECONDS
USE_FULL_AUDIO_WINDOWS = os.environ.get("USE_FULL_AUDIO_WINDOWS", "1") != "0"
MAX_WINDOWS = int(os.environ.get("MAX_WINDOWS", "6"))
N_MELS = 128
HOP_LENGTH = 512
N_FFT = 2048
TRIM_TOP_DB = 20
EPSILON = 1e-6
STATS_FEATURE_DIM_DEFAULT = 5
DEFAULT_LABELS = ["Happy", "Sad", "Energetic"]
MODEL_CANDIDATES = [
    "advanced_vibe_meter.keras",
    "advanced_vibe_meter_best.keras",
    "vibe_meter_final.h5",
]
MODEL_PATH = os.environ.get("VIBE_MODEL_PATH") or next(
    (candidate for candidate in MODEL_CANDIDATES if os.path.exists(candidate)),
    MODEL_CANDIDATES[0],
)
LABEL_MAP_PATH = "label_map.json"


def load_labels(num_classes: int) -> List[str]:
    labels = list(DEFAULT_LABELS)

    if os.path.exists(LABEL_MAP_PATH):
        try:
            with open(LABEL_MAP_PATH, "r", encoding="utf-8") as file:
                loaded = json.load(file)
            if isinstance(loaded, list) and loaded:
                labels = [str(item) for item in loaded]
        except Exception:
            pass

    if len(labels) < num_classes:
        labels.extend([f"Class {i + 1}" for i in range(len(labels), num_classes)])
    elif len(labels) > num_classes:
        labels = labels[:num_classes]

    return labels


def select_window_starts(y: np.ndarray) -> List[int]:
    if len(y) <= WINDOW_SAMPLES:
        return [0]

    max_start = len(y) - WINDOW_SAMPLES
    starts = list(range(0, max_start + 1, HOP_SAMPLES))
    if starts[-1] != max_start:
        starts.append(max_start)

    # Full-file mode processes all 5-second windows across the track.
    if USE_FULL_AUDIO_WINDOWS:
        return starts

    if MAX_WINDOWS <= 0:
        return starts

    energy_and_starts = []
    for start in starts:
        window = y[start:start + WINDOW_SAMPLES]
        energy = float(np.mean(np.square(window)))
        energy_and_starts.append((energy, start))

    energy_and_starts.sort(key=lambda item: item[0], reverse=True)
    selected = [start for _, start in energy_and_starts[:MAX_WINDOWS]]
    selected.sort()
    return selected


def build_feature_from_window(
    window: np.ndarray,
    target_width: int,
    target_height: int,
    target_channels: int,
) -> np.ndarray:
    mel = librosa.feature.melspectrogram(
        y=window,
        sr=SAMPLE_RATE,
        n_mels=N_MELS,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        power=2.0,
    )
    mel_db = librosa.power_to_db(mel, ref=np.max).astype(np.float32)
    image_size = (target_width, target_height)

    mel_resized = cv2.resize(mel_db, image_size).astype(np.float32)

    if target_channels >= 3:
        delta = librosa.feature.delta(mel_db).astype(np.float32)
        delta2 = librosa.feature.delta(mel_db, order=2).astype(np.float32)

        delta_resized = cv2.resize(delta, image_size).astype(np.float32)
        delta2_resized = cv2.resize(delta2, image_size).astype(np.float32)

        feature_stack = np.stack([mel_resized, delta_resized, delta2_resized], axis=-1)
        channel_means = np.mean(feature_stack, axis=(0, 1), keepdims=True)
        channel_stds = np.std(feature_stack, axis=(0, 1), keepdims=True) + EPSILON
        normalized = (feature_stack - channel_means) / channel_stds

        if target_channels == 3:
            return normalized

        padding = np.zeros((target_height, target_width, target_channels - 3), dtype=np.float32)
        return np.concatenate([normalized, padding], axis=-1)

    mean = float(np.mean(mel_resized))
    std = float(np.std(mel_resized)) + EPSILON
    normalized = (mel_resized - mean) / std
    return np.expand_dims(normalized, axis=-1)


def handcrafted_stats_from_window(window: np.ndarray) -> np.ndarray:
    centroid = librosa.feature.spectral_centroid(
        y=window,
        sr=SAMPLE_RATE,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
    )
    rolloff = librosa.feature.spectral_rolloff(
        y=window,
        sr=SAMPLE_RATE,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        roll_percent=0.85,
    )
    rms = librosa.feature.rms(y=window, frame_length=N_FFT, hop_length=HOP_LENGTH)
    zcr = librosa.feature.zero_crossing_rate(y=window, frame_length=N_FFT, hop_length=HOP_LENGTH)
    tempo, _ = librosa.beat.beat_track(y=window, sr=SAMPLE_RATE, hop_length=HOP_LENGTH)
    tempo_value = float(np.asarray(tempo).reshape(-1)[0]) if np.asarray(tempo).size else 0.0

    nyquist = (SAMPLE_RATE / 2.0) + EPSILON
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


def preprocess_audio(
    file_path: str,
    target_width: int,
    target_height: int,
    target_channels: int,
) -> tuple[np.ndarray, np.ndarray]:
    y, _ = librosa.load(file_path, sr=SAMPLE_RATE, mono=True)

    # Match training-time audio cleanup before window sampling.
    y_trimmed, _ = librosa.effects.trim(y, top_db=TRIM_TOP_DB)
    if len(y_trimmed) > 0:
        y = y_trimmed
    y = librosa.util.normalize(y)

    starts = select_window_starts(y)
    image_features = []
    stats_features = []
    for start in starts:
        window = y[start:start + WINDOW_SAMPLES]
        if len(window) < WINDOW_SAMPLES:
            window = np.pad(window, (0, WINDOW_SAMPLES - len(window)))

        image_features.append(
            build_feature_from_window(
                window,
                target_width=target_width,
                target_height=target_height,
                target_channels=target_channels,
            )
        )
        stats_features.append(handcrafted_stats_from_window(window))

    return (
        np.array(image_features, dtype=np.float32),
        np.array(stats_features, dtype=np.float32),
    )


def validate_upload(file: UploadFile) -> None:
    filename = (file.filename or "").lower()
    content_type = (file.content_type or "").lower()

    if not filename.endswith(".wav"):
        raise HTTPException(status_code=400, detail="Only .wav files are supported.")

    if content_type and not (
        "wav" in content_type
        or content_type.startswith("audio/")
        or content_type == "application/octet-stream"
    ):
        raise HTTPException(status_code=400, detail="Uploaded file is not recognized as audio/wav.")


@asynccontextmanager
async def lifespan(app: FastAPI):
    if not os.path.exists(MODEL_PATH):
        candidates = ", ".join(MODEL_CANDIDATES)
        raise RuntimeError(
            f"Model file not found: {MODEL_PATH}. Place one of [{candidates}] in the project root or set VIBE_MODEL_PATH."
        )

    model = tf.keras.models.load_model(MODEL_PATH)
    model_input_shape = model.input_shape
    expects_stats = False
    stats_dim = STATS_FEATURE_DIM_DEFAULT

    if isinstance(model_input_shape, list):
        image_input_shape = None
        stats_input_shape = None
        for shape in model_input_shape:
            if len(shape) == 4:
                image_input_shape = shape
            elif len(shape) == 2:
                stats_input_shape = shape

        if image_input_shape is None:
            raise RuntimeError("Unable to locate 4D spectrogram input in multi-input model.")

        if stats_input_shape is not None:
            expects_stats = True
            stats_dim = int(stats_input_shape[1] or STATS_FEATURE_DIM_DEFAULT)
        input_shape = image_input_shape
    else:
        input_shape = model_input_shape

    if len(input_shape) != 4:
        raise RuntimeError(
            f"Expected 4D spectrogram input shape (batch, height, width, channels), got {input_shape}."
        )

    target_height = int(input_shape[1] or 128)
    target_width = int(input_shape[2] or 128)
    target_channels = int(input_shape[3] or 1)
    if target_channels < 1:
        raise RuntimeError(f"Invalid model channel count: {target_channels}")

    output_shape = model.output_shape
    if isinstance(output_shape, list):
        raise RuntimeError("Loaded model has multiple outputs. Expected a single-output classifier.")

    num_classes = int(output_shape[-1] or len(DEFAULT_LABELS))
    labels = load_labels(num_classes)

    app.state.model = model
    app.state.model_input_shape = input_shape
    app.state.target_height = target_height
    app.state.target_width = target_width
    app.state.target_channels = target_channels
    app.state.expects_stats = expects_stats
    app.state.stats_dim = stats_dim
    app.state.labels = labels
    app.state.num_classes = num_classes

    if target_channels == 3 and expects_stats:
        mode = "3-channel+stats parity mode"
    elif target_channels == 3:
        mode = "3-channel image-only mode"
    else:
        mode = "single-channel compatibility mode"

    if USE_FULL_AUDIO_WINDOWS:
        window_mode = "full-file windows"
    else:
        window_mode = f"top-{MAX_WINDOWS} energy windows"

    print(
        f"Loaded model '{MODEL_PATH}' with input shape {input_shape}, "
        f"classes={num_classes}, preprocessing={mode}, window_sampling={window_mode}, "
        f"expects_stats={expects_stats}"
    )

    yield


app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def health() -> Dict[str, str]:
    return {"message": "Vibe Meter API is Live"}


@app.get("/app", response_class=HTMLResponse)
async def web_app() -> str:
    with open("index.html", "r", encoding="utf-8") as file:
        return file.read()


@app.post("/predict")
async def predict_vibe(file: UploadFile = File(...)):
    validate_upload(file)

    payload = await file.read()
    if not payload:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as buffer:
            buffer.write(payload)
            temp_path = buffer.name

        image_batch, stats_batch = preprocess_audio(
            temp_path,
            target_width=app.state.target_width,
            target_height=app.state.target_height,
            target_channels=app.state.target_channels,
        )
    except HTTPException:
        raise
    except Exception as exc:
        error_name = exc.__class__.__name__
        raise HTTPException(
            status_code=400,
            detail=f"Unable to decode/process audio ({error_name}). Upload a valid .wav file.",
        )
    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)

    if image_batch.ndim != 4:
        raise HTTPException(status_code=500, detail=f"Unexpected preprocessed rank: {image_batch.shape}")

    if image_batch.shape[1] != app.state.target_height:
        raise HTTPException(
            status_code=500,
            detail=f"Height mismatch: {image_batch.shape[1]} vs expected {app.state.target_height}.",
        )

    if image_batch.shape[2] != app.state.target_width:
        raise HTTPException(
            status_code=500,
            detail=f"Width mismatch: {image_batch.shape[2]} vs expected {app.state.target_width}.",
        )

    if image_batch.shape[3] != app.state.target_channels:
        raise HTTPException(
            status_code=500,
            detail=(
                "Channel mismatch between preprocessing and model input: "
                f"{image_batch.shape[3]} vs expected {app.state.target_channels}."
            ),
        )

    if app.state.expects_stats:
        if stats_batch.ndim != 2:
            raise HTTPException(status_code=500, detail=f"Unexpected stats rank: {stats_batch.shape}")

        if stats_batch.shape[0] != image_batch.shape[0]:
            raise HTTPException(
                status_code=500,
                detail=(
                    "Stats batch size mismatch: "
                    f"{stats_batch.shape[0]} vs image windows {image_batch.shape[0]}."
                ),
            )

        if stats_batch.shape[1] != app.state.stats_dim:
            raise HTTPException(
                status_code=500,
                detail=f"Stats feature size mismatch: {stats_batch.shape[1]} vs expected {app.state.stats_dim}.",
            )

        model_inputs = [image_batch, stats_batch]
    else:
        model_inputs = image_batch

    prediction_batch = app.state.model.predict(model_inputs, verbose=0)
    if prediction_batch.ndim != 2:
        raise HTTPException(status_code=500, detail="Model returned an unexpected output rank.")

    probabilities = np.mean(prediction_batch, axis=0).astype(np.float32)

    if probabilities.ndim != 1:
        raise HTTPException(status_code=500, detail="Model returned an unexpected output shape.")

    if probabilities.shape[0] != app.state.num_classes:
        raise HTTPException(
            status_code=500,
            detail=(
                "Model class count mismatch. "
                f"Expected {app.state.num_classes} classes, got {probabilities.shape[0]}."
            ),
        )

    total = float(np.sum(probabilities))
    if total <= 0.0:
        raise HTTPException(status_code=500, detail="Model returned invalid probabilities.")

    if not np.isclose(total, 1.0, atol=1e-2):
        probabilities = probabilities / total

    top_index = int(np.argmax(probabilities))
    labels = app.state.labels
    distribution = {labels[i]: float(probabilities[i]) for i in range(len(labels))}
    confidence = round(float(probabilities[top_index]) * 100.0, 2)

    return {
        "vibe": labels[top_index],
        "confidence": confidence,
        "distribution": distribution,
    }
