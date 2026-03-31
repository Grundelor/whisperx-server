"""WhisperX Transcription Server with Pyannote Diarization.

FastAPI server that accepts audio files and returns transcription
with speaker diarization in JSON format.

Optimized for Russian legal/courtroom audio:
- Uses bond005/whisper-podlodka-turbo (fine-tuned for Russian)
- Higher beam_size for accuracy
- VAD filtering to handle pauses and background noise
- initial_prompt support for domain context (legal terms, names)

Endpoints:
    GET  /health     - Health check (for heartbeat monitoring)
    POST /transcribe - Upload audio, get transcription + diarization

# TODO: Evaluate NVIDIA Canary 1B v2 (nvidia/canary-1b-v2) as an alternative.
#       Canary v2 supports 25 languages incl. Russian, has encoder-decoder
#       architecture with FastConformer, and built-in punctuation/capitalization.
#       Would require switching from WhisperX to NeMo toolkit — full server rewrite.
#       See: https://huggingface.co/nvidia/canary-1b-v2
"""

import os
import sys
import time
import uuid
import logging
import tempfile
from pathlib import Path
from contextlib import asynccontextmanager
from typing import Optional

import torch
import whisperx
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
import uvicorn

# ── Logging ───────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
log = logging.getLogger("whisperx-server")

# ── Configuration ─────────────────────────────────────────────────────

# Model: bond005/whisper-podlodka-turbo — fine-tuned for Russian ASR
# Pre-converted to faster-whisper format: bzikst/faster-whisper-podlodka-turbo
# Benchmarks vs openai/whisper-large-v3-turbo on Russian:
#   - golos_farfield: 11.61% vs 20.12% WER (-42%)
#   - golos_crowd:    11.85% vs 14.55% WER (-19%)
#   - common_voice:    5.22% vs  6.63% WER (-21%)
#   - noisy (2dB SNR): 46.58% vs 75.20% WER (-38%)
#   - Built-in punctuation & capitalization
#   - Trained to reduce hallucinations on non-speech audio
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "bzikst/faster-whisper-podlodka-turbo")
LANGUAGE = os.getenv("LANGUAGE", "ru")
HF_TOKEN = os.getenv("HF_TOKEN", "")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
COMPUTE_TYPE = "float16" if DEVICE == "cuda" else "int8"
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "16"))
PORT = int(os.getenv("PORT", "5000"))

# Transcription quality parameters
BEAM_SIZE = int(os.getenv("BEAM_SIZE", "10"))  # Higher = more accurate, slower

# Default prompt for Russian legal/courtroom context.
# Overridden by per-request 'prompt' field if provided.
DEFAULT_PROMPT = os.getenv(
    "DEFAULT_PROMPT",
    "Судебное заседание. Арбитражный суд. Протокол."
)

# ── Global model references ──────────────────────────────────────────

whisper_model = None
diarize_pipeline = None


def load_models():
    """Load WhisperX and Pyannote models into GPU memory.

    WhisperX passes asr_options to the underlying faster-whisper model.
    initial_prompt is set via asr_options at load time (WhisperX limitation,
    see https://github.com/m-bain/whisperX/issues/1020).

    We set the default prompt here; per-request prompts require model reload
    or patching — handled via the transcribe endpoint logic.
    """
    global whisper_model, diarize_pipeline

    asr_options = {
        "beam_size": BEAM_SIZE,
        "word_timestamps": True,
        "condition_on_previous_text": True,
        "initial_prompt": DEFAULT_PROMPT,
    }

    vad_options = {
        "vad_onset": 0.500,
        "vad_offset": 0.363,
    }

    log.info(f"Loading WhisperX model '{WHISPER_MODEL}' on {DEVICE} ({COMPUTE_TYPE})...")
    log.info(f"ASR options: beam_size={BEAM_SIZE}, "
             f"condition_on_previous_text=True, "
             f"initial_prompt='{DEFAULT_PROMPT[:50]}...'")
    t0 = time.time()
    whisper_model = whisperx.load_model(
        WHISPER_MODEL,
        device=DEVICE,
        compute_type=COMPUTE_TYPE,
        language=LANGUAGE,
        asr_options=asr_options,
        vad_options=vad_options,
    )
    log.info(f"WhisperX model loaded in {time.time() - t0:.1f}s")

    if HF_TOKEN:
        log.info("Loading Pyannote diarization pipeline...")
        t0 = time.time()
        from whisperx.diarize import DiarizationPipeline
        diarize_pipeline = DiarizationPipeline(
            token=HF_TOKEN,
            device=DEVICE
        )
        log.info(f"Pyannote pipeline loaded in {time.time() - t0:.1f}s")
    else:
        log.warning("HF_TOKEN not set — diarization disabled!")


def _update_model_prompt(prompt: str):
    """Update the initial_prompt on the loaded model's options.

    WhisperX stores the transcription options as model.options.
    We patch initial_prompt directly to avoid full model reload.
    """
    global whisper_model
    if whisper_model is None:
        return

    try:
        # WhisperX FasterWhisperPipeline stores options in .options attribute
        if hasattr(whisper_model, 'options') and whisper_model.options is not None:
            whisper_model.options = whisper_model.options._replace(
                initial_prompt=prompt
            )
            log.info(f"Updated initial_prompt to: '{prompt[:80]}...'")
        else:
            log.warning("Could not update initial_prompt — model.options not found")
    except Exception as e:
        log.warning(f"Failed to update initial_prompt: {e}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load models on startup."""
    load_models()
    yield
    log.info("Server shutting down.")


# ── FastAPI App ───────────────────────────────────────────────────────

app = FastAPI(
    title="WhisperX Transcription Server",
    version="2.0.0",
    description="Optimized Russian ASR with podlodka-turbo model",
    lifespan=lifespan
)


@app.get("/health")
async def health():
    """Health check endpoint for heartbeat monitoring."""
    return {
        "status": "ok",
        "device": DEVICE,
        "model": WHISPER_MODEL,
        "beam_size": BEAM_SIZE,
        "diarization": diarize_pipeline is not None,
        "gpu": torch.cuda.get_device_name(0) if DEVICE == "cuda" else "none"
    }


@app.post("/transcribe")
async def transcribe(
    file: UploadFile = File(...),
    prompt: Optional[str] = Form(None),
):
    """Transcribe an audio file with speaker diarization.

    Accepts any audio format supported by ffmpeg.

    Args:
        file: Audio file upload.
        prompt: Optional context prompt for better recognition accuracy.
                Example: "Судебное заседание. Участники: Иванов, Петров.
                         Дело о признании товаров контрафактными."

    Returns:
        JSON with segments: [{speaker, text, start, end}]
    """
    if whisper_model is None:
        raise HTTPException(503, "Models not loaded yet")

    # Apply per-request prompt if provided, otherwise use default
    active_prompt = prompt.strip() if prompt else DEFAULT_PROMPT
    _update_model_prompt(active_prompt)

    # Save uploaded file to temp
    suffix = Path(file.filename or "audio.wav").suffix or ".wav"
    job_id = uuid.uuid4().hex[:8]
    tmp_path = os.path.join(tempfile.gettempdir(), f"whisperx_{job_id}{suffix}")

    try:
        log.info(f"[{job_id}] Received file: {file.filename} ({file.content_type})")
        if prompt:
            log.info(f"[{job_id}] Context prompt: '{active_prompt[:100]}'")
        content = await file.read()
        with open(tmp_path, "wb") as f:
            f.write(content)
        file_size_mb = len(content) / (1024 * 1024)
        log.info(f"[{job_id}] Saved {file_size_mb:.1f} MB to {tmp_path}")

        # ── Step 1: Transcribe ────────────────────────────────────
        log.info(f"[{job_id}] Step 1/4: Transcribing "
                 f"(beam={BEAM_SIZE}, model={WHISPER_MODEL})...")
        t0 = time.time()
        audio = whisperx.load_audio(tmp_path)
        result = whisper_model.transcribe(audio, batch_size=BATCH_SIZE, language=LANGUAGE)
        t_transcribe = time.time() - t0
        log.info(f"[{job_id}] Transcription done in {t_transcribe:.1f}s "
                 f"({len(result.get('segments', []))} segments)")

        # ── Step 2: Align ─────────────────────────────────────────
        log.info(f"[{job_id}] Step 2/4: Aligning...")
        t0 = time.time()
        align_model, align_metadata = whisperx.load_align_model(
            language_code=LANGUAGE, device=DEVICE
        )
        result = whisperx.align(
            result["segments"], align_model, align_metadata,
            audio, DEVICE, return_char_alignments=False
        )
        t_align = time.time() - t0
        log.info(f"[{job_id}] Alignment done in {t_align:.1f}s")

        # Free alignment model memory
        del align_model
        if DEVICE == "cuda":
            torch.cuda.empty_cache()

        # ── Step 3: Diarize ───────────────────────────────────────
        if diarize_pipeline is not None:
            log.info(f"[{job_id}] Step 3/4: Diarizing...")
            t0 = time.time()
            diarize_df = diarize_pipeline(audio)
            result = whisperx.assign_word_speakers(diarize_df, result)
            t_diarize = time.time() - t0
            log.info(f"[{job_id}] Diarization done in {t_diarize:.1f}s")
        else:
            t_diarize = 0
            log.info(f"[{job_id}] Step 3/4: Diarization skipped (no HF_TOKEN)")

        # ── Step 4: Format output ─────────────────────────────────
        log.info(f"[{job_id}] Step 4/4: Formatting output...")
        segments = []
        for seg in result.get("segments", []):
            segments.append({
                "speaker": seg.get("speaker", "UNKNOWN"),
                "text": seg.get("text", "").strip(),
                "start": round(seg.get("start", 0), 2),
                "end": round(seg.get("end", 0), 2),
            })

        total_time = t_transcribe + t_align + t_diarize
        audio_duration = len(audio) / 16000  # WhisperX loads at 16kHz

        response = {
            "job_id": job_id,
            "segments": segments,
            "meta": {
                "audio_duration_sec": round(audio_duration, 1),
                "processing_time_sec": round(total_time, 1),
                "realtime_factor": round(audio_duration / total_time, 2) if total_time > 0 else 0,
                "num_segments": len(segments),
                "num_speakers": len(set(s["speaker"] for s in segments)),
                "model": WHISPER_MODEL,
                "beam_size": BEAM_SIZE,
                "prompt_used": active_prompt[:100],
                "device": DEVICE,
                "gpu": torch.cuda.get_device_name(0) if DEVICE == "cuda" else "cpu",
            }
        }

        log.info(f"[{job_id}] Done! {len(segments)} segments, "
                 f"{response['meta']['num_speakers']} speakers, "
                 f"{total_time:.1f}s processing, "
                 f"{response['meta']['realtime_factor']}x realtime")

        return JSONResponse(content=response)

    except Exception as e:
        log.error(f"[{job_id}] Error: {e}", exc_info=True)
        raise HTTPException(500, detail=str(e))
    finally:
        # Cleanup temp file
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

        # Restore default prompt after per-request override
        if prompt:
            _update_model_prompt(DEFAULT_PROMPT)


if __name__ == "__main__":
    log.info(f"Starting WhisperX server on port {PORT}...")
    uvicorn.run(app, host="0.0.0.0", port=PORT, log_level="info")
