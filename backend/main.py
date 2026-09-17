import asyncio
from collections import deque
import datetime
from datetime import datetime as dt, timedelta
import json
import logging
import os
from pathlib import Path
import shutil
import threading
from typing import AsyncGenerator, List, Dict, Any
import uuid
import time

import cv2
from fastapi import BackgroundTasks, Depends, FastAPI, File, HTTPException, Query, Request, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from sqlalchemy import func
from sqlalchemy.orm import Session

from database import DetectionLog, HighlightClip, HygieneEvent, get_db, SessionLocal, init_db
from detector import HygieneDetector, VALID_HYGIENE_CLASSES, normalize_class_name

# Use Uvicorn's configured logger so lifecycle messages appear in the same
# terminal as its request logs without requiring a separate logging setup.
logger = logging.getLogger("uvicorn.error")

BASE_DIR = Path(__file__).resolve().parent
os.chdir(BASE_DIR)
TEMP_DIR = BASE_DIR / "temp_videos"
STATIC_DIR = BASE_DIR / "static"
CLIPS_DIR = STATIC_DIR / "clips"
TEMP_DIR.mkdir(exist_ok=True)
STATIC_DIR.mkdir(exist_ok=True)
CLIPS_DIR.mkdir(exist_ok=True)

app = FastAPI(title="Kitchen Hygiene AI Monitor API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

@app.on_event("startup")
async def startup_event() -> None:
    """Ensure required database tables exist before requests are served."""
    init_db()



detector = HygieneDetector(pose_weights="yolov8s-pose.pt", hygiene_weights="best.pt")
video_source: int | str = 0

FPS = 30
frame_buffer: deque = deque(maxlen=FPS * 3)
active_cooldowns: dict[str, dt] = {}
CLIP_COOLDOWN_SECONDS: int = 180
active_recordings: List[Dict[str, Any]] = []
# Ultralytics trackers are stateful.  Do not run a live stream and an uploaded
# analysis concurrently against the same detector instance.
detector_lock = threading.Lock()

class AlertManager:
    def __init__(self) -> None:
        self.connections: set[WebSocket] = set()

    async def broadcast(self, detection: dict) -> None:
        if detection.get("status") != "violation":
            return
        message = json.dumps({
            "time": dt.now().strftime("%H:%M:%S"),
            "id": detection.get("person_id", "unknown"),
            "violation": f"Violation: {str(detection.get('class_name', 'unknown')).replace('-', ' ').title()}",
            "zone": "Live monitoring",
        })
        stale = []
        for connection in self.connections:
            try:
                await connection.send_text(message)
            except Exception:
                stale.append(connection)
        for connection in stale:
            self.connections.discard(connection)

alert_manager = AlertManager()

def is_in_cooldown(person_id: str) -> bool:
    last_triggered = active_cooldowns.get(person_id)
    if not last_triggered:
        return False
    return (dt.utcnow() - last_triggered).total_seconds() < CLIP_COOLDOWN_SECONDS

def set_cooldown(person_id: str) -> None:
    active_cooldowns[person_id] = dt.utcnow()

def parse_timeframe(timeframe: str) -> timedelta:
    clean = timeframe.strip().lower()
    if clean.endswith("m"): return timedelta(minutes=int(clean[:-1]))
    if clean.endswith("h"): return timedelta(hours=int(clean[:-1]))
    if clean.endswith("d"): return timedelta(days=int(clean[:-1]))
    try: return timedelta(minutes=int(clean))
    except ValueError: raise ValueError(f"Unsupported timeframe format: '{timeframe}'")

def remove_file(path: Path) -> None:
    try: path.unlink(missing_ok=True)
    except OSError: pass

def write_video_file(frames: list, filepath: str):
    if not frames: return
    height, width, _ = frames[0].shape
    out = detector._open_mp4_writer(filepath, float(FPS), (width, height))
    try:
        for frame in frames:
            out.write(frame)
    finally:
        out.release()
    if not detector._valid_video(filepath, float(FPS)):
        raise RuntimeError(f"Generated live highlight is not a valid readable MP4: {filepath}")

async def finalize_clip_task(rec_data: dict, filepath: str, clip_url: str):
    await asyncio.to_thread(write_video_file, rec_data["frames"], filepath)
    db = SessionLocal()
    try:
        new_clip = HighlightClip(
            person_id=rec_data["person_id"],
            reason=rec_data["reason"],
            clip_url=clip_url
        )
        db.add(new_clip)
        db.commit()
    except Exception as e:
        print(f"DB Error saving clip: {e}")
    finally:
        db.close()

async def batch_save_detections(logs: list):
    if not logs: return
    db = SessionLocal()
    try:
        db.bulk_save_objects(logs)
        db.commit()
    except Exception as e:
        print(f"DB Error saving logs: {e}")
    finally:
        db.close()

def save_upload_results(stats: dict) -> int:
    """Persist the upload's detection timeline and its generated clips atomically."""
    db = SessionLocal()
    try:
        events = stats.get("events", [])
        clips = stats.get("highlight_clips", [])
        clips_by_event = {
            (clip["person_id"], clip["class_name"], clip["start_frame"], clip["end_frame"]): clip
            for clip in clips
        }
        for event in events:
            if normalize_class_name(event.get("class_name")) is None:
                continue
            key = (event["person_id"], event["class_name"], event["start_frame"], event["end_frame"])
            clip = clips_by_event.get(key)
            db.add(HygieneEvent(
                source_id=stats["source_id"], person_id=event["person_id"], class_name=event["class_name"],
                status=event["status"], start_frame=event["start_frame"], end_frame=event["end_frame"],
                start_seconds=event["start_seconds"], end_seconds=event["end_seconds"], confidence=event["confidence"],
                clip_url=f"/static/clips/{clip['filename']}" if clip else None,
            ))
        db.commit()
        return len(clips)
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()

async def generate_frames_async() -> AsyncGenerator[bytes, None]:
    camera = cv2.VideoCapture(video_source)
    detection_log_buffer = []
    last_db_commit = time.time()
    
    try:
        while camera.isOpened():
            ok, frame = camera.read()
            if not ok:
                if isinstance(video_source, str):
                    camera.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                break

            result = detector.process_frame(frame)
            if isinstance(result, tuple):
                annotated, current_detections = result
            else:
                annotated = result
                current_detections = []

            frame_buffer.append(annotated.copy())

            for rec in active_recordings:
                if rec["remaining_frames"] > 0:
                    rec["frames"].append(annotated.copy())
                    rec["remaining_frames"] -= 1

            completed_recs = [r for r in active_recordings if r["remaining_frames"] == 0]
            active_recordings[:] = [r for r in active_recordings if r["remaining_frames"] > 0]

            for rec in completed_recs:
                filename = f"clip_{dt.utcnow().strftime('%Y%m%d%H%M%S')}_{rec['person_id']}.mp4"
                filepath = str(CLIPS_DIR / filename)
                clip_url = f"/static/clips/{filename}"
                asyncio.create_task(finalize_clip_task(rec, filepath, clip_url))

            for det in current_detections:
                p_id = str(det.get("person_id", "unknown"))
                c_name = normalize_class_name(det.get("class_name"))
                status = det.get("status", "compliant")

                if c_name is None:
                    continue

                detection_log_buffer.append(DetectionLog(
                    person_id=p_id, class_name=c_name, status=status
                ))

                if status == "violation" and not is_in_cooldown(p_id):
                    set_cooldown(p_id)
                    active_recordings.append({
                        "person_id": p_id,
                        "reason": f"Violation: {c_name.replace('-', ' ').title()}",
                        "frames": list(frame_buffer),
                        "remaining_frames": FPS * 5
                    })
                if status == "violation":
                    await alert_manager.broadcast(det)

            if time.time() - last_db_commit > 2.0 and detection_log_buffer:
                asyncio.create_task(batch_save_detections(detection_log_buffer.copy()))
                detection_log_buffer.clear()
                last_db_commit = time.time()

            encoded, buffer = cv2.imencode(".jpg", annotated)
            if encoded:
                yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n"
            
            await asyncio.sleep(0.001)
    finally:
        camera.release()

@app.get("/api/live_feed")
@app.get("/live_camera_feed")
async def live_camera_feed() -> StreamingResponse:
    return StreamingResponse(generate_frames_async(), media_type="multipart/x-mixed-replace; boundary=frame")

@app.get("/api/reports/analytics")
def get_analytics(
    timeframe: str = Query("1h", description="Timeframe filter e.g., 30m, 1h, 24h, 7d"),
    db: Session = Depends(get_db),
) -> dict:
    try:
        delta = parse_timeframe(timeframe)
    except ValueError as err:
        raise HTTPException(status_code=400, detail=str(err))

    start_time = dt.utcnow() - delta

    records = (
        db.query(
            HygieneEvent.class_name,
            HygieneEvent.status,
            func.count(HygieneEvent.id).label("count"),
        )
        .filter(HygieneEvent.created_at >= start_time)
        .group_by(HygieneEvent.class_name, HygieneEvent.status)
        .all()
    )

    class_counts: dict[str, int] = {}
    status_counts: dict[str, int] = {"compliant": 0, "violation": 0}
    total_detections = 0

    for row in records:
        c_name = normalize_class_name(row.class_name)
        if c_name is None:
            continue
        stat = (row.status or "unknown").lower()
        cnt = int(row.count)

        class_counts[c_name] = class_counts.get(c_name, 0) + cnt
        if stat in status_counts:
            status_counts[stat] += cnt
        else:
            status_counts[stat] = cnt
        total_detections += cnt

    compliance_score = (
        round((status_counts.get("compliant", 0) / total_detections) * 100, 1)
        if total_detections > 0
        else 100.0
    )

    return {
        "timeframe": timeframe,
        "start_time": start_time.isoformat(),
        "end_time": dt.utcnow().isoformat(),
        "total_detections": total_detections,
        "compliance_score": compliance_score,
        "class_counts": class_counts,
        "status_counts": status_counts,
    }

@app.get("/api/highlights")
def get_highlights(db: Session = Depends(get_db)) -> list[dict]:
    clips = (
        db.query(HygieneEvent)
        .filter(HygieneEvent.clip_url.is_not(None), HygieneEvent.class_name.in_(VALID_HYGIENE_CLASSES), HygieneEvent.status == "violation")
        .order_by(HygieneEvent.created_at.desc())
        .limit(6)
        .all()
    )

    return [
        {
            "id": clip.id,
            "timestamp": clip.created_at.isoformat() if clip.created_at else None,
            "person_id": clip.person_id,
            "class_name": clip.class_name,
            "title": f"{clip.class_name.title()} Detected",
            "reason": f"Violation: {clip.class_name.title()}",
            "clip_url": clip.clip_url,
            "start_seconds": clip.start_seconds,
            "end_seconds": clip.end_seconds,
            "confidence": clip.confidence,
        }
        for clip in clips
    ]

@app.post("/api/analyze_video")
async def analyze_video(request: Request, background_tasks: BackgroundTasks, file: UploadFile = File(...)) -> JSONResponse:
    if not file.filename:
        raise HTTPException(status_code=400, detail="Please select a video file.")

    job_id = uuid.uuid4().hex
    input_path = TEMP_DIR / f"{job_id}_input{Path(file.filename).suffix or '.mp4'}"
    output_name = f"analysis_{job_id}.mp4"
    output_path = STATIC_DIR / output_name

    try:
        # Uvicorn's access log is emitted only after this synchronous endpoint
        # returns. Emit lifecycle logs now so an operator can see the POST as
        # soon as its multipart body has reached FastAPI.
        logger.info("analysis %s received: filename=%s content_type=%s", job_id, file.filename, file.content_type)
        with input_path.open("wb") as destination:
            shutil.copyfileobj(file.file, destination)
        logger.info("analysis %s upload saved: %s bytes", job_id, input_path.stat().st_size)

        # Serialize stateful model/tracker use and make the response only after
        # output, detection logs, and highlight rows are all durable.
        def analyze_and_store() -> tuple[dict, int]:
            with detector_lock:
                logger.info("analysis %s inference started", job_id)
                analysis_stats = detector.process_video(
                    input_path, output_path, CLIPS_DIR, job_id
                )
                analysis_stats["source_id"] = job_id
                clip_count = save_upload_results(analysis_stats)
                logger.info(
                    "analysis %s persisted: frames=%s detections=%s highlights=%s",
                    job_id,
                    analysis_stats.get("frames_processed", 0),
                    len(analysis_stats.get("events", [])),
                    clip_count,
                )
                return analysis_stats, clip_count

        stats, highlights_created = await asyncio.to_thread(analyze_and_store)
        if not output_path.exists() or output_path.stat().st_size == 0:
            raise RuntimeError("Detector did not produce an output video.")

        video_url = str(request.url_for("static", path=output_name))
        payload = {
            "video_url": video_url,
            "active_staff": stats.get("active_staff", 0),
            "gloves_detected": stats.get("class_counts", {}).get("gloves", 0),
            "mask_detected": stats.get("class_counts", {}).get("mask", 0),
            "hairnet_detected": stats.get("class_counts", {}).get("hairnet", 0),
            "no_mask": stats.get("class_counts", {}).get("no mask", 0),
            "no_gloves": stats.get("class_counts", {}).get("no gloves", 0),
            "compliant": stats.get("compliant", 0),
            "total_alerts": stats.get("total_alerts", 0),
            "total_checks": stats.get("total_checks", 0),
            "compliance_score": stats.get("compliance_score", 100),
            "highlights_created": highlights_created,
        }

        # Upload analysis is synchronous, so publish its real violations once
        # the associated video, logs, and clips are committed.
        sent_alerts: set[tuple[str, str]] = set()
        for event in stats.get("events", []):
            alert_key = (event["person_id"], event["class_name"])
            if event["status"] == "violation" and alert_key not in sent_alerts:
                sent_alerts.add(alert_key)
                await alert_manager.broadcast(event)

        background_tasks.add_task(remove_file, input_path)
        logger.info("analysis %s completed: output=%s", job_id, output_name)
        return JSONResponse(payload, background=background_tasks)
    except ValueError as error:
        remove_file(output_path)
        logger.warning("analysis %s rejected: %s", job_id, error)
        raise HTTPException(status_code=400, detail=str(error)) from error
    except Exception as error:
        remove_file(output_path)
        logger.exception("analysis %s failed", job_id)
        raise HTTPException(status_code=500, detail=f"Video analysis failed: {error}") from error
    finally:
        await file.close()

@app.post("/upload_test_video")
async def upload_test_video(file: UploadFile = File(...)) -> dict[str, str]:
    global video_source
    source_path = BASE_DIR / "test_footage.mp4"
    with source_path.open("wb") as destination:
        shutil.copyfileobj(file.file, destination)
    await file.close()
    video_source = str(source_path)
    return {"message": "Test video uploaded successfully.", "source": video_source}

@app.websocket("/ws/alerts")
async def alerts(websocket: WebSocket) -> None:
    await websocket.accept()
    alert_manager.connections.add(websocket)
    try:
        while True:
            # Keep the connection open; real alerts are pushed by either
            # video-analysis or live-monitoring detections.
            await websocket.receive_text()
    except WebSocketDisconnect:
        pass
    finally:
        alert_manager.connections.discard(websocket)
