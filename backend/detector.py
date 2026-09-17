"""Person-associated, event-based hygiene analysis for uploaded and live video."""

from collections import defaultdict
import json
import logging
import math
import os
from pathlib import Path
import subprocess
import time

import cv2
import torch
from ultralytics import YOLO

try:
    import imageio_ffmpeg
except ImportError:  # Clear runtime error below explains the required dependency.
    imageio_ffmpeg = None

logger = logging.getLogger("uvicorn.error")

VALID_CLASS_FILE = Path(__file__).resolve().parents[1] / "frontend" / "src" / "hygiene_classes.json"
VALID_HYGIENE_CLASSES = frozenset(json.loads(VALID_CLASS_FILE.read_text(encoding="utf-8")))
VIOLATION_CLASSES = frozenset({"no mask", "no gloves"})
HEAD_CLASSES = frozenset({"mask", "no mask", "hairnet"})
HAND_CLASSES = frozenset({"gloves", "no gloves"})


def normalize_class_name(name):
    """Normalize spelling only; never turn an unknown model label into PPE."""
    normalized = " ".join(str(name).strip().lower().replace("_", " ").replace("-", " ").split())
    return normalized if normalized in VALID_HYGIENE_CLASSES else None


class HygieneDetector:
    """Runs the supplied pose and PPE models without replacing their metadata."""

    # Match the proven Colab inference thresholds.  A common base confidence
    # keeps the batch call efficient; ROI-specific thresholds are applied
    # after inference so HEAD=0.25 and HAND=0.30 behave like Colab.
    PPE_CONFIDENCE = 0.25
    HEAD_CONFIDENCE = 0.25
    HAND_CONFIDENCE = 0.30
    PPE_IOU = 0.70
    PERSON_CONFIDENCE = 0.45
    HEAD_X_PADDING = 60
    HEAD_TOP_PADDING = 130
    HEAD_BOTTOM_PADDING = 100
    WRIST_HALF_SIZE = 60
    WRIST_CONFIDENCE = 0.30
    PPE_FRAME_INTERVAL = 2
    POSE_IMAGE_SIZE = 640

    def __init__(self, pose_weights="yolov8s-pose.pt", hygiene_weights="best.pt"):
        self.hygiene_weights = Path(hygiene_weights).resolve()
        self.pose_weights = Path(pose_weights).resolve()
        self.hygiene_model = YOLO(self.hygiene_weights)

        # These are the class IDs used by the Colab notebook that produced
        # the correct predictions. Keep this mapping explicit in production
        # instead of depending on whichever names ultralytics exposes.
        self.hygiene_model.model.names = {
            0: "gloves",
            1: "hairnet",
            2: "no-mask",
            3: "mask",
            4: "no-gloves",
        }

        self.person_pose_model = YOLO(self.pose_weights)
        requested_interval = os.getenv("PPE_FRAME_INTERVAL", str(self.PPE_FRAME_INTERVAL))
        try:
            self.frame_interval = max(1, int(requested_interval))
        except ValueError:
            self.frame_interval = self.PPE_FRAME_INTERVAL
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.debug_detections = os.getenv("HYGIENE_DEBUG_DETECTIONS", "").lower() in {"1", "true", "yes"}

        # These are the labels embedded in best.pt.  Do not overwrite them.
        self.class_names = {
            int(class_id): normalized
            for class_id, name in self.hygiene_model.names.items()
            if (normalized := normalize_class_name(name)) is not None
        }
        self.violation_classes = VIOLATION_CLASSES
        self.last_frame_stats = self._empty_stats()
        self.last_inference_batches = 0
        self.pose_inference_count = 0
        self.ppe_inference_count = 0
        self.perf = defaultdict(float)
        self.perf_counts = defaultdict(int)
        logger.info(
            "hygiene_model_loaded weights=%s classes=%s names=%s",
            self.hygiene_weights, len(self.hygiene_model.names), self.hygiene_model.names,
        )
        logger.info("inference_device=%s ppe_frame_interval=%s", self.device, self.frame_interval)

    def _trace(self, stage, **values):
        """Opt-in structured trace of the one detection source of truth."""
        if self.debug_detections:
            logger.info("hygiene_trace=%s", json.dumps({"stage": stage, **values}, default=str))

    def _timed(self, name, started_at):
        self.perf[name] += time.perf_counter() - started_at
        self.perf_counts[name] += 1

    def _performance_profile(self, source_frames, sampled_frames, elapsed):
        def value(name):
            total, count = self.perf.get(name, 0.0), self.perf_counts.get(name, 0)
            return {"total": round(total, 3), "avg": round(total / count, 4) if count else 0.0}
        return {
            "source_frames": source_frames, "sampled_frames": sampled_frames,
            "pose_calls": self.pose_inference_count, "ppe_calls": self.ppe_inference_count,
            "read": value("read"), "pose": value("pose"), "pose_post": value("pose_post"),
            "roi": value("roi"), "ppe": value("ppe"), "ppe_post": value("ppe_post"),
            "annotation": value("annotation"), "temporal": value("temporal"),
            "writer": value("writer"), "process_frame": value("process_frame"),
            "elapsed": round(elapsed, 3), "effective_fps": round(source_frames / elapsed, 3),
        }

    def _empty_stats(self):
        return {
            "active_staff": 0,
            **{f"{name.replace(' ', '_')}_detected": 0 for name in VALID_HYGIENE_CLASSES},
        }

    @staticmethod
    def _clip_roi(roi, width, height):
        x1, y1, x2, y2 = (int(value) for value in roi)
        return max(0, x1), max(0, y1), min(width, x2), min(height, y2)

    @classmethod
    def _person_rois(cls, person_box, keypoints, width, height):
        """Build Colab-style HEAD and left/right WRIST ROIs from pose keypoints."""
        x1, y1, x2, y2 = person_box
        person_height = max(y2 - y1, 1)

        # -------------------------------------------------------------
        # HEAD: nose, eyes and ears (keypoints 0..4), exactly as Colab.
        # -------------------------------------------------------------
        head_kpts = [kp for kp in keypoints[0:5] if kp[2] > 0.2]

        if head_kpts:
            hx_min = max(0, int(min(kp[0] for kp in head_kpts)) - cls.HEAD_X_PADDING)
            hy_min = max(0, int(min(kp[1] for kp in head_kpts)) - cls.HEAD_TOP_PADDING)
            hx_max = min(width, int(max(kp[0] for kp in head_kpts)) + cls.HEAD_X_PADDING)
            hy_max = min(height, int(max(kp[1] for kp in head_kpts)) + cls.HEAD_BOTTOM_PADDING)
        else:
            # Same fallback idea as the Colab notebook.
            hx_min, hy_min = max(0, x1), max(0, y1)
            hx_max, hy_max = min(width, x2), min(height, y1 + int(person_height * 0.30))

        rois = [
            ("HEAD", (hx_min, hy_min, hx_max, hy_max), cls.HEAD_CONFIDENCE),
        ]

        # -------------------------------------------------------------
        # HANDS: left wrist (9) and right wrist (10), 120x120 crops.
        # -------------------------------------------------------------
        for side, index in (("HAND_LEFT", 9), ("HAND_RIGHT", 10)):
            wrist = keypoints[index]
            if wrist[2] <= cls.WRIST_CONFIDENCE:
                continue

            wx, wy = int(wrist[0]), int(wrist[1])
            wx1 = max(0, wx - cls.WRIST_HALF_SIZE)
            wy1 = max(0, wy - cls.WRIST_HALF_SIZE)
            wx2 = min(width, wx + cls.WRIST_HALF_SIZE)
            wy2 = min(height, wy + cls.WRIST_HALF_SIZE)

            rois.append((side, (wx1, wy1, wx2, wy2), cls.HAND_CONFIDENCE))

        return rois

    @staticmethod
    def _inside(person_box, point):
        x1, y1, x2, y2 = person_box
        px, py = point
        pad_x, pad_y = (x2 - x1) * 0.08, (y2 - y1) * 0.08
        return x1 - pad_x <= px <= x2 + pad_x and y1 - pad_y <= py <= y2 + pad_y

    @staticmethod
    def _person_distance(person_box, point):
        x1, y1, x2, y2 = person_box
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        diagonal = max(((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5, 1)
        return ((point[0] - cx) ** 2 + (point[1] - cy) ** 2) ** 0.5 / diagonal

    @staticmethod
    def _spatially_plausible(person_box, ppe_box, class_name):
        """Reject model labels in anatomy-incompatible regions before state tracking.

        This is a guardrail, not a fabricated detection: an out-of-region model
        prediction is logged and ignored rather than relabelled.
        """
        px1, py1, px2, py2 = person_box
        x1, y1, x2, y2 = ppe_box
        person_height = max(py2 - py1, 1)
        relative_y = (((y1 + y2) / 2) - py1) / person_height
        if class_name in {"mask", "no mask", "hairnet"}:
            return -0.10 <= relative_y <= 0.62
        if class_name in {"gloves", "no gloves"}:
            # A glove state predicted around a person's head is not credible.
            return 0.25 <= relative_y <= 1.12
        return False

    def _associate(self, people, ppe_boxes, frame_index=None, timestamp=None):
        """Assign each PPE box to one containing/nearest tracked person."""
        assigned = defaultdict(list)
        for ppe in ppe_boxes:
            x1, y1, x2, y2 = ppe["box"]
            center = ((x1 + x2) / 2, (y1 + y2) / 2)
            candidates = [
                person for person in people
                if self._inside(person["box"], center)
                and self._spatially_plausible(person["box"], ppe["box"], ppe["class_name"])
            ]
            if not candidates:
                self._trace(
                    "association_rejected", frame=frame_index, timestamp=timestamp,
                    class_name=ppe["class_name"], bbox=ppe["box"], reason="outside_valid_person_region",
                )
                logger.debug(
                    "rejected_ppe_association class=%s box=%s reason=no spatially plausible person",
                    ppe["class_name"], ppe["box"],
                )
                continue
            owner = min(candidates, key=lambda person: self._person_distance(person["box"], center))
            self._trace(
                "association_accepted", frame=frame_index, timestamp=timestamp,
                class_name=ppe["class_name"], bbox=ppe["box"], person_id=owner["person_id"],
                person_bbox=owner["box"], distance=self._person_distance(owner["box"], center),
            )
            assigned[owner["person_id"]].append(ppe)
        return assigned

    def process_frame(self, frame, frame_index=None, fps=None):
        """Run Colab-aligned pose-guided PPE inference and return raw detections."""
        height, width = frame.shape[:2]
        timestamp = frame_index / fps if frame_index is not None and fps else None
        self.last_frame_stats = self._empty_stats()

        frame_started = time.perf_counter()

        # -------------------------------------------------------------
        # 1) Person pose tracking
        # -------------------------------------------------------------
        stage_started = time.perf_counter()
        pose_result = self.person_pose_model.track(
            frame,
            classes=[0],
            conf=self.PERSON_CONFIDENCE,
            persist=True,
            tracker="bytetrack.yaml",
            imgsz=self.POSE_IMAGE_SIZE,
            verbose=False,
            device=self.device,
        )[0]
        self._timed("pose", stage_started)
        self.pose_inference_count += 1

        # Keep the same requirement as the successful Colab code:
        # tracking ID + pose keypoints must be available.
        stage_started = time.perf_counter()
        people = []
        if (
            pose_result.boxes is not None
            and pose_result.boxes.id is not None
            and pose_result.keypoints is not None
        ):
            boxes = pose_result.boxes.xyxy.cpu().numpy()
            track_ids = pose_result.boxes.id.int().cpu().numpy()
            keypoints_data = pose_result.keypoints.data.cpu().numpy()

            for box, track_id, kpts in zip(boxes, track_ids, keypoints_data):
                x1, y1, x2, y2 = box.astype(int)
                people.append({
                    "person_id": f"Staff_{track_id}",
                    "box": (
                        max(0, x1),
                        max(0, y1),
                        min(width, x2),
                        min(height, y2),
                    ),
                    "keypoints": kpts,
                })
        self._timed("pose_post", stage_started)

        # -------------------------------------------------------------
        # 2) Build targeted Colab-style ROIs
        # -------------------------------------------------------------
        stage_started = time.perf_counter()
        roi_jobs = []
        for person in people:
            for roi_type, roi, min_conf in self._person_rois(
                person["box"], person["keypoints"], width, height
            ):
                rx1, ry1, rx2, ry2 = roi
                if rx2 <= rx1 or ry2 <= ry1:
                    continue
                roi_jobs.append({
                    "person": person,
                    "roi_type": roi_type,
                    "roi": roi,
                    "min_conf": min_conf,
                    "crop": frame[ry1:ry2, rx1:rx2],
                })
        self._timed("roi", stage_started)

        # -------------------------------------------------------------
        # 3) Run best.pt on focused ROIs
        # -------------------------------------------------------------
        self.last_inference_batches = 0
        results = []
        if roi_jobs:
            stage_started = time.perf_counter()
            results = self.hygiene_model(
                [job["crop"] for job in roi_jobs],
                conf=self.PPE_CONFIDENCE,
                iou=self.PPE_IOU,
                imgsz=640,
                verbose=False,
                device=self.device,
            )
            self.last_inference_batches = 1
            self.ppe_inference_count += 1
            self._timed("ppe", stage_started)

        # -------------------------------------------------------------
        # 4) Convert crop coordinates back to source-frame coordinates
        # -------------------------------------------------------------
        stage_started = time.perf_counter()
        assigned = defaultdict(list)

        for job, result in zip(roi_jobs, results):
            person = job["person"]
            roi_type = job["roi_type"]
            rx1, ry1, rx2, ry2 = job["roi"]
            min_conf = job["min_conf"]

            if result.boxes is None:
                continue

            if roi_type == "HEAD":
                allowed = HEAD_CLASSES
            else:
                allowed = HAND_CLASSES

            for box in result.boxes:
                confidence = float(box.conf.item())

                # Reproduce Colab thresholds per ROI.
                if confidence < min_conf:
                    continue

                class_id = int(box.cls.item())
                raw_name = str(self.hygiene_model.names.get(class_id, "unknown"))
                mapped_name = self.class_names.get(class_id)
                raw_box = tuple(round(value, 1) for value in box.xyxy[0].tolist())

                local_x1, local_y1, local_x2, local_y2 = map(
                    int, box.xyxy[0].tolist()
                )
                source_box = self._clip_roi(
                    (
                        rx1 + local_x1,
                        ry1 + local_y1,
                        rx1 + local_x2,
                        ry1 + local_y2,
                    ),
                    width,
                    height,
                )

                accepted = mapped_name in allowed
                self._trace(
                    "raw_model_output",
                    frame=frame_index,
                    timestamp=timestamp,
                    source_width=width,
                    source_height=height,
                    model_input=640,
                    roi_type=roi_type,
                    roi=(rx1, ry1, rx2, ry2),
                    person_id=person["person_id"],
                    class_id=class_id,
                    raw_class_name=raw_name,
                    mapped_class_name=mapped_name,
                    confidence=round(confidence, 4),
                    raw_bbox=raw_box,
                    source_bbox=source_box,
                    survives_confidence=True,
                    survives_model_nms=True,
                    roi_accepted=accepted,
                )

                if not accepted:
                    self._trace(
                        "roi_rejected",
                        frame=frame_index,
                        timestamp=timestamp,
                        person_id=person["person_id"],
                        roi_type=roi_type,
                        class_id=class_id,
                        class_name=mapped_name or raw_name,
                        bbox=source_box,
                        reason="unsupported_class_for_roi",
                    )
                    continue

                assigned[person["person_id"]].append({
                    "class_name": mapped_name,
                    "confidence": confidence,
                    "box": source_box,
                    "roi_type": roi_type,
                })

        self._timed("ppe_post", stage_started)

        # -------------------------------------------------------------
        # 5) Keep strongest class per person and resolve contradictions
        # -------------------------------------------------------------
        stage_started = time.perf_counter()
        detections = []

        for person in people:
            person_id = person["person_id"]
            strongest = {}

            for ppe in assigned[person_id]:
                previous = strongest.get(ppe["class_name"])
                if previous is None or ppe["confidence"] > previous["confidence"]:
                    strongest[ppe["class_name"]] = ppe

            # Keep the stronger state if both states appear.
            contradiction_pairs = (("mask", "no mask"), ("gloves", "no gloves"))
            for positive, negative in contradiction_pairs:
                if positive in strongest and negative in strongest:
                    if strongest[positive]["confidence"] >= strongest[negative]["confidence"]:
                        strongest.pop(negative)
                    else:
                        strongest.pop(positive)

            violations = []

            for name, ppe in strongest.items():
                status = "violation" if name in self.violation_classes else "compliant"
                detections.append({
                    "person_id": person_id,
                    "class_name": name,
                    "status": status,
                    "confidence": ppe["confidence"],
                    "box": ppe["box"],
                })

                self._trace(
                    "validated_detection",
                    frame=frame_index,
                    timestamp=timestamp,
                    person_id=person_id,
                    class_name=name,
                    status=status,
                    confidence=round(ppe["confidence"], 4),
                    bbox=ppe["box"],
                    survives_temporal_stabilization=False,
                )

                self.last_frame_stats[f"{name.replace(' ', '_')}_detected"] += 1
                if status == "violation":
                    violations.append(name)

                x1, y1, x2, y2 = ppe["box"]
                color = (0, 0, 255) if status == "violation" else (0, 220, 100)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(
                    frame,
                    f"{name} {ppe['confidence']:.2f}",
                    (x1, max(18, y1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    2,
                )

            px1, py1, px2, py2 = person["box"]
            status = "VIOLATION: " + ", ".join(violations) if violations else "MONITORED"
            color = (0, 0, 255) if violations else (0, 255, 0)
            cv2.rectangle(frame, (px1, py1), (px2, py2), color, 2)
            cv2.putText(
                frame,
                f"{person_id} {status}",
                (px1, max(20, py1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                color,
                2,
            )

        self._timed("annotation", stage_started)
        self._timed("process_frame", frame_started)
        self.last_frame_stats["active_staff"] = len(people)
        return frame, detections

    @staticmethod
    def _open_mp4_writer(destination, fps, size):
        """Open the MP4 encoder bundled with this Windows OpenCV build.

        OpenH264 is not installed here, so probing avc1/H264/X264 only emits
        encoder failures and can leave unusable writers.  ``mp4v`` is the
        installed FFmpeg MPEG-4 Part 2 encoder and is verified before frames
        are accepted.
        """
        fourcc_name = "mp4v"
        destination = Path(destination)
        writer = cv2.VideoWriter(
            str(destination), cv2.VideoWriter_fourcc(*fourcc_name), float(fps), tuple(size)
        )
        if writer.isOpened():
            logger.info(
                "video_writer_opened codec=%s path=%s fps=%.3f size=%sx%s",
                fourcc_name, destination, fps, size[0], size[1],
            )
            return writer
        writer.release()
        logger.error(
            "video_writer_failed codec=%s path=%s fps=%.3f size=%sx%s",
            fourcc_name, destination, fps, size[0], size[1],
        )
        raise RuntimeError(
            f"Unable to initialize MP4 VideoWriter (codec={fourcc_name}, path={destination}, "
            f"fps={fps}, size={size[0]}x{size[1]})."
        )

    @staticmethod
    def _ffmpeg_path():
        if imageio_ffmpeg is None:
            raise RuntimeError(
                "Browser-compatible H.264 finalization requires imageio-ffmpeg. "
                "Install backend requirements before analyzing video."
            )
        return imageio_ffmpeg.get_ffmpeg_exe()

    @classmethod
    def _finalize_browser_mp4(cls, intermediate_path, output_path):
        """Transcode OpenCV's intermediate mp4v into HTML5 H.264 MP4.

        OpenCV on this Windows build cannot initialize OpenH264.  The bundled
        FFmpeg binary provides libx264, yuv420p compatibility and faststart.
        """
        intermediate_path, output_path = Path(intermediate_path), Path(output_path)
        command = [
            cls._ffmpeg_path(), "-y", "-v", "error", "-i", str(intermediate_path),
            "-an", "-c:v", "libx264", "-preset", "veryfast", "-pix_fmt", "yuv420p",
            "-movflags", "+faststart", str(output_path),
        ]
        completed = subprocess.run(command, capture_output=True, text=True, check=False)
        if completed.returncode != 0:
            raise RuntimeError(f"FFmpeg H.264 finalization failed for {output_path}: {completed.stderr.strip()}")
        if not output_path.exists() or output_path.stat().st_size <= 1024:
            raise RuntimeError(f"FFmpeg did not create a non-empty MP4: {output_path}")
        logger.info("browser_mp4_finalized codec=h264 path=%s bytes=%s", output_path, output_path.stat().st_size)

    @staticmethod
    def _intermediate_path(destination):
        destination = Path(destination)
        return destination.with_name(f"{destination.stem}.intermediate{destination.suffix}")

    @staticmethod
    def _valid_video(path, expected_fps):
        path = Path(path)
        if not path.exists() or path.stat().st_size <= 1024:
            return False
        capture = cv2.VideoCapture(str(path))
        try:
            fps = capture.get(cv2.CAP_PROP_FPS)
            frame_count = capture.get(cv2.CAP_PROP_FRAME_COUNT)
            width = capture.get(cv2.CAP_PROP_FRAME_WIDTH)
            height = capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
            ok, _ = capture.read()
            return bool(ok and fps > 0 and frame_count > 0 and frame_count / fps > 0 and width > 0 and height > 0)
        finally:
            capture.release()

    @classmethod
    def _write_clip(cls, source_path, destination, fps, start_frame, end_frame):
        capture = cv2.VideoCapture(str(source_path))
        if not capture.isOpened():
            return False
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        capture.set(cv2.CAP_PROP_POS_FRAMES, max(0, start_frame))
        intermediate = cls._intermediate_path(destination)
        try:
            writer = cls._open_mp4_writer(intermediate, fps, (width, height))
        except Exception:
            capture.release()
            raise
        count = 0
        try:
            for _ in range(max(0, end_frame - start_frame + 1)):
                ok, frame = capture.read()
                if not ok:
                    break
                writer.write(frame)
                count += 1
        finally:
            capture.release()
            writer.release()
        try:
            if count <= 0 or not cls._valid_video(intermediate, fps):
                raise RuntimeError(f"Generated highlight intermediate is invalid: {intermediate}")
            cls._finalize_browser_mp4(intermediate, destination)
        finally:
            intermediate.unlink(missing_ok=True)
        if not cls._valid_video(destination, fps):
            raise RuntimeError(f"Generated highlight is not a valid readable MP4: {destination}")
        return True

    def process_video(self, input_path, output_path, clips_dir=None, job_id=None):
        """Create stable events and clips; frame detections never become events directly."""
        capture = cv2.VideoCapture(str(input_path))
        if not capture.isOpened():
            raise ValueError("The uploaded file is not a readable video.")
        width, height = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = capture.get(cv2.CAP_PROP_FPS) or 25.0
        total_frames = max(0, int(capture.get(cv2.CAP_PROP_FRAME_COUNT)))
        if width <= 0 or height <= 0:
            capture.release()
            raise ValueError("The uploaded video has invalid dimensions.")
        fps = min(max(float(fps), 1.0), 60.0)
        logger.info(
            "video_opened path=%s frames=%s fps=%.3f size=%sx%s",
            input_path, total_frames, fps, width, height,
        )
        logger.info("inference_started path=%s", input_path)
        try:
            writer = self._open_mp4_writer(output_path, fps, (width, height))
        except Exception:
            capture.release()
            raise

        # Hits only exist on sampled frames.  Scale the hit count so the
        # confirmation duration remains approximately 0.35 source seconds.
        min_stable_frames = max(2, math.ceil((fps * 0.35) / self.frame_interval))
        end_gap_frames = max(3, round(fps * 0.75))
        active, events = {}, []
        frame_index, max_staff, inference_batches = 0, 0, 0
        self.perf.clear()
        self.perf_counts.clear()
        sampled_frames = 0
        self.pose_inference_count = 0
        self.ppe_inference_count = 0
        started_at = time.perf_counter()
        try:
            while True:
                read_started = time.perf_counter()
                ok, frame = capture.read()
                self._timed("read", read_started)
                if not ok:
                    break
                sampled = frame_index % self.frame_interval == 0
                if sampled:
                    annotated, raw = self.process_frame(frame.copy(), frame_index=frame_index, fps=fps)
                    sampled_frames += 1
                else:
                    # Preserve source-frame timing/duration without manufacturing
                    # detections or temporal evidence on skipped source frames.
                    annotated, raw = frame, []
                writer_started = time.perf_counter()
                writer.write(annotated)
                self._timed("writer", writer_started)
                inference_batches += self.last_inference_batches
                max_staff = max(max_staff, self.last_frame_stats["active_staff"])
                seen = {(item["person_id"], item["class_name"]): item for item in raw}

                temporal_started = time.perf_counter()
                for key, item in seen.items():
                    state = active.setdefault(key, {"first": frame_index, "last": frame_index, "hits": 0, "confidence": 0.0, "opened": False, "item": item})
                    state["last"] = frame_index
                    state["hits"] += 1
                    state["confidence"] = max(state["confidence"], item["confidence"])
                    state["item"] = item
                    if not state["opened"] and state["hits"] >= min_stable_frames:
                        state["opened"] = True
                        self._trace(
                            "temporal_confirmed", frame=frame_index, timestamp=frame_index / fps,
                            person_id=item["person_id"], class_name=item["class_name"],
                            confidence=round(item["confidence"], 4), required_frames=min_stable_frames,
                        )

                for key, state in list(active.items()):
                    if frame_index - state["last"] <= end_gap_frames:
                        continue
                    if state["opened"]:
                        item = state["item"]
                        events.append({
                            "person_id": item["person_id"], "class_name": item["class_name"], "status": item["status"],
                            "start_frame": state["first"], "end_frame": state["last"],
                            "start_seconds": state["first"] / fps, "end_seconds": state["last"] / fps,
                            "confidence": state["confidence"],
                        })
                    active.pop(key)
                self._timed("temporal", temporal_started)
                frame_index += 1
                if frame_index == 1 or frame_index % 100 == 0:
                    elapsed = max(time.perf_counter() - started_at, 0.001)
                    logger.info(
                        "analysis_progress frame=%s total=%s elapsed=%.1fs processing_fps=%.2f pose_calls=%s ppe_calls=%s inference_batches=%s",
                        frame_index, total_frames, elapsed, frame_index / elapsed,
                        self.pose_inference_count, self.ppe_inference_count, inference_batches,
                    )
                    logger.info("performance_profile=%s", json.dumps(
                        self._performance_profile(frame_index, sampled_frames, elapsed)
                    ))
        finally:
            capture.release()
            writer.release()
            logger.info("writer_released path=%s frames=%s", output_path, frame_index)

        elapsed = max(time.perf_counter() - started_at, 0.001)
        logger.info(
            "frame_loop_finished frames=%s total=%s elapsed=%.1fs processing_fps=%.2f pose_calls=%s ppe_calls=%s inference_batches=%s",
            frame_index, total_frames, elapsed, frame_index / elapsed,
            self.pose_inference_count, self.ppe_inference_count, inference_batches,
        )
        logger.info("performance_profile=%s", json.dumps(
            self._performance_profile(frame_index, sampled_frames, elapsed)
        ))
        if not self._valid_video(output_path, fps):
            raise RuntimeError(f"Generated analysis output is not a valid readable MP4: {output_path}")

        for state in active.values():
            if state["opened"]:
                item = state["item"]
                events.append({
                    "person_id": item["person_id"], "class_name": item["class_name"], "status": item["status"],
                    "start_frame": state["first"], "end_frame": state["last"],
                    "start_seconds": state["first"] / fps, "end_seconds": state["last"] / fps,
                    "confidence": state["confidence"],
                })

        clips = []
        if clips_dir:
            for index, event in enumerate((event for event in events if event["status"] == "violation"), start=1):
                start = max(0, event["start_frame"] - round(fps * 3))
                end = min(max(frame_index - 1, 0), event["end_frame"] + round(fps * 3))
                filename = f"clip_{job_id}_{index}.mp4"
                # The analysis output carries the real model/person overlays.
                if self._write_clip(output_path, Path(clips_dir) / filename, fps, start, end):
                    clips.append({**event, "filename": filename})

        class_counts = defaultdict(int)
        status_counts = defaultdict(int)
        for event in events:
            class_counts[event["class_name"]] += 1
            status_counts[event["status"]] += 1
        logger.info(
            "analysis_completed output=%s bytes=%s frames=%s processing_fps=%.2f pose_calls=%s ppe_calls=%s",
            output_path, Path(output_path).stat().st_size, frame_index, frame_index / elapsed,
            self.pose_inference_count, self.ppe_inference_count,
        )
        return {
            "frames_processed": frame_index,
            "processing_fps": round(frame_index / elapsed, 2),
            "pose_inference_count": self.pose_inference_count,
            "ppe_inference_count": self.ppe_inference_count,
            "active_staff": max_staff,
            "events": events,
            "highlight_clips": clips,
            "class_counts": dict(class_counts),
            "status_counts": dict(status_counts),
            "total_checks": len(events),
            "total_alerts": status_counts["violation"],
            "compliant": status_counts["compliant"],
            "compliance_score": round(100 * status_counts["compliant"] / len(events), 1) if events else 100.0,
        }
