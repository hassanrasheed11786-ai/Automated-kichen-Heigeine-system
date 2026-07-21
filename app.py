# ==============================================================================
# STEP 1: LOAD OPTIMIZED TWO-STAGE MODELS (5 CORE CLASSES ONLY)
# ==============================================================================
print("🔄 Initializing Local Native Inference Engine with ByteTrack tracking...")
import os
import cv2
import numpy as np
from ultralytics import YOLO

# Ensure 'best.pt' file is present inside this project directory folder
model_path = "best.pt"

if os.path.exists(model_path):
    print(f"🎯 Loading Custom Kitchen Hygiene Weights from: {model_path}")
    hygiene_model = YOLO(model_path)
    
    # Strictly 5 classes format mapped perfectly
    custom_names = {
        0: 'gloves',     
        1: 'hairnet', 
        2: 'no-mask', 
        3: 'mask',       
        4: 'no-gloves'
    }
    
    # YOLOv8 deep layer structural patch to safely bypass read-only property constraints
    if hasattr(hygiene_model, 'model') and hasattr(hygiene_model.model, 'names'):
        hygiene_model.model.names = custom_names
    else:
        hygiene_model.__dict__['names'] = custom_names
        
    print("✅ Local model weights configurations fully aligned!")
else:
    print(f"❌ Error: '{model_path}' file aapke laptop par nahi mili! Pehle best.pt is folder mein rakhein.")
    exit()

# Load standard backbone model for lightning fast human tracking
print("📦 Loading base architecture for Human Tracking...")
person_model = YOLO('yolov8s.pt')

# ==============================================================================
# STEP 2: OPEN NATIVE LAPTOP CAMERA POINTER
# ==============================================================================
# 💡 NOTE: Laptop inbuilt camera chalana hai toh 0 rakhein. External USB Cam lagaya hai toh 1 karein.
cap = cv2.VideoCapture(0)

# Fallback auto-control loop: Agar index 0 responsive na ho, toh external cam 1 check karein
if not cap.isOpened():
    print("⚠️ Selected camera port not ready, attempting fallback channel shift...")
    cap = cv2.VideoCapture(1)

# Laptop video stream resolution matching limits configuration
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

print("\n🚀 Native Local Camera Stream Active! Apne face/haath camera ke samne layein...")
print("💡 Tip: Monitoring execution window ko stop karne ke liye keyboard se 'q' key dabaen.\n")

# ==============================================================================
# STEP 3: NATIVE REAL-TIME INFERENCE STREAMING LOOP
# ==============================================================================
try:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("❌ Error: Camera frame track matrix fetch nahi kar paa raha.")
            break
            
        # Run Stage 1: Active ByteTrack tracking directly on native local hardware stream
        person_results = person_model.track(
            source=frame, 
            classes=0, 
            conf=0.45, 
            persist=True, 
            tracker="bytetrack.yaml", 
            verbose=False
        )
        
        for pr in person_results:
            if pr.boxes is None or pr.boxes.id is None:
                continue
                
            # 🔥 FIX 1: Extract flat numpy array correctly for parent bounding coordinates
            boxes = pr.boxes.xyxy.cpu().numpy()
            track_ids = pr.boxes.id.int().cpu().numpy()
            
            for box, track_id in zip(boxes, track_ids):
                # Unpack the 1D flat row components safely
                x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
                
                # Dynamic boundaries clamping to prevent local frame layout overflows
                h, w, _ = frame.shape
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)
                
                person_id = f"Staff_{track_id}"
                person_crop = frame[y1:y2, x1:x2]
                
                if person_crop.size == 0:
                    continue
                
                # Run Stage 2: Inference inside the localized tracked sub-frame matrix
                hygiene_results = hygiene_model(person_crop, conf=0.55, verbose=False)
                
                detected_classes_inside = set()
                for hr in hygiene_results:
                    for h_box in hr.boxes:
                        c_id = int(h_box.cls.item())
                        
                        # Safeguard filter to skip out-of-index classes (apron/shoes)
                        if c_id not in custom_names:
                            continue
                            
                        detected_classes_inside.add(c_id)
                        
                        # 🔥 FIX 2: Dynamic 0-index flattening patch to prevent scalar type-errors permanently
                        coords = h_box.xyxy.cpu().numpy()[0]
                        hx1, hy1, hx2, hy2 = int(coords[0]), int(coords[1]), int(coords[2]), int(coords[3])
                        
                        label = hygiene_model.model.names[c_id]
                        c_conf = float(h_box.conf.item())
                        
                        # Har class (gloves, hairnet, mask, etc.) ka cyan inner box draw hoga
                        cv2.rectangle(frame, (x1 + hx1, y1 + hy1), (x1 + hx2, y1 + hy2), (255, 255, 0), 2)
                        cv2.putText(frame, f"{label} {c_conf:.2f}", (x1 + hx1, y1 + hy1 - 5),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
                
                # Active logical lookup definitions matrix
                has_no_mask   = 2 in detected_classes_inside
                has_no_gloves = 4 in detected_classes_inside
                
                # DETERMINISTIC INSTANT VIOLATION ALERT LOGIC
                if has_no_gloves or has_no_mask:
                    box_color = (0, 0, 255) # Solid Red
                    status_text = f"CRITICAL VIOLATION - {person_id}"
                else:
                    box_color = (0, 255, 0) # Safe Green
                    status_text = f"COMPLIANT - {person_id}"
                
                # Render Outer Parent Frame Box with active assigned ByteTrack Staff ID
                cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 3)
                cv2.putText(frame, status_text, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, box_color, 2, cv2.LINE_AA)
        
        # LOCAL DESKTOP DISPLAY WINDOW: High-speed native refresh window
        cv2.imshow("Automated AI Kitchen Hygiene System - Live Output", frame)
        
        # Continuous loop monitoring pointer check key listener 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print("\n🛑 Camera feed execution successfully stopped by the developer.")

# Port releases and system garbage collection clean routines
cap.release()
cv2.destroyAllWindows()
print("🛑 Monitoring pipeline safely terminated. Local hardware components detached.")
