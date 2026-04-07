"""
=============================================================
  Exam Face Verification System  –  Flask Backend
=============================================================
Endpoints
  Pages:   /  /register  /login  /exam  /admin
  API:     /api/liveness-challenge   GET   (get random challenge)
           /api/liveness-check       POST  (validate liveness frames)
           /api/register-face        POST
           /api/verify-face          POST
           /api/students             GET
           /api/delete-student/<id>  DELETE
============================================================="""

import os
import json
import uuid
import base64
import random
import datetime
from io import BytesIO
from functools import wraps

import cv2
import numpy as np
import mediapipe as mp
from PIL import Image, ImageFilter, ImageStat
import face_recognition
from flask import (
    Flask, render_template, request, jsonify,
    session, redirect, url_for
)
import mysql.connector

# =============================================================
# App configuration
# =============================================================
app = Flask(__name__)
app.secret_key = "super-secret-key-change-in-production"

# Folder where uploaded images will be stored
UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), "uploads")
REGISTRATION_FOLDER = os.path.join(UPLOAD_FOLDER, "registrations")
VERIFICATION_FOLDER = os.path.join(UPLOAD_FOLDER, "verifications")

# Create directories if they don't exist
os.makedirs(REGISTRATION_FOLDER, exist_ok=True)
os.makedirs(VERIFICATION_FOLDER, exist_ok=True)

# Face matching threshold (lower = stricter)
FACE_MATCH_TOLERANCE = 0.45

# =============================================================
# Liveness detection configuration
# =============================================================
# MediaPipe FaceMesh: detects 468 facial landmarks per face.
# We use specific landmark indices for eyes and nose.
#
# Supported challenges:
#   "blink"      – user must blink once (EAR dip detected)
#   "turn_left"  – user turns head left  (nose shifts right in image)
#   "turn_right" – user turns head right (nose shifts left in image)
# =============================================================

# Initialize MediaPipe Face Mesh (singleton, reused across requests)
mp_face_mesh = mp.solutions.face_mesh.FaceMesh(
    static_image_mode=True,       # process individual frames
    max_num_faces=1,
    refine_landmarks=True,        # includes iris landmarks
    min_detection_confidence=0.5,
)

# EAR thresholds for blink detection
EAR_BLINK_THRESHOLD = 0.21   # EAR below this = eyes closed
EAR_OPEN_THRESHOLD  = 0.26   # EAR above this = eyes open

# Head turn: how far nose must shift from center (as fraction of face width)
HEAD_TURN_THRESHOLD = 0.12

# MediaPipe landmark indices
# Left eye:  [362, 385, 387, 263, 373, 380]  (right side of image)
# Right eye: [33,  160, 158, 133, 153, 144]  (left side of image)
# Nose tip:  1
# Left face edge: 234     Right face edge: 454
LEFT_EYE_IDX  = [362, 385, 387, 263, 373, 380]
RIGHT_EYE_IDX = [33,  160, 158, 133, 153, 144]
NOSE_TIP_IDX  = 1
LEFT_EDGE_IDX = 234
RIGHT_EDGE_IDX = 454

# =============================================================
# Database helper
# =============================================================
DB_CONFIG = {
    "host": "localhost",
    "user": "root",
    "password": "",           # ← change to your MySQL password
    "database": "face_exam_db",
    "charset": "utf8mb4",
}


def get_db():
    """Return a fresh MySQL connection."""
    return mysql.connector.connect(**DB_CONFIG)


# =============================================================
# Helper utilities
# =============================================================

def decode_base64_image(data_url: str) -> np.ndarray:
    """
    Convert a base64 data-URL to a numpy RGB array.
    Accepts 'data:image/...;base64,XXXX' or raw base64.
    """
    if "," in data_url:
        data_url = data_url.split(",", 1)[1]
    img_bytes = base64.b64decode(data_url)
    img = Image.open(BytesIO(img_bytes)).convert("RGB")
    return np.array(img)


def image_quality_ok(img_array: np.ndarray) -> tuple:
    """
    Basic image quality checks:
      - brightness (not too dark / too bright)
      - blurriness  (Laplacian variance)
    Returns (ok: bool, reason: str).
    """
    pil_img = Image.fromarray(img_array)
    gray = pil_img.convert("L")

    # --- Brightness check ---
    stat = ImageStat.Stat(gray)
    mean_brightness = stat.mean[0]          # 0-255
    if mean_brightness < 40:
        return False, "Image is too dark. Please improve lighting."
    if mean_brightness > 230:
        return False, "Image is too bright / overexposed."

    # --- Blur check (variance of edge-detect filter) ---
    edges = gray.filter(ImageFilter.FIND_EDGES)
    edge_stat = ImageStat.Stat(edges)
    edge_var = edge_stat.var[0]
    if edge_var < 100:
        return False, "Image appears blurry. Hold still and ensure focus."

    return True, "ok"


def save_numpy_image(img_array: np.ndarray, folder: str, prefix: str) -> str:
    """Save a numpy image array to disk. Return the relative path."""
    filename = f"{prefix}_{uuid.uuid4().hex[:8]}.jpg"
    filepath = os.path.join(folder, filename)
    Image.fromarray(img_array).save(filepath, "JPEG", quality=85)
    # Return path relative to the project root for storage
    return os.path.relpath(filepath, os.path.dirname(__file__))


def login_required(f):
    """Decorator: require face-verified session."""
    @wraps(f)
    def wrapper(*args, **kwargs):
        if "student_id" not in session:
            return redirect(url_for("login_page"))
        return f(*args, **kwargs)
    return wrapper


def assess_face_posture(img_array: np.ndarray) -> tuple:
    """
    Validate whether face posture is suitable for capture/verification.

    Checks:
      - exactly one face
      - face size is reasonable (not too far / too close)
      - face is near frame center
      - head is not heavily tilted

    Returns (ok: bool, reason: str)
    """
    h, w = img_array.shape[:2]
    face_locations = face_recognition.face_locations(img_array, model="hog")

    if len(face_locations) == 0:
        return False, "Face not detected. Look directly at the camera."
    if len(face_locations) > 1:
        return False, "Multiple faces detected. Only one person should appear in frame."

    top, right, bottom, left = face_locations[0]
    face_w = max(1, right - left)
    face_h = max(1, bottom - top)
    face_area_ratio = (face_w * face_h) / float(max(1, w * h))

    if face_area_ratio < 0.08:
        return False, "Move closer to the camera."
    if face_area_ratio > 0.65:
        return False, "Move slightly away from the camera."

    face_cx = (left + right) / 2.0
    face_cy = (top + bottom) / 2.0
    center_x_offset = abs(face_cx - (w / 2.0)) / float(max(1, w))
    center_y_offset = abs(face_cy - (h / 2.0)) / float(max(1, h))

    if center_x_offset > 0.17 or center_y_offset > 0.20:
        return False, "Center your face inside the guide and keep it steady."

    landmarks = face_recognition.face_landmarks(img_array, [face_locations[0]])
    if landmarks:
        lm = landmarks[0]
        left_eye = lm.get("left_eye", [])
        right_eye = lm.get("right_eye", [])
        if left_eye and right_eye:
            left_eye_y = np.mean([p[1] for p in left_eye])
            right_eye_y = np.mean([p[1] for p in right_eye])
            tilt_ratio = abs(left_eye_y - right_eye_y) / float(face_h)
            if tilt_ratio > 0.08:
                return False, "Keep your head upright and face forward."

    return True, "Posture is good."


# =============================================================
# Liveness detection helpers
# =============================================================

def compute_ear(landmarks, eye_indices, img_w, img_h):
    """
    Compute the Eye Aspect Ratio (EAR) for one eye.

    EAR = (|p2-p6| + |p3-p5|) / (2 * |p1-p4|)

    When the eye is open, EAR is ~0.25-0.35.
    When closed (blink), EAR drops below ~0.20.

    Args:
        landmarks: MediaPipe NormalizedLandmarkList
        eye_indices: list of 6 landmark indices [p1,p2,p3,p4,p5,p6]
        img_w, img_h: image dimensions for de-normalizing

    Returns:
        float: Eye Aspect Ratio
    """
    pts = []
    for idx in eye_indices:
        lm = landmarks[idx]
        pts.append((lm.x * img_w, lm.y * img_h))

    # Vertical distances
    v1 = np.linalg.norm(np.array(pts[1]) - np.array(pts[5]))  # |p2-p6|
    v2 = np.linalg.norm(np.array(pts[2]) - np.array(pts[4]))  # |p3-p5|
    # Horizontal distance
    h  = np.linalg.norm(np.array(pts[0]) - np.array(pts[3]))  # |p1-p4|

    if h == 0:
        return 0.3  # avoid division by zero
    return (v1 + v2) / (2.0 * h)


def compute_head_turn_ratio(landmarks, img_w):
    """
    Compute how far the nose tip is from the center of the face.

    Returns a value between -1 and 1:
      - Negative = nose is to the LEFT of center  (head turned RIGHT in real life)
      - Positive = nose is to the RIGHT of center (head turned LEFT in real life)

    We calculate: (nose_x - face_center_x) / face_width
    """
    nose_x       = landmarks[NOSE_TIP_IDX].x * img_w
    left_edge_x  = landmarks[LEFT_EDGE_IDX].x * img_w
    right_edge_x = landmarks[RIGHT_EDGE_IDX].x * img_w

    face_width  = abs(right_edge_x - left_edge_x)
    face_center = (left_edge_x + right_edge_x) / 2.0

    if face_width == 0:
        return 0.0
    return (nose_x - face_center) / face_width


def analyze_liveness_frames(frames_b64, challenge):
    """
    Analyze a sequence of frames to check if the user performed
    the requested liveness action.

    Args:
        frames_b64: list of base64 image strings
        challenge:  "blink", "turn_left", or "turn_right"

    Returns:
        (passed: bool, detail: str)
    """
    ear_values = []    # per-frame average EAR (left+right eye)
    turn_values = []   # per-frame head turn ratio

    for frame_b64 in frames_b64:
        try:
            img_array = decode_base64_image(frame_b64)
        except Exception:
            continue

        img_h, img_w = img_array.shape[:2]

        # Convert RGB → MediaPipe expects RGB (already RGB from PIL)
        results = mp_face_mesh.process(img_array)

        if not results.multi_face_landmarks:
            continue  # no face in this frame

        lm = results.multi_face_landmarks[0].landmark

        # Compute EAR for both eyes
        ear_left  = compute_ear(lm, LEFT_EYE_IDX, img_w, img_h)
        ear_right = compute_ear(lm, RIGHT_EYE_IDX, img_w, img_h)
        avg_ear   = (ear_left + ear_right) / 2.0
        ear_values.append(avg_ear)

        # Compute head turn ratio
        turn = compute_head_turn_ratio(lm, img_w)
        turn_values.append(turn)

    # Need at least 5 analyzable frames
    if len(ear_values) < 5:
        return False, "Could not detect your face in enough frames. Try again."

    # ---- Evaluate based on challenge type ----

    if challenge == "blink":
        # Look for at least one EAR dip below threshold,
        # surrounded by frames where eyes are open.
        found_closed = False
        found_open_before = False
        found_open_after  = False

        for i, ear in enumerate(ear_values):
            if ear >= EAR_OPEN_THRESHOLD:
                if not found_closed:
                    found_open_before = True
                else:
                    found_open_after = True
            elif ear < EAR_BLINK_THRESHOLD:
                if found_open_before:
                    found_closed = True

        if found_open_before and found_closed and found_open_after:
            return True, "Blink detected."
        else:
            return False, "Blink not detected. Please blink clearly once."

    elif challenge == "turn_left":
        # "Turn left" from user's perspective = nose shifts RIGHT in image
        # (because webcam is mirrored, nose moves right when user looks left)
        max_turn = max(turn_values)
        if max_turn >= HEAD_TURN_THRESHOLD:
            return True, "Head turn left detected."
        else:
            return False, "Head turn left not detected. Please turn your head to the left."

    elif challenge == "turn_right":
        # "Turn right" from user's perspective = nose shifts LEFT in image
        min_turn = min(turn_values)
        if min_turn <= -HEAD_TURN_THRESHOLD:
            return True, "Head turn right detected."
        else:
            return False, "Head turn right not detected. Please turn your head to the right."

    return False, "Unknown challenge."


# =============================================================
# Page routes
# =============================================================

@app.route("/")
def index():
    """Landing page – redirect to login."""
    return redirect(url_for("login_page"))


@app.route("/register")
def register_page():
    return render_template("register.html")


@app.route("/login")
def login_page():
    return render_template("login.html")


@app.route("/exam")
@login_required
def exam_page():
    return render_template(
        "exam.html",
        student_name=session.get("student_name", "Student"),
        student_id=session.get("student_id", ""),
    )


@app.route("/admin")
def admin_page():
    return render_template("admin.html")


@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login_page"))


# =============================================================
# API: Liveness challenge (get a random action)
# =============================================================
#
# The frontend calls this BEFORE face verification.
# We return one random action the user must perform.
# We also store the challenge in the Flask session so the
# /api/liveness-check endpoint can verify the correct action.
# =============================================================

@app.route("/api/liveness-challenge", methods=["GET"])
def api_liveness_challenge():
    """Return a random liveness challenge for the login flow."""
    challenges = ["blink", "turn_left", "turn_right"]
    chosen = random.choice(challenges)

    # Store in session so we can verify later
    session["liveness_challenge"] = chosen
    session["liveness_passed"]    = False

    # Human-readable instructions
    instructions = {
        "blink":      "Please blink once clearly",
        "turn_left":  "Please turn your head to the LEFT",
        "turn_right": "Please turn your head to the RIGHT",
    }

    return jsonify(
        success=True,
        challenge=chosen,
        instruction=instructions[chosen],
    )


# =============================================================
# API: Liveness check (validate frames against challenge)
# =============================================================
#
# The frontend captures ~15 frames while the user performs the
# challenge action, then sends them here for analysis.
#
# We use MediaPipe Face Mesh to:
#   - Track Eye Aspect Ratio (EAR) for blink detection
#   - Track nose position relative to face center for head turns
#
# If liveness passes, we set session["liveness_passed"] = True
# which the /api/verify-face endpoint will check.
# =============================================================

@app.route("/api/liveness-check", methods=["POST"])
def api_liveness_check():
    """
    Validate liveness from a series of webcam frames.

    Expects JSON:
      { "frames": ["data:image/jpeg;base64,...", ...] }

    The challenge type is read from session["liveness_challenge"].
    """
    data = request.get_json(silent=True)
    if not data:
        return jsonify(success=False, message="No data received."), 400

    frames = data.get("frames", [])
    if len(frames) < 5:
        return jsonify(success=False, message="Not enough frames captured."), 400

    # Get the challenge from session
    challenge = session.get("liveness_challenge")
    if not challenge:
        return jsonify(success=False, message="No liveness challenge active. Refresh and try again."), 400

    # Analyze the frames
    passed, detail = analyze_liveness_frames(frames, challenge)

    if passed:
        session["liveness_passed"] = True
        return jsonify(success=True, message=f"✅ Liveness passed! {detail}")
    else:
        session["liveness_passed"] = False
        return jsonify(success=False, message=f"❌ {detail}"), 400


@app.route("/api/posture-check", methods=["POST"])
def api_posture_check():
    """Validate face posture before capture or verification."""
    data = request.get_json(silent=True)
    if not data:
        return jsonify(success=False, message="No data received."), 400

    image_b64 = data.get("image", "")
    if not image_b64:
        return jsonify(success=False, message="No image received."), 400

    try:
        img_array = decode_base64_image(image_b64)
    except Exception:
        return jsonify(success=False, message="Invalid image data."), 400

    ok, reason = image_quality_ok(img_array)
    if not ok:
        return jsonify(success=False, message=reason)

    ok, reason = assess_face_posture(img_array)
    if not ok:
        return jsonify(success=False, message=reason)

    return jsonify(success=True, message="Posture is acceptable.")


# =============================================================
# API: Register face
# =============================================================

@app.route("/api/register-face", methods=["POST"])
def api_register_face():
    """
    Register a new student with face encoding.

    Expects JSON payload:
      {
        "student_id":   "STU-001",
        "student_name": "John Doe",
        "email":        "john@example.com",   (optional)
        "images":       ["data:image/jpeg;base64,...", ...]   (5 images)
      }

    Processing pipeline:
      1. Validate required fields (student_id, student_name)
      2. Reject if student_id already exists in DB
      3. For each image:
         a. Decode base64 → numpy RGB array
         b. Quality check (brightness + blur)
         c. Detect faces — require exactly 1 face
         d. Extract 128-d face encoding
      4. Average all valid encodings for robustness
      5. Save one representative photo to disk
      6. Insert record into MySQL `students` table
      7. Return JSON with success + rejection stats
    """

    # ---- Parse incoming JSON ----
    data = request.get_json(silent=True)
    if not data:
        return jsonify(success=False, message="No data received."), 400

    student_id   = (data.get("student_id") or "").strip()
    student_name = (data.get("student_name") or "").strip()
    email        = (data.get("email") or "").strip()
    images       = data.get("images", [])

    # ---- Validate required fields ----
    if not student_id or not student_name:
        return jsonify(success=False, message="Student ID and name are required."), 400

    if len(images) < 3:
        return jsonify(success=False, message="Please capture at least 3 face images."), 400

    # ---- Check for duplicate student_id in database ----
    conn = get_db()
    cur = conn.cursor(dictionary=True)
    cur.execute("SELECT student_id FROM students WHERE student_id = %s", (student_id,))
    if cur.fetchone():
        cur.close(); conn.close()
        return jsonify(success=False, message="This Student ID is already registered."), 409

    # ---- Process each captured image ----
    encodings = []           # list of valid 128-d face encodings
    first_good_image = None  # save the first usable frame as the record photo

    # Rejection counters for detailed feedback
    skipped_decode     = 0   # could not decode base64
    skipped_quality    = 0   # failed brightness or blur check
    skipped_no_face    = 0   # zero faces detected
    skipped_multi_face = 0   # more than one face detected
    skipped_no_enc     = 0   # face found but encoding failed

    for idx, img_b64 in enumerate(images):
        # Step 3a: Decode base64 data-URL → numpy array
        try:
            img_array = decode_base64_image(img_b64)
        except Exception:
            skipped_decode += 1
            continue

        # Step 3b: Quality check (brightness + sharpness)
        ok, reason = image_quality_ok(img_array)
        if not ok:
            skipped_quality += 1
            continue

        # Step 3c: Detect faces using HOG model (fast, CPU-friendly)
        face_locations = face_recognition.face_locations(img_array, model="hog")

        if len(face_locations) == 0:
            skipped_no_face += 1
            continue  # no face found in this frame

        if len(face_locations) > 1:
            skipped_multi_face += 1
            continue  # multiple faces → reject for safety

        # Step 3d: Extract the 128-dimensional face encoding
        enc = face_recognition.face_encodings(img_array, face_locations)
        if len(enc) == 0:
            skipped_no_enc += 1
            continue

        encodings.append(enc[0])

        # Keep the first valid image for the registration photo
        if first_good_image is None:
            first_good_image = img_array

    # ---- Check we have enough valid samples ----
    if len(encodings) < 2:
        cur.close(); conn.close()
        return jsonify(
            success=False,
            message=(
                f"Only {len(encodings)} valid face sample(s) detected out of "
                f"{len(images)} images. Need at least 2. "
                "Make sure your face is clearly visible with good lighting "
                "and no other people in the frame."
            ),
            stats={
                "total_images": len(images),
                "good": len(encodings),
                "skipped_quality": skipped_quality,
                "skipped_no_face": skipped_no_face,
                "skipped_multi_face": skipped_multi_face,
            },
        ), 400

    # ---- Step 4: Average the valid encodings for robustness ----
    avg_encoding = np.mean(encodings, axis=0).tolist()

    # ---- Step 5: Save one representative photo to uploads/ ----
    photo_path = save_numpy_image(first_good_image, REGISTRATION_FOLDER, student_id)

    # ---- Step 6: Insert into MySQL ----
    try:
        cur.execute(
            """INSERT INTO students
                   (student_id, student_name, email, photo_path, face_encoding)
               VALUES (%s, %s, %s, %s, %s)""",
            (
                student_id,
                student_name,
                email or None,
                photo_path,
                json.dumps(avg_encoding),   # serialize 128-d list as JSON text
            )
        )
        conn.commit()
    except Exception as e:
        cur.close(); conn.close()
        return jsonify(success=False, message=f"Database error: {str(e)}"), 500
    finally:
        cur.close(); conn.close()

    # ---- Step 7: Return success response with stats ----
    return jsonify(
        success=True,
        message=f"Registration successful! {len(encodings)} face samples captured.",
        good_images=len(encodings),
        stats={
            "total_images": len(images),
            "good": len(encodings),
            "skipped_quality": skipped_quality,
            "skipped_no_face": skipped_no_face,
            "skipped_multi_face": skipped_multi_face,
        },
    )


# =============================================================
# API: Verify face (login)
# =============================================================
#
# This endpoint handles the face verification (login) process.
# It is called from the login page when the student clicks
# "Verify Face". The frontend sends the student_id and a
# base64-encoded webcam snapshot.
#
# Verification flow:
#   1. Validate inputs
#   2. Look up the student in MySQL by student_id
#   3. Decode the base64 image → numpy array
#   4. Run image quality checks (brightness + blur)
#   5. Detect faces — must find exactly 1 face
#   6. Extract the 128-d face encoding from the live image
#   7. Compare it to the stored encoding using Euclidean distance
#   8. If distance ≤ threshold → match → create Flask session
#   9. Save a login log (pass or fail) into the login_logs table
#  10. Return JSON with success/failure + confidence score
#
# Face distance explained:
#   face_recognition.face_distance() returns the Euclidean distance
#   between two 128-d vectors.  Lower distance = more similar.
#   - 0.0  = identical
#   - 0.4  = same person (typical)
#   - 0.6+ = different person
#   We use FACE_MATCH_TOLERANCE (default 0.45) as the cutoff.
#
# Session explained:
#   Flask's `session` object stores data in a signed cookie.
#   After a successful match we set session["student_id"] and
#   session["student_name"].  The @login_required decorator on
#   the /exam route checks for session["student_id"] before
#   allowing access.
# =============================================================

@app.route("/api/verify-face", methods=["POST"])
def api_verify_face():
    """
    Verify a student's face for exam login.

    Expects JSON payload:
      {
        "student_id": "STU-001",
        "image":      "data:image/jpeg;base64,/9j/4AAQ..."
      }

    Returns JSON:
      { success: bool, message: str, confidence: float }
    """

    # ---- Step 1: Parse and validate inputs ----
    data = request.get_json(silent=True)
    if not data:
        return jsonify(success=False, message="No data received."), 400

    student_id = (data.get("student_id") or "").strip()
    image_b64  = data.get("image", "")

    # Reject if either field is empty
    if not student_id:
        return jsonify(success=False, message="Student ID is required."), 400
    if not image_b64:
        return jsonify(success=False, message="No image received."), 400

    # ---- Step 1b: Ensure liveness check was passed ----
    # The frontend must call /api/liveness-check before /api/verify-face.
    # This prevents someone from bypassing liveness by calling the API directly.
    if not session.get("liveness_passed"):
        return jsonify(
            success=False,
            message="Liveness check required. Please complete the liveness challenge first."
        ), 403

    # ---- Step 2: Look up the student in the database ----
    # We need the stored face encoding to compare against.
    conn = get_db()
    cur = conn.cursor(dictionary=True)
    cur.execute(
        "SELECT student_id, student_name, face_encoding FROM students WHERE student_id = %s",
        (student_id,)
    )
    student = cur.fetchone()

    if not student:
        # Student doesn't exist → tell them to register first
        cur.close(); conn.close()
        return jsonify(success=False, message="Student ID not found. Please register first."), 404

    # Deserialize the stored encoding from JSON text → numpy array (128 floats)
    stored_encoding = np.array(json.loads(student["face_encoding"]))

    # ---- Step 3: Decode the base64 image ----
    # The frontend sends "data:image/jpeg;base64,XXXX".
    # decode_base64_image() strips the prefix and converts to RGB numpy array.
    try:
        img_array = decode_base64_image(image_b64)
    except Exception:
        cur.close(); conn.close()
        return jsonify(success=False, message="Invalid image data."), 400

    # ---- Step 4: Image quality check ----
    # Reject images that are too dark, too bright, or too blurry.
    ok, reason = image_quality_ok(img_array)
    if not ok:
        cur.close(); conn.close()
        return jsonify(success=False, message=reason), 400

    # ---- Step 5: Detect faces in the image ----
    # face_recognition.face_locations() returns a list of bounding boxes.
    # We require exactly ONE face — no face or multiple faces = rejection.
    face_locations = face_recognition.face_locations(img_array, model="hog")

    if len(face_locations) == 0:
        # No face found → log as failed attempt with snapshot
        snap = save_numpy_image(img_array, VERIFICATION_FOLDER, f"verify_{student_id}")
        cur.execute(
            "INSERT INTO login_logs (student_id, snapshot_path, verification_result) VALUES (%s,%s,%s)",
            (student_id, snap, "failed")
        )
        conn.commit(); cur.close(); conn.close()
        return jsonify(success=False, message="No face detected. Please face the camera directly."), 400

    if len(face_locations) > 1:
        # Multiple faces → could be impersonation attempt → reject
        snap = save_numpy_image(img_array, VERIFICATION_FOLDER, f"verify_{student_id}")
        cur.execute(
            "INSERT INTO login_logs (student_id, snapshot_path, verification_result) VALUES (%s,%s,%s)",
            (student_id, snap, "failed")
        )
        conn.commit(); cur.close(); conn.close()
        return jsonify(
            success=False,
            message="Multiple faces detected. Only the registered student should be in the frame."
        ), 400

    # ---- Step 6: Extract the 128-d face encoding from the live image ----
    live_encoding = face_recognition.face_encodings(img_array, face_locations)
    if len(live_encoding) == 0:
        cur.close(); conn.close()
        return jsonify(success=False, message="Could not encode face. Try again."), 400

    # ---- Step 7: Compare live encoding vs stored encoding ----
    # face_distance() returns the Euclidean distance between the two encodings.
    # Lower distance = more similar faces.
    distance = face_recognition.face_distance([stored_encoding], live_encoding[0])[0]

    # Check if the distance is within our threshold
    # FACE_MATCH_TOLERANCE is defined at the top of app.py (default: 0.45)
    match = distance <= FACE_MATCH_TOLERANCE

    # Convert distance to a human-friendly "confidence" percentage
    # confidence = (1 - distance) * 100  →  e.g. distance 0.3 → 70% confidence
    confidence = round((1 - distance) * 100, 1)

    # ---- Step 8: Save verification snapshot to disk ----
    # We save every attempt (pass or fail) for admin review.
    snap = save_numpy_image(img_array, VERIFICATION_FOLDER, f"verify_{student_id}")

    # ---- Step 9: Insert a login log record ----
    # This creates an audit trail visible in the admin panel.
    # Fields: student_id, snapshot_path, verification_result, login_time (auto)
    result_str = "success" if match else "failed"
    cur.execute(
        "INSERT INTO login_logs (student_id, snapshot_path, verification_result) VALUES (%s,%s,%s)",
        (student_id, snap, result_str)
    )
    conn.commit()

    # ---- Step 10: Return the result ----
    if match:
        # ✅ Face matched → create a Flask session for this student.
        # This stores student_id and student_name in a signed cookie.
        # The @login_required decorator on /exam checks for this.
        session["student_id"]   = student["student_id"]
        session["student_name"] = student["student_name"]

        cur.close(); conn.close()
        return jsonify(
            success=True,
            message="Face verified! Redirecting to exam...",
            confidence=confidence,
        )
    else:
        # ❌ Face did not match → return error with confidence score
        cur.close(); conn.close()
        return jsonify(
            success=False,
            message="Face does not match. Verification failed.",
            confidence=confidence,
        ), 401


# =============================================================
# API: List students + logs (admin)
# =============================================================
#
# This endpoint powers the admin dashboard.
# It returns ALL registered students and ALL login logs
# in a single JSON response, so the frontend can build
# both the Students table and the Login Logs table.
# =============================================================

@app.route("/api/students", methods=["GET"])
def api_students():
    """
    Return all students and their login logs for the admin dashboard.

    Returns JSON:
      {
        "success":  true,
        "students": [ { student_id, student_name, email, photo_path, created_at }, ... ],
        "logs":     [ { id, student_id, snapshot_path, verification_result, login_time }, ... ]
      }
    """
    conn = get_db()
    cur = conn.cursor(dictionary=True)

    # ---- Fetch all registered students (newest first) ----
    cur.execute("""
        SELECT student_id, student_name, email, photo_path, created_at
        FROM students
        ORDER BY created_at DESC
    """)
    students = cur.fetchall()

    # Convert Python datetime objects to strings for JSON serialization
    # (MySQL returns datetime objects, but JSON doesn't support them)
    for s in students:
        if isinstance(s.get("created_at"), datetime.datetime):
            s["created_at"] = s["created_at"].strftime("%Y-%m-%d %H:%M:%S")

    # ---- Fetch all login logs (newest first) ----
    cur.execute("""
        SELECT id, student_id, snapshot_path, verification_result, login_time
        FROM login_logs
        ORDER BY login_time DESC
    """)
    logs = cur.fetchall()

    # Convert datetime objects in logs too
    for l in logs:
        if isinstance(l.get("login_time"), datetime.datetime):
            l["login_time"] = l["login_time"].strftime("%Y-%m-%d %H:%M:%S")

    cur.close(); conn.close()

    return jsonify(success=True, students=students, logs=logs)


# =============================================================
# API: List login logs only (admin)
# =============================================================
#
# Optional dedicated endpoint if the frontend only needs logs.
# The /api/students endpoint above returns BOTH, but this one
# returns ONLY login logs (lighter payload).
# =============================================================

@app.route("/api/login-logs", methods=["GET"])
def api_login_logs():
    """
    Return all login log records for the admin dashboard.

    Returns JSON:
      {
        "success": true,
        "logs": [ { id, student_id, snapshot_path, verification_result, login_time }, ... ]
      }
    """
    conn = get_db()
    cur = conn.cursor(dictionary=True)

    cur.execute("""
        SELECT id, student_id, snapshot_path, verification_result, login_time
        FROM login_logs
        ORDER BY login_time DESC
    """)
    logs = cur.fetchall()

    # Convert datetime to string for JSON
    for l in logs:
        if isinstance(l.get("login_time"), datetime.datetime):
            l["login_time"] = l["login_time"].strftime("%Y-%m-%d %H:%M:%S")

    cur.close(); conn.close()

    return jsonify(success=True, logs=logs)


# =============================================================
# API: Delete student (admin)
# =============================================================
#
# Deletes a student by their student_id.
# The MySQL schema uses ON DELETE CASCADE on the login_logs
# foreign key, so deleting a student automatically removes
# all of their login log records too.
#
# The admin dashboard calls this when the user clicks the
# 🗑 Delete button and confirms in the modal.
# =============================================================

@app.route("/api/delete-student/<student_id>", methods=["DELETE"])
def api_delete_student(student_id):
    """
    Delete a student and (via CASCADE) their login logs.

    URL parameter:
      student_id  — e.g. "STU-001"

    Returns JSON:
      { "success": true/false, "message": "..." }
    """
    conn = get_db()
    cur = conn.cursor()
    try:
        cur.execute("DELETE FROM students WHERE student_id = %s", (student_id,))
        conn.commit()

        if cur.rowcount == 0:
            # No rows deleted → student_id doesn't exist
            return jsonify(success=False, message="Student not found."), 404

    except Exception as e:
        return jsonify(success=False, message=f"Database error: {str(e)}"), 500
    finally:
        cur.close(); conn.close()

    return jsonify(success=True, message=f"Student {student_id} deleted.")


# =============================================================
# Serve uploaded images (for admin photo preview)
# =============================================================

@app.route("/uploads/<path:filename>")
def serve_upload(filename):
    """Serve files from the uploads directory."""
    from flask import send_from_directory
    return send_from_directory(UPLOAD_FOLDER, filename)


# =============================================================
# Run the server
# =============================================================

if __name__ == "__main__":
    print("=" * 55)
    print("  Exam Face Verification System")
    print("  http://127.0.0.1:5000")
    print("=" * 55)
    app.run(debug=True, host="0.0.0.0", port=5000)
