# 🛡️ Exam Face Verification System

A full-stack web application that uses **facial recognition** to verify student identity before allowing exam access.

---

## 📁 Project Structure

```
face login/
├── app.py                     # Flask backend (all routes + API)
├── requirements.txt           # Python dependencies
├── schema.sql                 # MySQL database schema
├── README.md                  # This file
├── uploads/                   # Auto-created: stored face images
│   ├── registrations/         # Registration photos
│   └── verifications/         # Login verification snapshots
├── static/
│   ├── css/
│   │   └── style.css          # All styling
│   └── js/
│       └── main.js            # Shared JS utilities
└── templates/
    ├── base.html              # Jinja base template (navbar + layout)
    ├── register.html          # Student registration page
    ├── login.html             # Face verification login page
    ├── exam.html              # Protected exam page
    └── admin.html             # Admin dashboard
```

---

## 🚀 Step-by-Step Run Instructions

### Prerequisites

- **Python 3.8+** installed
- **MySQL 8.0+** installed and running
- **CMake** installed (required for `dlib` compilation)
- **Visual Studio Build Tools** (Windows) or `build-essential` (Linux)
- A webcam

### Step 1: Install CMake and Visual Studio Build Tools (Windows)

```bash
# Install CMake from https://cmake.org/download/
# Install Visual Studio Build Tools from https://visualstudio.microsoft.com/visual-cpp-build-tools/
# Make sure to select "Desktop development with C++" workload
```

### Step 2: Create a Python virtual environment

```bash
cd "face login"
python -m venv venv

# Activate it:
# Windows:
venv\Scripts\activate

# macOS/Linux:
source venv/bin/activate
```

### Step 3: Install Python dependencies

```bash
pip install -r requirements.txt
```

> **Note:** `dlib` and `face_recognition` may take several minutes to compile. If you encounter errors, make sure CMake is installed and accessible from your terminal.

### Step 4: Set up the MySQL database

1. Open MySQL client or MySQL Workbench.
2. Run the `schema.sql` file:

```bash
mysql -u root -p < schema.sql
```

Or paste the contents of `schema.sql` into your MySQL client and execute.

### Step 5: Configure database credentials (if needed)

Open `app.py` and update the `DB_CONFIG` dictionary (around line 47):

```python
DB_CONFIG = {
    "host": "localhost",
    "user": "root",
    "password": "",           # ← your MySQL password here
    "database": "face_exam_db",
}
```

### Step 6: Run the application

```bash
python app.py
```

You should see:

```
=======================================================
  Exam Face Verification System
  http://127.0.0.1:5000
=======================================================
```

### Step 7: Open in browser

Navigate to: **http://127.0.0.1:5000**

---

## 📋 How to Use

### Register a Student
1. Go to `/register`
2. Enter Student ID, Name, and optionally Email
3. Click "Next → Open Camera"
4. Position your face in the oval guide
5. Click "Scan Face" – it captures 7 images automatically
6. Click "Complete Registration" to submit

### Login with Face
1. Go to `/login`
2. Enter your Student ID
3. Click "Open Camera"
4. Click "Verify Face"
5. If matched, you'll be redirected to the exam page

### Admin Panel
1. Go to `/admin`
2. View all registered students and their photos
3. View all login attempt logs with snapshots
4. Delete students as needed

---

## 🔑 API Endpoints

| Method | Endpoint                        | Description               |
|--------|---------------------------------|---------------------------|
| POST   | `/api/register-face`            | Register student + face   |
| POST   | `/api/verify-face`              | Verify face for login     |
| GET    | `/api/students`                 | List all students & logs  |
| DELETE | `/api/delete-student/<id>`      | Delete a student          |

---

## ⚙️ Configuration

| Setting              | Location      | Default         |
|----------------------|---------------|-----------------|
| Face match threshold | `app.py:42`   | `0.45`          |
| Number of captures   | `register.html` JS | `7`        |
| MySQL credentials    | `app.py:47`   | root / no pass  |
| Session secret key   | `app.py:27`   | Change in prod! |

---

## 🔒 Security Notes

- Face encodings are stored server-side only; never sent to the frontend
- Flask sessions protect the exam page
- Image quality checks reject dark/blurry photos
- Multiple face detection is rejected
- Duplicate student IDs are rejected
- All API responses use proper HTTP status codes
