-- =============================================================
-- Exam Face Verification System - Database Schema
-- =============================================================

-- Create the database (run this first if it doesn't exist)
CREATE DATABASE IF NOT EXISTS face_exam_db
  CHARACTER SET utf8mb4
  COLLATE utf8mb4_unicode_ci;

USE face_exam_db;

-- -----------------------------------------------------------
-- Table: students
-- Stores registered student info and their face encodings
-- -----------------------------------------------------------
CREATE TABLE IF NOT EXISTS students (
    student_id   VARCHAR(20)   PRIMARY KEY,
    student_name VARCHAR(100)  NOT NULL,
    email        VARCHAR(100)  DEFAULT NULL,
    photo_path   TEXT          DEFAULT NULL,
    face_encoding LONGTEXT     DEFAULT NULL,   -- JSON-serialized averaged encoding
    created_at   TIMESTAMP     DEFAULT CURRENT_TIMESTAMP
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- -----------------------------------------------------------
-- Table: login_logs
-- Tracks every face-verification attempt (pass or fail)
-- -----------------------------------------------------------
CREATE TABLE IF NOT EXISTS login_logs (
    id                  INT           AUTO_INCREMENT PRIMARY KEY,
    student_id          VARCHAR(20)   NOT NULL,
    snapshot_path       TEXT          DEFAULT NULL,
    verification_result VARCHAR(20)   NOT NULL,   -- 'success' or 'failed'
    login_time          TIMESTAMP     DEFAULT CURRENT_TIMESTAMP,

    FOREIGN KEY (student_id) REFERENCES students(student_id)
        ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
