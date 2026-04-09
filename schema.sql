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
  is_flagged   TINYINT(1)    NOT NULL DEFAULT 0,
  failed_attempts INT        NOT NULL DEFAULT 0,
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
  liveness_status     VARCHAR(20)   NOT NULL DEFAULT 'unknown',
  match_score         DECIMAL(5,2)  DEFAULT NULL,
  exam_name           VARCHAR(100)  NOT NULL DEFAULT 'General Exam',
  is_flagged          TINYINT(1)    NOT NULL DEFAULT 0,
    login_time          TIMESTAMP     DEFAULT CURRENT_TIMESTAMP,

    FOREIGN KEY (student_id) REFERENCES students(student_id)
        ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

  CREATE INDEX idx_login_logs_time ON login_logs(login_time);
  CREATE INDEX idx_login_logs_result ON login_logs(verification_result);
  CREATE INDEX idx_login_logs_liveness ON login_logs(liveness_status);
  CREATE INDEX idx_login_logs_exam ON login_logs(exam_name);
  CREATE INDEX idx_login_logs_flagged ON login_logs(is_flagged);
