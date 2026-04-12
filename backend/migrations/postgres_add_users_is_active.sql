-- Fix: SQLAlchemy User model expects users.is_active (Boolean).
-- Run once on PostgreSQL / Supabase if you see:
--   psycopg2.errors.UndefinedColumn: column users.is_active does not exist
ALTER TABLE users ADD COLUMN IF NOT EXISTS is_active BOOLEAN NOT NULL DEFAULT TRUE;
