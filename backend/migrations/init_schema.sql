CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    email TEXT UNIQUE NOT NULL,
    password_hash TEXT NOT NULL,
    role TEXT NOT NULL CHECK(role IN ('applicant', 'officer', 'admin')),
    full_name TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    is_active INTEGER NOT NULL DEFAULT 1
);

CREATE TABLE IF NOT EXISTS applications (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL REFERENCES users(id),
    status TEXT NOT NULL DEFAULT 'pending' CHECK(status IN ('pending', 'ml_reviewed', 'officer_reviewing', 'approved', 'rejected', 'appealed')),
    code_gender INTEGER NOT NULL,
    flag_own_car INTEGER NOT NULL,
    flag_own_realty INTEGER NOT NULL,
    cnt_children INTEGER NOT NULL,
    amt_income_total REAL NOT NULL,
    name_income_type INTEGER NOT NULL,
    name_education_type INTEGER NOT NULL,
    name_family_status INTEGER NOT NULL,
    name_housing_type INTEGER NOT NULL,
    occupation_type INTEGER NOT NULL,
    flag_work_phone INTEGER NOT NULL,
    flag_phone INTEGER NOT NULL,
    flag_email INTEGER NOT NULL,
    cnt_fam_members INTEGER NOT NULL,
    age_years REAL NOT NULL,
    years_employed REAL NOT NULL,
    credit_score INTEGER,
    ml_probability_risky REAL,
    blended_risk_score REAL,
    ml_decision TEXT CHECK(ml_decision IN ('approved', 'rejected')),
    ml_strength_label TEXT,
    final_decision TEXT CHECK(final_decision IN ('approved', 'rejected')),
    recommended_credit_limit REAL,
    top_factors_json TEXT,
    model_version TEXT,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS officer_reviews (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    application_id INTEGER NOT NULL REFERENCES applications(id),
    officer_id INTEGER NOT NULL REFERENCES users(id),
    override_decision TEXT CHECK(override_decision IN ('approved', 'rejected')),
    justification TEXT NOT NULL,
    reviewed_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS audit_logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    application_id INTEGER REFERENCES applications(id),
    user_id INTEGER REFERENCES users(id),
    action TEXT NOT NULL,
    details_json TEXT,
    ip_address TEXT,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS notifications (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL REFERENCES users(id),
    type TEXT NOT NULL,
    title TEXT NOT NULL,
    message TEXT NOT NULL,
    is_read INTEGER NOT NULL DEFAULT 0,
    link TEXT,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS what_if_simulations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL REFERENCES users(id),
    base_application_id INTEGER REFERENCES applications(id),
    input_json TEXT NOT NULL,
    result_json TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_applications_user_id ON applications(user_id);
CREATE INDEX IF NOT EXISTS idx_applications_status ON applications(status);
CREATE INDEX IF NOT EXISTS idx_applications_created_at ON applications(created_at);
CREATE INDEX IF NOT EXISTS idx_officer_reviews_application_id ON officer_reviews(application_id);
CREATE INDEX IF NOT EXISTS idx_audit_logs_application_id ON audit_logs(application_id);
CREATE INDEX IF NOT EXISTS idx_notifications_user_id ON notifications(user_id);
CREATE INDEX IF NOT EXISTS idx_notifications_is_read ON notifications(is_read);
