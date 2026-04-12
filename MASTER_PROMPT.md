# Master prompt — Credit Card Approval (north star)

**Goal:** A strong **AI-assisted credit approval** demo: clear ML outputs, **applicant** vs **bank staff** experiences, and production-minded code.

## Stack (current — do not regress)

- **Backend:** FastAPI + Uvicorn, Pydantic, JWT auth.
- **Database:** **Supabase (PostgreSQL)** via **`DATABASE_URL`** in `.env` — SQLAlchemy + `psycopg2`. **No SQLite** in this project.
- **ML:** `model/credit_approval_model.joblib`, `features.py`, `decision.py`, `credit_score.py`; training at repo root (`train_model.py`, etc.).
- **UI:** Server-rendered **Jinja** + **static** JS/CSS (React can be added later without deleting this).

## Roles

- **`customer`** — applicants: apply, see **My applications**.
- **`officer`** / **`admin`** — bank staff: **Bank queue** (all applications). Create staff users in Supabase (update `users.role`) or a seed script; registration defaults to `customer`.

## Feature priorities

1. **Applicant:** application form, saved predictions when logged in, **My applications** history.
2. **Staff:** **Bank queue** — list all recent applications with applicant identity fields where useful.
3. **Shared:** auth, role-gated routes, consistent API + UI tabs.
4. fairness dashboards, what-if simulator, exports — only after core flows feel solid.

## Out of scope (unless reopened)

- External **LLM** APIs for explanations.
- **SQLite** or swapping off Supabase without an explicit migration plan.

## Success criteria

- Two clear **UI modes** (tabs): customer flow vs bank queue.
- Staff and customer hit different endpoints; **403** if wrong role.
- `.env.example` documents **`DATABASE_URL`** for Supabase; no secrets in git.
