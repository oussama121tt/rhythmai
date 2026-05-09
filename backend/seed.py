"""RhythmAI Database Seeder

Initialize the database with sample users and analyses for testing.
Usage: python seed.py
Creates:
  - 1 admin account (admin@rhythmai.ai / admin2026)
  - 5 demo doctors with various roles and statuses
  - 11 sample ECG analyses across multiple doctors

Only runs if database is empty. Safe to run multiple times.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import logging
from datetime import datetime, timezone, timedelta
import random
from database import create_tables, SessionLocal, User, Analysis, AnalysisScore, AuditLog, next_ecg_id
from auth import hash_password

logger = logging.getLogger("rhythmai.seed")

CLASSES = ["NORM", "MI", "STTC", "CD", "ARR"]

# Admin password may be provided via environment for CI/production
ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD", "admin2026")

DEMO_USERS = [
    dict(name="System Administrator", email="admin@rhythmai.ai", password=ADMIN_PASSWORD,
         role="admin", medical_id="ADMIN-001", specialty="System", hospital="CardioScan HQ",
         status="verified", is_admin=True),
    dict(name="Dr. Amira Belhaj",    email="amira@hopital-charles.tn", password="doctor123",
         role="doctor", medical_id="MED-TN-2019-00142", specialty="Cardiology",
         hospital="Hôpital Charles Nicolle, Tunis", status="verified"),
    dict(name="Dr. Karim Mansouri",  email="karim@clinique-tawfik.tn", password="doctor123",
         role="doctor", medical_id="MED-TN-2015-00089", specialty="Internal Medicine",
         hospital="Clinique Tawfik, Sousse", status="verified"),
    dict(name="Dr. Sonia Trabelsi",  email="sonia@chu-sfax.tn",        password="doctor123",
         role="resident", medical_id="RES-TN-2024-00311", specialty="Cardiology",
         hospital="CHU Hédi Chaker, Sfax", status="verified"),
    dict(name="Yassine Ben Salah",   email="yassine@etudiant.um.tn",   password="student123",
         role="student", medical_id="ETU-UM-2024-01892", specialty="Medicine (5th year)",
         hospital="Université de Médecine, Monastir", status="pending"),
    dict(name="Dr. Nour Khelifi",    email="nour@ihec.tn",             password="doctor123",
         role="doctor", medical_id="MED-TN-2021-00576", specialty="Emergency Medicine",
         hospital="Hôpital Régional, Nabeul", status="suspended"),
]

SAMPLE_ANALYSES = [
    # (doctor_email, result, input_mode, patient_ref, age, sex, notes, days_ago)
    ("amira@hopital-charles.tn",  "NORM", "signal", "PT-4421", 52, "F",
     "Routine checkup — all clear.", 2),
    ("amira@hopital-charles.tn",  "MI",   "fusion",  "PT-3817", 67, "M",
     "Urgent — referred to cath lab immediately. Confirmed STEMI.", 4),
    ("amira@hopital-charles.tn",  "ARR",  "image",   "PT-5592", 45, "F",
     "", 7),
    ("amira@hopital-charles.tn",  "STTC", "signal",  "PT-2201", 59, "M",
     "Patient on beta-blockers, may influence ST.", 11),
    ("amira@hopital-charles.tn",  "CD",   "fusion",  "PT-6603", 73, "M",
     "", 13),
    ("amira@hopital-charles.tn",  "NORM", "signal",  "PT-1109", 38, "F",
     "Routine checkup — all clear.", 18),
    ("karim@clinique-tawfik.tn",  "MI",   "fusion",  "PT-8841", 71, "M",
     "", 3),
    ("karim@clinique-tawfik.tn",  "ARR",  "image",   "PT-9023", 55, "F",
     "AFib suspected. Holter ordered.", 6),
    ("karim@clinique-tawfik.tn",  "NORM", "signal",  "PT-7712", 29, "M",
     "", 9),
    ("sonia@chu-sfax.tn",         "STTC", "signal",  "PT-3345", 61, "M",
     "", 5),
    ("sonia@chu-sfax.tn",         "CD",   "fusion",  "PT-2218", 66, "F",
     "LBBB pattern — monitor closely.", 10),
]


def _make_scores(result: str):
    """Generate realistic probability scores for ECG classes.
    
    Creates a distribution where the predicted class has high probability
    and other classes have lower probabilities, all summing to ~100%.
    
    Args:
        result: The predicted ECG class (NORM, MI, STTC, CD, ARR)
    Returns:
        List of 5 probabilities (float) for each class
    """
    idx = CLASSES.index(result)
    base = [random.uniform(1, 8) for _ in CLASSES]
    base[idx] = random.uniform(65, 97)
    total = sum(base)
    return [round((v / total) * 100, 1) for v in base]


def seed():
    """Initialize the database with sample data.
    
    Creates tables if they don't exist, then populates with:
    - Admin user account
    - Demo doctor/resident/student accounts
    - Sample ECG analyses with simulated model outputs
    
    Idempotent: does nothing if database already contains users.
    """
    create_tables()
    db = SessionLocal()
    try:
        if db.query(User).count() > 0:
            logger.info("Database already seeded — skipping.")
            return

        # Insert users
        user_map = {}
        for u in DEMO_USERS:
            pwd = u.pop("password")
            obj = User(**u, password_hash=hash_password(pwd))
            db.add(obj)
            db.flush()
            user_map[obj.email] = obj.id

        db.flush()

        # Insert analyses
        now = datetime.now(timezone.utc)
        for (email, result, mode, pat_ref, age, sex, notes, days_ago) in SAMPLE_ANALYSES:
            doc_id = user_map.get(email)
            if not doc_id:
                continue

            scores = _make_scores(result)
            confidence = max(scores)
            ecg_id = f"ECG-{(now - timedelta(days=days_ago)).strftime('%Y%m%d')}-{str(db.query(Analysis).count()+1).zfill(4)}"
            created = now - timedelta(days=days_ago, hours=random.randint(0, 8))

            analysis = Analysis(
                ecg_id         = ecg_id,
                doctor_id      = doc_id,
                patient_ref    = pat_ref,
                patient_age    = age,
                patient_sex    = sex,
                input_mode     = mode,
                result         = result,
                confidence     = confidence,
                inference_time = round(random.uniform(0.25, 0.55), 2),
                notes          = notes,
                created_at     = created,
                updated_at     = created,
            )
            db.add(analysis)
            db.flush()

            for label, prob in zip(CLASSES, scores):
                db.add(AnalysisScore(analysis_id=analysis.id, label=label, probability=prob))

        db.commit()
        logger.info("✅ Seeded %d users and %d analyses.", len(DEMO_USERS), len(SAMPLE_ANALYSES))
        # Log created accounts without revealing passwords
        logger.info("Admin créé : %s", "admin@rhythmai.ai")
        other_emails = [e for e in user_map.keys() if e != "admin@rhythmai.ai"]
        if other_emails:
            logger.info("Autres comptes démo créés: %s", ", ".join(other_emails))

    except Exception as e:
        db.rollback()
        logger.exception("❌ Seed failed: %s", e)
        raise
    finally:
        db.close()


if __name__ == "__main__":
    seed()
