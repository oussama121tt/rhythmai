"""
RhythmAI — FastAPI Backend (main.py)
=====================================
Routes
------
POST   /api/auth/register          Sign-up (with optional ID card photo upload)
POST   /api/auth/login             Sign-in → JWT
GET    /api/auth/me                Current user profile

GET    /api/analyses               Doctor's own analysis list
POST   /api/analyses/predict       Upload signal(.dat+.hea) / image → run model → save
GET    /api/analyses/{id}          Detail + scores
PATCH  /api/analyses/{id}/notes    Update notes
DELETE /api/analyses/{id}          Delete

GET    /api/admin/stats            Platform-wide statistics
GET    /api/admin/users            All users (with id_card_photo)
PATCH  /api/admin/users/{id}       Update status / role
DELETE /api/admin/users/{id}       Delete user
GET    /api/admin/analyses         All analyses

Run:
    cd rhythmai/backend
    uvicorn main:app --reload --port 8000
"""

import os, io, sys, base64, logging, subprocess
from datetime import datetime, timezone
from typing import Optional
from pathlib import Path

import numpy as np
import concurrent.futures
import asyncio
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from fastapi import (FastAPI, Depends, HTTPException, UploadFile, File,
                     Form, Header, Request)
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sqlalchemy.orm import Session

try:
    from dotenv import load_dotenv
except Exception:  # pragma: no cover - optional dependency fallback
    load_dotenv = None

from database import (get_db, create_tables, next_ecg_id,
                      User, Analysis, AnalysisScore, AuditLog,
                      UserStatus, UserRole)
from auth import (hash_password, verify_password,
                  create_token, get_current_user_id)
from preprocessing import (preprocess_ecg_signal, preprocess_ecg_image,
                            load_signal_csv, load_signal_npy)
from model_loader import run_inference, CLASSES
from notifications import fetch_recent_ecg_articles

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("rhythmai")

if load_dotenv is not None:
    load_dotenv()

app = FastAPI(
    title="RhythmAI API",
    description=("Plateforme d'analyse ECG par IA pour cardiologues. "
                 "Authentification JWT requise pour toutes les routes protégées."),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup():
    create_tables()
    logger.info("✅ RhythmAI DB tables ready")
    scheduler = AsyncIOScheduler(timezone="UTC")
    scheduler.add_job(
        _refresh_notifications_job,
        trigger=CronTrigger(hour=7, minute=0),
        id="daily_notif_refresh",
        replace_existing=True,
    )
    scheduler.start()
    app.state.scheduler = scheduler
    logger.info("Daily notification scheduler started (07:00 UTC)")
    await _refresh_notifications_job()


@app.on_event("shutdown")
async def shutdown_event():
    if hasattr(app.state, "scheduler"):
        app.state.scheduler.shutdown(wait=False)
        logger.info("Notification scheduler stopped")


# ── Pydantic schemas ──────────────────────────────────────────────────────────
class LoginBody(BaseModel):
    email   : str
    password: str

class NotesBody(BaseModel):
    notes: str


class MarkReadBody(BaseModel):
    pmids: list[str] = []

class UpdateUserBody(BaseModel):
    status  : Optional[str]  = None
    role    : Optional[str]  = None
    is_admin: Optional[bool] = None


# ── Auth helpers ──────────────────────────────────────────────────────────────
def _current_user(authorization: str = Header(None), db: Session = Depends(get_db)) -> User:
    try:
        uid = get_current_user_id(authorization)
    except ValueError as e:
        raise HTTPException(401, str(e))
    user = db.query(User).filter(User.id == uid).first()
    if not user:
        raise HTTPException(401, "User not found")
    if user.status == "suspended":
        raise HTTPException(403, "Account suspended")
    return user


def _admin_user(authorization: str = Header(None), db: Session = Depends(get_db)) -> User:
    user = _current_user(authorization, db)
    if not user.is_admin:
        raise HTTPException(403, "Admin access required")
    return user


def _audit(db, user_id, action, target="", detail=""):
    db.add(AuditLog(user_id=user_id, action=action, target=target, detail=detail))


def _user_dict(u: User, include_email=True) -> dict:
    d = dict(
        id              = u.id,
        name            = u.name,
        role            = u.role,
        medical_id      = u.medical_id,
        specialty       = u.specialty,
        hospital        = u.hospital,
        status          = u.status,
        is_admin        = u.is_admin,
        joined_at       = u.joined_at.isoformat() if u.joined_at else None,
        last_login      = u.last_login.isoformat() if u.last_login else None,
        id_card_photo   = u.id_card_photo,
        id_card_filename= u.id_card_filename,
    )
    if include_email:
        d["email"] = u.email
    return d


def _analysis_dict(a: Analysis) -> dict:
    return dict(
        id             = a.id,
        ecg_id         = a.ecg_id,
        doctor_id      = a.doctor_id,
        patient_ref    = a.patient_ref,
        patient_age    = a.patient_age,
        patient_sex    = a.patient_sex,
        input_mode     = a.input_mode,
        result         = a.result,
        confidence     = a.confidence,
        inference_time = a.inference_time,
        notes          = a.notes,
        created_at     = a.created_at.isoformat() if a.created_at else None,
        scores         = {s.label: s.probability for s in a.scores},
    )


# ═══════════════════════════════════════════════════════════════════════════════
#  AUTH ROUTES
# ═══════════════════════════════════════════════════════════════════════════════

@app.post("/api/auth/register", status_code=201)
async def register(
    name        : str           = Form(...),
    email       : str           = Form(...),
    password    : str           = Form(...),
    role        : str           = Form("doctor"),
    medical_id  : str           = Form(...),
    specialty   : Optional[str] = Form(None),
    hospital    : Optional[str] = Form(None),
    id_card_photo: Optional[UploadFile] = File(None),
    db          : Session = Depends(get_db),
):
    """Register a new doctor with optional ID card photo (requires admin approval)."""
    if db.query(User).filter(User.email == email.lower().strip()).first():
        raise HTTPException(409, "Email already registered")
    if db.query(User).filter(User.medical_id == medical_id).first():
        raise HTTPException(409, "Medical ID already registered")

    # Handle ID card photo
    photo_b64  = None
    photo_name = None
    if id_card_photo:
        raw        = await id_card_photo.read()
        photo_b64  = base64.b64encode(raw).decode()
        photo_name = id_card_photo.filename

    user = User(
        name            = name,
        email           = email.lower().strip(),
        password_hash   = hash_password(password),
        role            = role,
        medical_id      = medical_id,
        specialty       = specialty,
        hospital        = hospital,
        status          = UserStatus.pending,
        is_admin        = False,
        id_card_photo   = photo_b64,
        id_card_filename= photo_name,
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    logger.info("New registration: %s (%s) photo=%s", user.email, user.medical_id, bool(photo_b64))
    return {"message": "Registration submitted. Await admin verification.", "user_id": user.id}


@app.post("/api/auth/login")
def login(body: LoginBody, db: Session = Depends(get_db)):
    """Authenticate a doctor and return a JWT token."""
    user = db.query(User).filter(User.email == body.email.lower().strip()).first()

    verified = False
    if user:
        try:
            verified = verify_password(body.password, user.password_hash)
            logger.debug("Login attempt for %s: %s", body.email, verified)
        except Exception as e:
            logger.exception("Password verification error for %s: %s", body.email, e)
            verified = False

    if not user or not verified:
        raise HTTPException(401, "Invalid email or password")

    if user.status == "pending":
        raise HTTPException(403, "Account pending admin verification")

    if user.status == "suspended":
        raise HTTPException(403, "Account suspended — contact admin")

    user.last_login = datetime.now(timezone.utc)
    db.commit()

    token = create_token(user.id, user.role, user.is_admin)

    return {
        "access_token": token,
        "token_type": "bearer",
        "user": _user_dict(user)
    }

@app.get("/api/auth/me")
def me(current: User = Depends(_current_user)):
    """Get current authenticated user profile."""
    return _user_dict(current)


# ═══════════════════════════════════════════════════════════════════════════════
#  ANALYSIS ROUTES
# ═══════════════════════════════════════════════════════════════════════════════

@app.get("/api/analyses")
def list_analyses(
    skip: int = 0, limit: int = 50,
    result_filter: Optional[str] = None,
    current: User = Depends(_current_user),
    db: Session = Depends(get_db),
):
    """List all ECG analyses performed by the current doctor."""
    q = db.query(Analysis).filter(Analysis.doctor_id == current.id)
    if result_filter:
        q = q.filter(Analysis.result == result_filter.upper())
    analyses = q.order_by(Analysis.created_at.desc()).offset(skip).limit(limit).all()
    return [_analysis_dict(a) for a in analyses]


@app.post("/api/analyses/predict", status_code=201)
async def predict(
    mode            : str            = Form(...),
    patient_ref     : Optional[str]  = Form(None),
    patient_age     : Optional[int]  = Form(None),
    patient_sex     : Optional[str]  = Form(None),
    signal_file     : Optional[UploadFile] = File(None),   # .dat / .npy / .csv
    image_file      : Optional[UploadFile] = File(None),
    current         : User = Depends(_current_user),
    db              : Session = Depends(get_db),
):
    """Run ECG inference on signal and/or image and save results."""
    if current.status != "verified":
        raise HTTPException(403, "Account must be verified before running analyses")

    signal_np    = None
    image_tensor = None

    if signal_file and mode in ("signal", "fusion"):
        raw = await signal_file.read()
        fname = (signal_file.filename or "signal.dat").lower()

        try:
            if fname.endswith(".npy"):
                signal_np = np.load(io.BytesIO(raw), allow_pickle=False)

            elif fname.endswith(".csv") or fname.endswith(".txt"):
                text = raw.decode("utf-8", errors="replace")
                signal_np = np.loadtxt(io.StringIO(text), delimiter=",")

            elif fname.endswith(".dat"):
                # WFDB binary format: int16 little-endian, 12 leads
                signal_np = np.frombuffer(raw, dtype=np.int16).astype(np.float32)
                n_samples = len(signal_np) // 12
                if n_samples < 1:
                    raise ValueError("File too small to contain 12-lead signal")
                signal_np = signal_np.reshape(12, n_samples)

            else:
                # Try int16 as fallback
                signal_np = np.frombuffer(raw, dtype=np.int16).astype(np.float32)
                n_samples = len(signal_np) // 12
                signal_np = signal_np.reshape(12, n_samples)

        except Exception as parse_err:
            raise HTTPException(
                status_code=422,
                detail=f"Cannot parse signal file '{fname}': {parse_err}"
            )

        # Fix orientation if needed
        if signal_np.ndim == 2 and signal_np.shape[0] != 12:
            signal_np = signal_np.T

        # Validate
        if signal_np.ndim != 2 or signal_np.shape[0] != 12:
            raise HTTPException(
                status_code=422,
                detail=f"Signal must have 12 leads. Got shape: {signal_np.shape}"
            )

        logger.info(f"Signal parsed: shape={signal_np.shape} dtype={signal_np.dtype}")

    # ── Process image ─────────────────────────────────────────────────────────
    if image_file and mode in ("image", "fusion"):
        raw = await image_file.read()
        from PIL import Image as PilImage
        img    = PilImage.open(io.BytesIO(raw)).convert("RGB")
        img_np = np.array(img)
        image_tensor = preprocess_ecg_image(img_np, target_size=224)

    # ── Validate ──────────────────────────────────────────────────────────────
    if mode == "signal" and signal_np is None:
        raise HTTPException(400, "Signal file required for signal mode")
    if mode == "image" and image_tensor is None:
        raise HTTPException(400, "Image file required for image mode")
    if mode == "fusion" and (signal_np is None or image_tensor is None):
        raise HTTPException(400, "Both signal and image required for fusion mode")

    # ── Run model ─────────────────────────────────────────────────────────────
    try:
        result = run_inference(
            signal_np,
            image_tensor,
            mode,
            patient_age=patient_age,
            patient_sex=patient_sex,
        )
    except Exception as exc:
        logger.error("Inference error: %s", exc, exc_info=True)
        raise HTTPException(500, f"Model inference failed: {exc}")
    logger.info("Inference complete: predicted=%s confidence=%s mode=%s", result["predicted"], max(result["scores"]), mode)

    # ── Save ──────────────────────────────────────────────────────────────────
    ecg_id   = next_ecg_id(db)
    analysis = Analysis(
        ecg_id         = ecg_id,
        doctor_id      = current.id,
        patient_ref    = patient_ref,
        patient_age    = patient_age,
        patient_sex    = patient_sex,
        input_mode     = mode,
        result         = result["predicted"],
        confidence     = max(result["scores"]),
        inference_time = result["inference_time"],
        notes          = "",
    )
    db.add(analysis)
    db.flush()
    for label, prob in zip(CLASSES, result["scores"]):
        db.add(AnalysisScore(analysis_id=analysis.id, label=label, probability=prob))
    db.commit()
    db.refresh(analysis)
    return _analysis_dict(analysis)


@app.get("/api/analyses/{analysis_id}")
def get_analysis(
    analysis_id: int,
    current: User = Depends(_current_user),
    db: Session = Depends(get_db),
):
    a = db.query(Analysis).filter(
        Analysis.id == analysis_id, Analysis.doctor_id == current.id
    ).first()
    if not a:
        raise HTTPException(404, "Analysis not found")
    return _analysis_dict(a)


@app.patch("/api/analyses/{analysis_id}/notes")
def update_notes(
    analysis_id: int,
    body: NotesBody,
    current: User = Depends(_current_user),
    db: Session = Depends(get_db),
):
    a = db.query(Analysis).filter(
        Analysis.id == analysis_id, Analysis.doctor_id == current.id
    ).first()
    if not a:
        raise HTTPException(404, "Analysis not found")
    a.notes      = body.notes
    a.updated_at = datetime.now(timezone.utc)
    db.commit()
    return {"message": "Notes updated"}


@app.delete("/api/analyses/{analysis_id}", status_code=204)
def delete_analysis(
    analysis_id: int,
    current: User = Depends(_current_user),
    db: Session = Depends(get_db),
):
    a = db.query(Analysis).filter(
        Analysis.id == analysis_id, Analysis.doctor_id == current.id
    ).first()
    if not a:
        raise HTTPException(404, "Analysis not found")
    db.delete(a)
    db.commit()


# ═══════════════════════════════════════════════════════════════════════════════
#  ADMIN ROUTES
# ═══════════════════════════════════════════════════════════════════════════════

@app.get("/api/admin/stats")
def admin_stats(admin: User = Depends(_admin_user), db: Session = Depends(get_db)):
    """Get platform-wide statistics (admin only)."""
    users    = db.query(User).filter(User.is_admin == False).all()  # noqa
    analyses = db.query(Analysis).all()

    class_dist  = {cls: 0 for cls in CLASSES}
    for a in analyses:
        class_dist[a.result] = class_dist.get(a.result, 0) + 1

    role_dist   = {"doctor": 0, "resident": 0, "student": 0}
    for u in users:
        role_dist[u.role] = role_dist.get(u.role, 0) + 1

    status_dist = {"verified": 0, "pending": 0, "suspended": 0}
    for u in users:
        status_dist[u.status] = status_dist.get(u.status, 0) + 1

    from collections import defaultdict
    daily = defaultdict(int)
    for a in analyses:
        if a.created_at:
            daily[a.created_at.strftime("%Y-%m-%d")] += 1

    per_user = sorted(
        [{"user_id": u.id, "name": u.name, "specialty": u.specialty,
          "count": sum(1 for a in analyses if a.doctor_id == u.id)} for u in users],
        key=lambda x: x["count"], reverse=True
    )

    return {
        "total_users"    : len(users),
        "total_analyses" : len(analyses),
        "status_dist"    : status_dist,
        "role_dist"      : role_dist,
        "class_dist"     : class_dist,
        "daily_analyses" : dict(sorted(daily.items())[-30:]),
        "top_users"      : per_user[:10],
        "avg_confidence" : round(sum(a.confidence for a in analyses) / len(analyses), 1) if analyses else 0,
    }


@app.get("/api/admin/users")
def admin_list_users(
    skip: int = 0, limit: int = 100,
    admin: User = Depends(_admin_user),
    db: Session = Depends(get_db),
):
    users = db.query(User).filter(User.is_admin == False).offset(skip).limit(limit).all()  # noqa
    return [_user_dict(u) for u in users]


@app.patch("/api/admin/users/{user_id}")
def admin_update_user(
    user_id: int,
    body: UpdateUserBody,
    admin: User = Depends(_admin_user),
    db: Session = Depends(get_db),
):
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(404, "User not found")
    if body.status   is not None: user.status   = body.status
    if body.role     is not None: user.role      = body.role
    if body.is_admin is not None: user.is_admin  = body.is_admin
    _audit(db, admin.id, "user_updated", f"user:{user_id}", f"status={body.status}")
    db.commit()
    return _user_dict(user)


@app.delete("/api/admin/users/{user_id}", status_code=204)
def admin_delete_user(
    user_id: int,
    admin: User = Depends(_admin_user),
    db: Session = Depends(get_db),
):
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(404, "User not found")
    _audit(db, admin.id, "user_deleted", f"user:{user_id}", user.email)
    db.delete(user)
    db.commit()


@app.get("/api/admin/analyses")
def admin_list_analyses(
    skip: int = 0, limit: int = 200,
    admin: User = Depends(_admin_user),
    db: Session = Depends(get_db),
):
    analyses = db.query(Analysis).order_by(Analysis.created_at.desc()).offset(skip).limit(limit).all()
    result = []
    for a in analyses:
        d = _analysis_dict(a)
        d["doctor_name"] = a.doctor.name if a.doctor else "Unknown"
        result.append(d)
    return result


@app.get("/api/health")
def health():
    """Health check endpoint."""
    return {"status": "ok", "version": "3.0.0", "app": "RhythmAI"}


_notif_cache: dict = {
    "data": [],
    "fetched_at": None,
    "unread_count": 0,
}
CACHE_TTL_MINUTES = 60


async def _refresh_notifications_job():
    """Scheduled job: fetch fresh ECG articles from PubMed daily."""
    global _notif_cache
    try:
        logger.info("Scheduled notification refresh started...")
        articles = await fetch_recent_ecg_articles(days=1, max_results=10)
        if articles:
            old_pmids = {a["pmid"] for a in _notif_cache.get("data", [])}
            new_articles = [a for a in articles if a["pmid"] not in old_pmids]
            merged = new_articles + _notif_cache.get("data", [])
            merged = merged[:30]
            _notif_cache = {
                "data": merged,
                "fetched_at": datetime.now(timezone.utc),
                "unread_count": _notif_cache.get("unread_count", 0) + len(new_articles),
            }
            logger.info(f"Notifications refreshed: {len(new_articles)} new articles found")
        else:
            logger.info("Notification refresh: no new articles found")
    except Exception:
        logger.exception("Notification refresh job failed")


from pydantic import BaseModel

class AIRequest(BaseModel):
    message: str
    messages: Optional[list[dict[str, str]]] = None
    system_prompt: Optional[str] = None
    pathology: Optional[str] = None
    patient_context: Optional[str] = None

AI_KNOWLEDGE_BASE = {
    "CD": {
        "title": "Conduction Disturbance",
        "summary": "Review QRS width, PR interval, AV block grade, bundle branch morphology, and symptoms such as syncope or presyncope.",
        "clinical_points": [
            "Correlate the ECG with symptoms, medication history, ischemia, and electrolyte abnormalities.",
            "High-grade AV block, alternating bundle branch block, or instability require urgent escalation.",
            "Consider pacing when bradycardia is symptomatic or conduction disease is advanced.",
        ],
        "guidelines": [
            "Rule out reversible causes first: drugs, hyperkalemia, ischemia, vagal triggers, and structural disease.",
            "Persistent Mobitz II or complete heart block generally needs electrophysiology/cardiology review.",
        ],
        "research": [
            "Recent literature emphasizes earlier recognition of infranodal disease and risk stratification with ambulatory monitoring and EP testing.",
            "Studies published in 2022-2025 continue to support symptom-guided pacing decisions and careful review of reversible triggers.",
        ],
        "keywords": ["cd", "conduction", "bundle branch", "av block", "brady", "heart block", "block"],
    },
    "ARR": {
        "title": "Arrhythmia",
        "summary": "Confirm the rhythm, assess hemodynamic stability, and identify whether the pattern is supraventricular or ventricular.",
        "clinical_points": [
            "Correct reversible triggers such as hypoxia, fever, infection, electrolyte imbalance, and ischemia.",
            "Rate control, rhythm control, or anticoagulation depends on the rhythm subtype and stroke risk.",
            "Wide-complex or unstable tachycardia should be treated as potentially dangerous until clarified.",
        ],
        "guidelines": [
            "Obtain a 12-lead ECG and compare with prior tracings whenever possible.",
            "Check QT interval, drug exposure, and structural heart disease when arrhythmia is recurrent.",
        ],
        "research": [
            "Recent work highlights continuous monitoring, wearable detection, and more individualized rhythm management.",
            "2022-2025 studies reinforce risk-based anticoagulation and early recognition of occult atrial arrhythmias.",
        ],
        "keywords": ["arrhythmia", "arr", "afib", "atrial fibrillation", "flutter", "tachy", "vt", "svt"],
    },
    "MI": {
        "title": "Myocardial Infarction",
        "summary": "Treat new ischemic symptoms or dynamic ST-T changes as time-sensitive acute coronary syndrome until excluded.",
        "clinical_points": [
            "Assess pain, troponin, ECG evolution, and contraindications to reperfusion.",
            "Urgently escalate if the patient is unstable, hypotensive, or has ongoing chest pain.",
            "Follow the local ACS pathway and document onset time carefully.",
        ],
        "guidelines": [
            "Serial ECGs and biomarkers are important when the first tracing is equivocal.",
            "STEMI needs rapid reperfusion planning; NSTEMI requires structured risk stratification.",
        ],
        "research": [
            "Recent studies continue to refine faster triage, prehospital recognition, and post-MI secondary prevention.",
            "2022-2025 evidence supports systems that reduce door-to-balloon delay and optimize long-term prevention.",
        ],
        "keywords": ["mi", "stemi", "nstemi", "infarction", "acs", "ischemia", "chest pain"],
    },
    "STTC": {
        "title": "ST/T Changes",
        "summary": "ST/T abnormalities are nonspecific without clinical context and should be compared with prior ECGs and biomarkers.",
        "clinical_points": [
            "Consider ischemia, electrolyte disorders, drugs, LVH, and repolarisation variants.",
            "Dynamic changes across serial ECGs matter more than a single isolated tracing.",
            "If symptoms are present, manage as possible ischemia until ruled out.",
        ],
        "guidelines": [
            "Use history, troponin, and repeat ECGs before labeling the change benign.",
            "Drug review and electrolyte correction are essential when repolarisation is abnormal.",
        ],
        "research": [
            "Recent literature focuses on better discrimination between benign repolarisation changes and ischemia.",
            "2022-2025 studies emphasize integrating symptoms, biomarkers, and serial tracings.",
        ],
        "keywords": ["st/t", "st depression", "t wave", "repolarisation", "repolarization", "sttc"],
    },
    "NORM": {
        "title": "Normal ECG",
        "summary": "The tracing is not showing a clear pathological pattern in the current classification.",
        "clinical_points": [
            "Confirm that lead quality and acquisition settings are adequate before relying on the result.",
            "If symptoms are concerning, consider repeat ECG, troponin, or rhythm monitoring.",
            "Clinical context can override a normal automated classification.",
        ],
        "guidelines": [
            "Always correlate with symptoms and exam findings.",
            "Do not dismiss syncope, chest pain, or dyspnea solely because the classification is normal.",
        ],
        "research": [
            "Recent work in ECG AI highlights that negative predictions are strongest when signal quality is good and symptoms are absent.",
            "2022-2025 papers continue to show the value of combining ECG interpretation with clinical metadata.",
        ],
        "keywords": ["normal", "no pathology", "norm"],
    },
}

SYSTEM_AGENT_PROMPT = """You are RhythmAI Clinical Assistant, a careful cardiology copilot.
You must answer in a concise, evidence-based, clinically useful way.
Rules:
- If the user asks for an exact diagnosis, explain the likely subtype and clearly state uncertainty when the input is insufficient.
- Do not invent ECG details that are not provided.
- Prefer actionable next steps, red flags, and differential diagnosis.
- If the question is outside cardiology, answer normally and briefly.
- Always mention when symptoms or instability require urgent evaluation.
"""


def _infer_pathology(message: str) -> str:
    text = message.lower()
    for key, entry in AI_KNOWLEDGE_BASE.items():
        if any(keyword in text for keyword in entry["keywords"]):
            return key
    return "NORM"


def _extract_user_question(message: str) -> str:
    for marker in ["user:", "assistant:"]:
        if marker in message.lower():
            pass
    parts = message.split("\n\n")
    tail = parts[-1].strip() if parts else message.strip()
    for prefix in ["user:", "User:"]:
        if tail.startswith(prefix):
            return tail[len(prefix):].strip()
    return tail


def _looks_french(text: str) -> bool:
    lowered = text.lower()
    return any(token in lowered for token in [" quelle ", "quel ", "quelle ", " amoxicilline", " adulte", " dosage", " posologie", " patient ", " traitement ", " douleur ", " merci "])


def _local_general_medical_response(body: AIRequest) -> str:
    messages = _normalize_messages(body)
    latest_user_message = ""
    for item in reversed(messages):
        if item.get("role") == "user" and item.get("content"):
            latest_user_message = item["content"]
            break
    if not latest_user_message:
        latest_user_message = body.message.strip()

    question = _extract_user_question(latest_user_message).lower()
    is_french = _looks_french(question + " " + (body.patient_context or ""))

    if "amoxicillin" in question or "amoxicilline" in question:
        if is_french:
            sections = [
                "Amoxicilline - posologie adulte",
                "Résumé: la dose dépend de l'indication clinique, de la sévérité de l'infection et de la fonction rénale.",
                "Réponse pratique:",
                "• Chez beaucoup d'adultes, on utilise souvent 500 mg toutes les 8 heures ou 875 mg toutes les 12 heures.",
                "• La dose exacte change selon le foyer infectieux, l'âge, le poids, l'allergie aux pénicillines et la clairance rénale.",
                "• Vérifiez les interactions, l'insuffisance rénale et les antécédents d'allergie avant de prescrire.",
                "• Ceci est une information générale, pas une prescription individuelle.",
                "Si vous me donnez l'indication clinique, je peux préciser une fourchette plus adaptée.",
            ]
        else:
            sections = [
                "Amoxicillin - adult dosing",
                "Summary: the dose depends on the indication, severity, renal function, and allergy history.",
                "Practical answer:",
                "• In many adults, common oral regimens are 500 mg every 8 hours or 875 mg every 12 hours.",
                "• The exact dose changes with the infection site, body size, kidney function, and local guidance.",
                "• Check penicillin allergy, renal impairment, and current medications before prescribing.",
                "• This is general information, not an individual prescription.",
                "If you share the indication, I can narrow it down further.",
            ]
        sections.append("This assistant is local and deterministic, so it stays usable even when external AI services are down.")
        return "\n".join(sections)

    if is_french:
        sections = [
            "Assistant médical général",
            "Résumé: je peux aider pour les questions médicales générales, les dosages usuels, les interactions, les diagnostics différentiels et les protocoles de prise en charge.",
            "Réponse pratique:",
            "• Donnez-moi l'âge, le sexe, les antécédents, les médicaments en cours et le contexte clinique pour une réponse plus précise.",
            "• Si vous cherchez un dosage, précisez l'indication, le poids et la fonction rénale.",
            "• Si les symptômes sont graves ou instables, il faut une évaluation urgente.",
            "Cette réponse est une aide à la décision et ne remplace pas le jugement clinique.",
        ]
    else:
        sections = [
            "General medical assistant",
            "Summary: I can help with general medical questions, common dosing ranges, drug interactions, differential diagnosis, and treatment protocols.",
            "Practical guidance:",
            "• Give me the age, sex, history, current medications, and the exact clinical context for a more precise answer.",
            "• If you want a dose, include the indication, weight, and renal function.",
            "• If symptoms are severe or unstable, urgent evaluation is needed.",
            "This is decision support only and does not replace clinical judgment.",
        ]

    sections.append("If you want, I can answer in a shorter bedside style or a more detailed teaching style.")
    sections.append("This assistant is local and deterministic, so it stays usable even when external AI services are down.")
    return "\n".join(sections)


def _local_ai_response(body: AIRequest) -> str:
    messages = _normalize_messages(body)
    latest_user_message = ""
    for item in reversed(messages):
        if item.get("role") == "user" and item.get("content"):
            latest_user_message = item["content"]
            break
    if not latest_user_message:
        latest_user_message = body.message.strip()

    question = _extract_user_question(latest_user_message).lower()
    conversation = "\n".join(item["content"] for item in messages if item.get("content"))

    if body.pathology is None:
        return _local_general_medical_response(body)

    pathology = body.pathology or _infer_pathology(conversation or latest_user_message)
    entry = AI_KNOWLEDGE_BASE.get(pathology, AI_KNOWLEDGE_BASE["NORM"])

    asks_exact_diagnosis = any(
        phrase in question
        for phrase in ["what disease", "what does he have", "what exactly", "exact disease", "which disease", "diagnosis exactly"]
    )
    asks_subtype = any(
        phrase in question
        for phrase in ["type", "subtype", "which block", "av block", "bundle branch", "mobitz", "complete heart block", "first degree"]
    )
    asks_if_ok = any(
        phrase in question
        for phrase in ["is he ok", "is she ok", "is patient ok", "is he safe", "is she safe", "is this dangerous", "serious", "stable"]
    )
    asks_management = any(
        phrase in question
        for phrase in ["what should", "management", "treat", "do", "next", "immediate", "urgent", "plan"]
    )
    asks_research = any(k in question for k in ["latest", "research", "recent", "update", "guideline", "guidelines"])
    asks_urgency = any(k in question for k in ["urgent", "emergency", "stable", "danger", "safe", "serious", "warning"])

    def is_urgent_pathology() -> bool:
        return pathology in {"MI", "ARR", "CD"} or any(k in question for k in ["chest pain", "syncope", "dyspnea", "hypotension"])

    sections: list[str] = [f"{entry['title']}", f"Summary: {entry['summary']}"]

    if body.patient_context:
        sections.append(f"Patient context: {body.patient_context}")

    if asks_if_ok or asks_urgency:
        sections.append("Clinical interpretation:")
        if pathology == "ARR":
            sections.extend([
                "• An arrhythmia label is not enough to say the patient is safe or unsafe on its own.",
                "• Decide urgency from symptoms, blood pressure, consciousness, chest pain, and the actual rhythm on the tracing.",
                "• If the patient has syncope, chest pain, hypotension, dyspnea, or sustained rapid rhythm, this needs urgent evaluation.",
            ])
        elif pathology == "MI":
            sections.extend([
                "• This should be treated as potentially time-sensitive until acute coronary syndrome is excluded.",
                "• Serial ECGs, troponin, pain history, and hemodynamic status matter more than a single automated label.",
            ])
        elif pathology == "CD":
            sections.extend([
                "• Conduction disease ranges from benign delay to high-grade block.",
                "• Syncope, severe bradycardia, pauses, or hypotension should be treated as urgent.",
            ])
        elif pathology == "STTC":
            sections.extend([
                "• ST/T changes are often nonspecific, but they become concerning when symptoms or biomarker changes are present.",
                "• Compare with prior tracings and look for dynamic evolution before calling it benign.",
            ])
        else:
            sections.extend([
                "• The current class does not by itself prove danger, but clinical context can override the automated result.",
                "• If symptoms are concerning, repeat the ECG and escalate the workup.",
            ])
        sections.append("This is decision support only; final judgment must be clinical.")

    if asks_exact_diagnosis:
        sections.append("Diagnostic nuance:")
        if pathology == "CD":
            sections.extend([
                "• The model currently flags conduction disturbance rather than a precise subtype.",
                "• To narrow it down, I need PR interval, QRS width, rhythm regularity, and the tracing itself.",
            ])
        else:
            sections.append("• The classifier suggests a category, but exact diagnosis still depends on the ECG, symptoms, and context.")

    if asks_subtype and pathology == "CD":
        sections.append("Subtype clues:")
        sections.extend([
            "• Prolonged PR with dropped beats points toward AV block.",
            "• Wide QRS with bundle morphology points toward bundle branch or infranodal disease.",
            "• Syncope or bradycardia raises concern for higher-grade block.",
        ])

    if asks_management:
        sections.append("Next steps:")
        if pathology == "MI":
            sections.extend([f"• {item}" for item in entry["clinical_points"][:2]])
            sections.append("• Use serial ECGs and troponin to confirm evolution.")
        elif pathology == "ARR":
            sections.extend([f"• {item}" for item in entry["clinical_points"]])
            sections.append("• Clarify the rhythm type before choosing antiarrhythmic or rate-control treatment.")
        else:
            sections.extend([f"• {item}" for item in entry["clinical_points"]])
            sections.extend([f"• {item}" for item in entry["guidelines"]])

    if asks_research:
        sections.append("Research & guidelines:")
        sections.extend([f"• {item}" for item in entry["research"]])
        sections.extend([f"• {item}" for item in entry["guidelines"]])

    if not any([asks_if_ok, asks_exact_diagnosis, asks_subtype, asks_management, asks_research]):
        sections.append("Clinical points:")
        sections.extend([f"• {item}" for item in entry["clinical_points"]])
        sections.append("What I would do next:")
        if is_urgent_pathology():
            sections.append("• Check symptoms, vitals, and the full ECG before assuming the label is benign.")
        sections.append("• Correlate the automated class with the patient story and prior ECGs.")

    if any(k in question for k in ["pacing", "pacemaker"]):
        sections.append("Pacing note: evaluate symptom burden, block grade, reversibility, and QRS width before deciding on pacing.")
    if any(k in question for k in ["anticoag", "stroke"]):
        sections.append("Anticoagulation note: confirm rhythm subtype and stroke risk factors before choosing therapy.")
    if any(k in question for k in ["troponin", "ischemia", "chest pain"]):
        sections.append("Ischemia note: serial ECGs and biomarkers matter more than a single tracing.")

    sections.append("If you want, send the ECG findings or a shorter clinical question and I can answer more precisely.")
    sections.append("This assistant is local and deterministic, so it stays usable even when external AI services are down.")
    return "\n".join(sections)


def _build_chat_prompt(message: str, pathology: Optional[str], patient_context: Optional[str]) -> str:
    context_lines = [SYSTEM_AGENT_PROMPT]
    if pathology:
        entry = AI_KNOWLEDGE_BASE.get(pathology, AI_KNOWLEDGE_BASE["NORM"])
        context_lines.append(f"Known pathology class: {pathology} ({entry['title']}).")
        context_lines.append(f"Clinical summary: {entry['summary']}")
    if patient_context:
        context_lines.append(f"Patient context: {patient_context}")
    context_lines.append("User question:")
    context_lines.append(message)
    return "\n".join(context_lines)


def _build_context_block(pathology: Optional[str], patient_context: Optional[str]) -> str:
    lines = []
    if pathology:
        entry = AI_KNOWLEDGE_BASE.get(pathology, AI_KNOWLEDGE_BASE["NORM"])
        lines.append(f"Known pathology class: {pathology} ({entry['title']}).")
        lines.append(f"Clinical summary: {entry['summary']}")
    if patient_context:
        lines.append(f"Patient context: {patient_context}")
    return "\n".join(lines)


async def _g4f_async_response(body: AIRequest) -> Optional[str]:
    if not _ensure_g4f_repo_on_path():
        return None

    try:
        from g4f.client import AsyncClient as G4FAsyncClient
        from g4f.Provider import PollinationsAI, DeepInfra, Yqcloud
        from g4f.Provider.hf_space import CohereForAI_C4AI_Command
    except Exception:
        logger.exception("g4f AsyncClient import failed")
        return None

    messages = _normalize_messages(body)
    if not messages:
        return None

    system_parts = [SYSTEM_AGENT_PROMPT]
    if body.system_prompt:
        system_parts.append(body.system_prompt)
    context_block = _build_context_block(body.pathology, body.patient_context)
    if context_block:
        system_parts.append(context_block)

    full_system_prompt = "\n\n".join(system_parts)
    payload_messages = [{"role": "system", "content": full_system_prompt}]
    payload_messages.extend([m for m in messages if m.get("role") != "system"])

    provider_chain = [
        (PollinationsAI, "openai-fast"),
        (Yqcloud, "gpt-4"),
        (CohereForAI_C4AI_Command, "command-a-03-2025"),
        (DeepInfra, "MiniMaxAI/MiniMax-M2.5"),
    ]

    content = None
    last_error = None
    for provider_cls, model_name in provider_chain:
        try:
            client = G4FAsyncClient(provider=provider_cls)
            response = await asyncio.wait_for(
                client.chat.completions.create(
                    model=model_name,
                    messages=payload_messages,
                ),
                timeout=12,
            )
            content = response.choices[0].message.content
            if content and len(content.strip()) > 5:
                logger.info(f"g4f success: provider={provider_cls.__name__} model={model_name}")
                break
        except Exception as e:
            last_error = e
            logger.warning(f"g4f provider {provider_cls.__name__}/{model_name} failed: {str(e)[:80]}")
            continue

    if not content:
        raise Exception(f"All providers failed. Last: {last_error}")

    try:
        content = content.strip()
    except Exception:
        content = None

    return content or None


def _resolve_g4f_repo_path() -> Optional[Path]:
    env_path = os.getenv("G4F_REPO_PATH", "").strip()
    if env_path:
        candidate = Path(env_path)
        if candidate.is_dir():
            return candidate

    candidate = Path(r"c:\Users\User\Downloads\rhythmai-v0-main\gpt4free-main")
    if candidate.is_dir():
        return candidate

    for parent in Path(__file__).resolve().parents:
        candidate = parent / "gpt4free-main"
        if candidate.is_dir():
            return candidate
    return None


def _ensure_g4f_repo_on_path() -> bool:
    repo_path = _resolve_g4f_repo_path()
    if not repo_path:
        return False
    repo_str = str(repo_path)
    if repo_str not in sys.path:
        sys.path.insert(0, repo_str)
    return True


def _local_g4f_response(body: AIRequest, messages: list[dict[str, str]], full_system_prompt: str) -> Optional[str]:
    # Disabled legacy sync g4f path; the async PollinationsAI path is the only g4f route now.
    return None


def _normalize_messages(body: AIRequest) -> list[dict[str, str]]:
    msgs: list[dict[str, str]] = []
    for item in (body.messages or []):
        role = str(item.get("role", "user")).lower().strip()
        if role not in {"user", "assistant", "system"}:
            role = "user"
        content = str(item.get("content", "")).strip()
        if content:
            msgs.append({"role": role, "content": content})
    if not msgs and body.message.strip():
        msgs.append({"role": "user", "content": body.message.strip()})
    return msgs


def _remote_ai_response(body: AIRequest) -> Optional[str]:
    messages = _normalize_messages(body)
    if not messages:
        return None

    context_block = _build_context_block(body.pathology, body.patient_context)
    system_parts = [SYSTEM_AGENT_PROMPT]
    if body.system_prompt:
        system_parts.append(body.system_prompt)
    if context_block:
        system_parts.append(context_block)
    full_system_prompt = "\n\n".join(system_parts)

    provider = os.getenv("LLM_PROVIDER", "g4f").strip().lower()

    local_first = provider in {"auto", "g4f", "local", "openai"}
    if local_first:
        local_text = _local_g4f_response(body, messages, full_system_prompt)
        if local_text:
            return local_text

    if provider == "custom":
        custom_url = os.getenv("LLM_CUSTOM_URL", "").strip()
        if not custom_url:
            return None

        try:
            import requests

            headers = {"Accept": "application/json", "Content-Type": "application/json"}
            auth_header = os.getenv("LLM_CUSTOM_AUTH_HEADER", "").strip()
            auth_value = os.getenv("LLM_CUSTOM_AUTH_VALUE", "").strip()
            if auth_header and auth_value:
                headers[auth_header] = auth_value

            payload = {
                "message": body.message,
                "messages": messages,
                "system_prompt": full_system_prompt,
                "pathology": body.pathology,
                "patient_context": body.patient_context,
            }

            res = requests.post(custom_url, headers=headers, json=payload, timeout=60)
            if not res.text.strip():
                return None

            data = res.json()

            if isinstance(data, dict):
                if isinstance(data.get("generated_text"), str) and data["generated_text"].strip():
                    return data["generated_text"].strip()
                if isinstance(data.get("response"), str) and data["response"].strip():
                    return data["response"].strip()
                if isinstance(data.get("content"), str) and data["content"].strip():
                    return data["content"].strip()
                if isinstance(data.get("message"), str) and data["message"].strip():
                    return data["message"].strip()
                choices = data.get("choices")
                if isinstance(choices, list) and choices:
                    first = choices[0]
                    if isinstance(first, dict):
                        msg = first.get("message")
                        if isinstance(msg, dict) and isinstance(msg.get("content"), str) and msg["content"].strip():
                            return msg["content"].strip()
                        if isinstance(first.get("text"), str) and first["text"].strip():
                            return first["text"].strip()

            if isinstance(data, list) and data:
                first = data[0]
                if isinstance(first, dict):
                    if isinstance(first.get("generated_text"), str) and first["generated_text"].strip():
                        return first["generated_text"].strip()
                    if isinstance(first.get("content"), str) and first["content"].strip():
                        return first["content"].strip()
                if isinstance(first, str) and first.strip():
                    return first.strip()

            return None
        except Exception:
            return None

    if provider == "openai":
        api_base = os.getenv("LLM_API_BASE_URL", "http://localhost:1337/v1").strip().rstrip("/")
        api_key = os.getenv("LLM_API_KEY", "").strip()
        model = os.getenv("LLM_MODEL", "gpt-4o-mini").strip()
        if not api_base:
            return None

        try:
            import requests

            headers = {"Accept": "application/json"}
            if api_key:
                headers["Authorization"] = f"Bearer {api_key}"

            payload = {
                "model": model,
                "messages": [
                    {"role": "system", "content": full_system_prompt},
                    *messages,
                ],
                "temperature": 0.3,
                "stream": False,
            }
            res = requests.post(f"{api_base}/chat/completions", headers=headers, json=payload, timeout=45)
            if not res.text.strip():
                return None
            data = res.json()
            return data["choices"][0]["message"]["content"].strip()
        except Exception:
            return None

    return None

@app.post("/api/ai")
async def ai_chat(body: AIRequest):
    """AI chat endpoint for clinical Q&A about ECG pathologies."""
    remote_text = None
    try:
        try:
            remote_text = await asyncio.wait_for(_g4f_async_response(body), timeout=20)
        except asyncio.TimeoutError:
            logger.warning("g4f timed out after 20s")
            remote_text = None
    except Exception:
        logger.exception("g4f AsyncClient error")

    if remote_text:
        return [{"generated_text": remote_text}]

    try:
        local_text = _local_ai_response(body)
        if local_text:
            logger.info("Using local fallback")
            return [{"generated_text": local_text}]
    except Exception:
        logger.exception("Local fallback failed")

    raise HTTPException(status_code=503, detail="AI service unavailable")


@app.get("/api/notifications")
async def get_notifications(authorization: str = Header(None), db: Session = Depends(get_db)):
    """Get recent ECG-related PubMed articles."""
    _current_user(authorization, db)
    return {
        "notifications": _notif_cache["data"],
        "count": len(_notif_cache["data"]),
        "unread_count": _notif_cache.get("unread_count", 0),
        "last_updated": _notif_cache["fetched_at"].isoformat()
        if _notif_cache["fetched_at"]
        else None,
    }


@app.post("/api/notifications/mark-read")
async def mark_notifications_read(authorization: str = Header(None), db: Session = Depends(get_db)):
    _current_user(authorization, db)
    global _notif_cache
    _notif_cache["unread_count"] = 0
    return {"status": "ok"}