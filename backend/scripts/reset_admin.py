from database import SessionLocal
from models import User
from auth import hash_password
import os

db = SessionLocal()

user = db.query(User).filter(User.email == "admin@rhythmai.ai").first()

if user:
    user.password_hash = hash_password(os.getenv("ADMIN_PASSWORD", "admin2026"))
    db.commit()
    print("Admin password reset OK")
else:
    print("Admin not found")

db.close()