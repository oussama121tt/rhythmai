import os
os.environ['FORCE_SIMPLE_HASH'] = '1'
from database import SessionLocal, User
from auth import hash_password

email = 'amira@hopital-charles.tn'
new_pw = 'doctor123'

db = SessionLocal()
try:
    user = db.query(User).filter(User.email == email).first()
    if not user:
        print('User not found')
    else:
        print('Before:', user.password_hash)
        user.password_hash = hash_password(new_pw)
        db.commit()
        print('Password updated for', email)
        print('After:', user.password_hash)
finally:
    db.close()
