## Scripts

Administrative and utility scripts for RhythmAI backend setup and maintenance.

### Usage

```bash
# Initialize admin account
python scripts/reset_admin.py

# Reset a specific user password
python scripts/reset_amira_password.py

# List registered API routes
python scripts/check_routes.py
```

### Script descriptions

- `reset_admin.py` — Create or reset the default admin account (`admin@rhythmai.ai`)
- `reset_amira_password.py` — Reset password for a specific demo user (Amira)
- `check_routes.py` — Display all registered API endpoints and their methods

**⚠️ Warning:** These scripts directly modify the database. Only run when the application is not in use.
