## Tests

Unit and integration test files for the RhythmAI backend.

### Running tests

```bash
# Run all tests
pytest

# Run specific test file
pytest test_login.py

# Run with verbose output
pytest -v
```

### Test files

- `test_health.py` — Health check endpoint
- `test_login.py` — Authentication endpoint
- `test_register.py` — User registration endpoint
- `test_ai.py` — AI chat endpoint
- `test_cors.py` — CORS configuration validation
- `test_final.py` — End-to-end flow validation
- `test_get.py` — GET endpoint tests
- `test_register_endpoint.py` — Extended registration tests

**Note:** Tests are for development only and should not be run in production environments.
