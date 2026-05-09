import requests

response = requests.post(
    'http://localhost:8000/api/auth/login',
    json={'email': 'admin@rhythmai.ai', 'password': 'admin2026'}
)
print(f"Status: {response.status_code}")
print(f"Response: {response.json()}")
