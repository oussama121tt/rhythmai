import requests

# Test register endpoint
response = requests.post(
    'http://localhost:8000/api/auth/register',
    data={
        'name': 'Test',
        'email': 'test@example.com',
        'password': 'test123',
        'role': 'doctor',
        'medical_id': 'MED-123',
    }
)
print(f"Register Status: {response.status_code}")
print(f"Register Response: {response.json()}")
