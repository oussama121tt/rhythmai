import requests

response = requests.get('http://localhost:8000/api/health')
print(f"Health check Status: {response.status_code}")
print(f"Response: {response.text}")
