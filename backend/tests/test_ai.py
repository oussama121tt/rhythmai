import requests

payload = {
    "message": "Patient has arrhythmia. is he ok?",
    "messages": None,
    "system_prompt": None,
    "pathology": "ARR",
    "patient_context": "Age: 67, Sex: M",
}
res = requests.post('http://127.0.0.1:8001/api/ai', json=payload, timeout=30)
print(res.status_code)
print(res.text)
