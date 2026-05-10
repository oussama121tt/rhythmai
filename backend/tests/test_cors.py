import requests

opts = requests.options('http://127.0.0.1:8001/api/auth/login', headers={'Origin':'http://localhost:5173','Access-Control-Request-Method':'POST','Access-Control-Request-Headers':'content-type'})
print('OPTIONS', opts.status_code)
print('ACAO:', opts.headers.get('access-control-allow-origin'))
print('ACAC:', opts.headers.get('access-control-allow-credentials'))
post = requests.post('http://127.0.0.1:8001/api/auth/login', json={'email':'amira@hopital-charles.tn','password':'doctor123'}, headers={'Origin':'http://localhost:5173'})
print('POST', post.status_code)
print('ACAO:', post.headers.get('access-control-allow-origin'))
print('ACAC:', post.headers.get('access-control-allow-credentials'))
print('BODY:', post.text)
