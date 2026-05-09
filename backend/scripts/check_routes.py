from main import app

print("Registered routes:")
for route in app.routes:
    print(f"  {route.path if hasattr(route, 'path') else route}")
