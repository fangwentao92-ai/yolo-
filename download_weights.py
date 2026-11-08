import os
import urllib.request

BASE = os.path.dirname(os.path.abspath(__file__))

# Each weight is tried in order: hf-mirror first (works in mainland China),
# official GitHub release as fallback.
FILES = {
    'yolo11n.pt': [
        'https://hf-mirror.com/Ultralytics/YOLO11/resolve/main/yolo11n.pt',
        'https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt',
    ],
    'yolo11n-pose.pt': [
        'https://hf-mirror.com/Ultralytics/YOLO11/resolve/main/yolo11n-pose.pt',
        'https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n-pose.pt',
    ],
}

for name, urls in FILES.items():
    dst = os.path.join(BASE, name)
    if os.path.exists(dst):
        print('already exists:', name)
        continue
    for url in urls:
        try:
            print('downloading:', url)
            urllib.request.urlretrieve(url, dst)
            print('done:', name)
            break
        except Exception as exc:
            print('failed:', exc)
    else:
        raise SystemExit('could not download ' + name)

print('all weights ready')
