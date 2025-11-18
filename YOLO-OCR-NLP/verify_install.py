# verify_install.py
import importlib, sys, platform

def check(mod, name=None):
    name = name or mod
    try:
        importlib.import_module(mod)
        print(f" {name} OK")
        return True
    except Exception as e:
        print(f" {name} FAILED → {e}")
        return False

modules = [
    ("numpy", "NumPy"),
    ("cv2", "OpenCV"),
    ("albumentations", "Albumentations"),
    ("paddle", "PaddlePaddle"),
    ("paddleocr", "PaddleOCR"),
    ("ultralytics", "YOLOv8"),
]

ok = sum(check(m, n) for m, n in modules)

print("\n--- Versions ---")
try:
    import numpy as np; print("NumPy:", np.__version__)
except: pass
try:
    import cv2; print("OpenCV:", cv2.__version__)
except: pass
try:
    import paddle
    print("Paddle:", paddle.__version__)
    print("CUDA compiled:", paddle.is_compiled_with_cuda())
except Exception as e:
    print("Paddle version check failed:", e)

print(f"\n Passed {ok}/{len(modules)} checks")
print("Python:", sys.version.split()[0], "| OS:", platform.system())
