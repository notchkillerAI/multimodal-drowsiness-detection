"""Download the model files the notebooks need. Run once:  python3 download_models.py"""
import bz2, os, shutil, urllib.request

FILES = {
    # OpenVINO open-closed-eye-0001 (used by DL_classifier.ipynb)
    "models/eye_classifier.onnx": (
        "https://storage.openvinotoolkit.org/repositories/open_model_zoo/"
        "public/2022.1/open-closed-eye-0001/open_closed_eye.onnx", False),
    # dlib 68-point landmark model (used by EAR_Detection.ipynb)
    "shape_predictor_68_face_landmarks.dat": (
        "https://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2", True),
}

os.makedirs("models", exist_ok=True)
for dest, (url, is_bz2) in FILES.items():
    if os.path.exists(dest):
        print(f"already have {dest}")
        continue
    print(f"downloading {dest} ...")
    tmp = dest + (".bz2" if is_bz2 else ".part")
    urllib.request.urlretrieve(url, tmp)
    if is_bz2:
        with bz2.open(tmp, "rb") as src, open(dest, "wb") as out:
            shutil.copyfileobj(src, out)
        os.remove(tmp)
    else:
        os.replace(tmp, dest)
    print(f"  saved {dest} ({os.path.getsize(dest)/1e6:.1f} MB)")
print("done")
