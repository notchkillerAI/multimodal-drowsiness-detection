"""Train and compare open/closed eye classifiers.

    python3 train_eye_model.py train [small|large] [size] [epochs]   # fine-tune MobileNetV3 on MRL Eye -> models/eye_state_<arch>_<size>.onnx
    python3 train_eye_model.py compare   # same held-out eyes for every candidate -> docs/eye_model_comparison.md

Data: MRL Eye Dataset (http://mrl.cs.vsb.cz/eyedataset), unzipped to data/mrl/mrlEyes_2018_01/.
84,898 eye crops of 37 people. Split by PERSON (never by image), so the test people are never
seen in training. Label 1 = closed (the model outputs p_closed).
"""
import glob, json, os, statistics, sys, time
import cv2
import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
MRL_GLOB = os.path.join(ROOT, "data/mrl/mrlEyes_2018_01/s*/*.png")
CACHE = os.path.join(ROOT, "data/mrl_{size}.npz")
WEBCAM_EVAL = os.path.join(ROOT, "data/webcam_eval.npz")        # 96x96 BGR eye crops from the user's own video
MODEL_OUT = os.path.join(ROOT, "models/eye_state.onnx")           # the chosen model (copied here after training)
N_TEST_IMAGES = 6000                                             # same random test images for every candidate


# ----------------------------------------------------------------------------- data
def download_mrl():
    """Fetch and unzip the MRL Eye dataset (340 MB) if it is not there yet."""
    if glob.glob(MRL_GLOB):
        return
    import urllib.request, zipfile
    os.makedirs(os.path.join(ROOT, "data/mrl"), exist_ok=True)
    zip_path = os.path.join(ROOT, "data/mrl/mrl.zip")
    print("downloading MRL Eye dataset (340 MB) ...")
    urllib.request.urlretrieve("http://mrl.cs.vsb.cz/data/eyedataset/mrlEyes_2018_01.zip", zip_path)
    zipfile.ZipFile(zip_path).extractall(os.path.join(ROOT, "data/mrl"))
    os.remove(zip_path)


def list_mrl():
    """File names look like s0001_00001_0_0_0_0_0_01.png: subject, image, gender, glasses, STATE (0 = closed), ..."""
    files = sorted(glob.glob(MRL_GLOB))
    subject = np.array([int(os.path.basename(f)[1:5]) for f in files])
    closed = np.array([os.path.basename(f).split("_")[4] == "0" for f in files]).astype(np.float32)
    return np.array(files), subject, closed


def split_subjects(subject, seed=0):
    """Whole people go to one split: 8 test, 4 validation, the rest train."""
    people = np.random.RandomState(seed).permutation(np.unique(subject))
    return {"test": people[:8], "val": people[8:12], "train": people[12:]}


def load_gray(files, size):
    """Read every image once, resize to size x size gray, cache (85k images, a few hundred MB)."""
    cache = CACHE.format(size=size)
    if os.path.exists(cache):
        return np.load(cache)["x"]
    x = np.stack([cv2.resize(cv2.imread(f, 0), (size, size), interpolation=cv2.INTER_AREA) for f in files])
    np.savez(cache, x=x)
    return x


# ----------------------------------------------------------------------------- our model
def build_model(arch="small"):
    import torch, torch.nn as nn
    from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights, mobilenet_v3_large, MobileNet_V3_Large_Weights

    class EyeNet(nn.Module):
        """Input: gray eye crop [N,1,size,size] in 0..1. Output: probability that the eye is closed."""
        def __init__(self):
            super().__init__()
            if arch == "small":
                self.net = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.IMAGENET1K_V1)
                self.net.classifier[-1] = nn.Linear(1024, 1)
            else:
                self.net = mobilenet_v3_large(weights=MobileNet_V3_Large_Weights.IMAGENET1K_V1)
                self.net.classifier[-1] = nn.Linear(1280, 1)

        def forward(self, x):
            x = (x - 0.45) / 0.225                    # ImageNet statistics, gray version
            return torch.sigmoid(self.net(x.repeat(1, 3, 1, 1))).squeeze(1)

    return EyeNet()


def augment(x):
    """Random changes on a GPU batch so the model survives webcams, light and glasses:
    small rotation / zoom / shift, mirror, brightness, contrast, gamma, blur, noise."""
    import torch, torch.nn.functional as F
    n, dev = x.shape[0], x.device
    angle = (torch.rand(n, device=dev) - 0.5) * 2 * np.radians(12)
    zoom = 0.9 + 0.25 * torch.rand(n, device=dev)
    flip = torch.where(torch.rand(n, device=dev) < 0.5, -1.0, 1.0)
    theta = torch.zeros(n, 2, 3, device=dev)
    theta[:, 0, 0], theta[:, 0, 1] = torch.cos(angle) / zoom * flip, -torch.sin(angle) / zoom
    theta[:, 1, 0], theta[:, 1, 1] = torch.sin(angle) / zoom * flip, torch.cos(angle) / zoom
    theta[:, :, 2] = (torch.rand(n, 2, device=dev) - 0.5) * 0.16
    x = F.grid_sample(x, F.affine_grid(theta, x.shape, align_corners=False), padding_mode="border", align_corners=False)

    x = x.clamp(1e-3, 1) ** (0.6 + torch.rand(n, 1, 1, 1, device=dev))                      # gamma
    x = (x - 0.5) * (0.6 + 0.8 * torch.rand(n, 1, 1, 1, device=dev)) + 0.5                  # contrast
    x = x + (torch.rand(n, 1, 1, 1, device=dev) - 0.5) * 0.3                                # brightness
    blur = F.avg_pool2d(x, 3, 1, 1)
    x = torch.where(torch.rand(n, 1, 1, 1, device=dev) < 0.3, blur, x)                      # blur
    x = x + torch.randn_like(x) * torch.rand(n, 1, 1, 1, device=dev) * 0.05                 # noise
    return x.clamp(0, 1)


def train(arch="small", size=64, epochs=6, batch=256):
    size, epochs = int(size), int(epochs)
    import torch
    download_mrl()
    files, subject, closed = list_mrl()
    x = load_gray(files, size)
    parts = split_subjects(subject)
    idx = {k: np.where(np.isin(subject, v))[0] for k, v in parts.items()}
    print({k: len(v) for k, v in idx.items()}, "images; test people:", sorted(parts["test"]))

    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    to_t = lambda i: torch.from_numpy(x[i]).float().div(255).unsqueeze(1)
    x_tr, y_tr = to_t(idx["train"]).to(dev), torch.from_numpy(closed[idx["train"]]).to(dev)
    x_va, y_va = to_t(idx["val"]).to(dev), torch.from_numpy(closed[idx["val"]]).to(dev)

    model = build_model(arch).to(dev)
    steps = epochs * (len(x_tr) // batch)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=2e-3, total_steps=steps)
    best = (0.0, None)

    for epoch in range(epochs):
        model.train()
        order = torch.randperm(len(x_tr), device=dev)
        for s in range(len(x_tr) // batch):
            b = order[s * batch:(s + 1) * batch]
            loss = torch.nn.functional.binary_cross_entropy(model(augment(x_tr[b])), y_tr[b])
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        model.eval()
        with torch.no_grad():
            pred = torch.cat([model(x_va[i:i + 1024]) for i in range(0, len(x_va), 1024)])
        acc = ((pred > 0.5).float() == y_va).float().mean().item()
        print(f"epoch {epoch + 1}/{epochs}  last train loss {loss.item():.3f}  validation accuracy {acc:.4f}")
        if acc > best[0]:
            best = (acc, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()})

    model.load_state_dict(best[1])
    model.cpu().eval()
    out = os.path.join(ROOT, f"models/eye_state_{arch}_{size}.onnx")
    torch.onnx.export(model, torch.zeros(1, 1, size, size), out, input_names=["eye"], output_names=["p_closed"],
                      dynamic_axes={"eye": {0: "n"}, "p_closed": {0: "n"}}, opset_version=17, dynamo=False)
    print(f"best validation accuracy {best[0]:.4f}; saved {out} ({os.path.getsize(out) / 1e6:.1f} MB)")


# ----------------------------------------------------------------------------- candidates
# Every candidate is a function: list of BGR uint8 eye crops -> numpy array of p_closed.
def ort_session(path, providers=("CPUExecutionProvider",)):
    import onnxruntime as ort
    return ort.InferenceSession(path, providers=list(providers))


def ours(path=MODEL_OUT, providers=("CPUExecutionProvider",)):
    sess = ort_session(path, providers)
    size = sess.get_inputs()[0].shape[-1]                     # the model knows its own input size
    def predict(crops):
        x = np.stack([cv2.resize(cv2.cvtColor(c, cv2.COLOR_BGR2GRAY), (size, size), interpolation=cv2.INTER_AREA)
                      for c in crops]).astype(np.float32)[:, None] / 255
        return sess.run(None, {"eye": x})[0]
    return predict


def openvino_baseline(closed_index=0):
    """The old model (OpenVINO open-closed-eye-0001, 32x32 BGR, 2 outputs). Which output is 'closed' is
    not stated in the old notebook (two comments disagree) so `closed_index` is chosen on validation people."""
    net = cv2.dnn.readNetFromONNX(os.path.join(ROOT, "models/eye_classifier.onnx"))
    def predict(crops):
        out = []
        for c in crops:
            net.setInput(cv2.dnn.blobFromImage(cv2.resize(c, (32, 32), interpolation=cv2.INTER_AREA), 1 / 255.0))
            p = net.forward().flatten()
            p = np.exp(p - p.max()); p /= p.sum()
            out.append(p[closed_index])
        return np.array(out)
    return predict


def michal_mobilenetv2():
    """MichalMlodawski/open-closed-eye-classification-mobilev2 (CC BY-NC-ND 4.0, trained on AI-generated eyes)."""
    sess = ort_session(os.path.join(ROOT, "models/candidates/michal_mbv2.onnx"))
    name = sess.get_inputs()[0].name
    def predict(crops):
        x = np.stack([cv2.resize(cv2.cvtColor(c, cv2.COLOR_BGR2RGB), (224, 224)) for c in crops]).astype(np.float32)
        x = ((x / 255 - 0.5) / 0.5).transpose(0, 3, 1, 2)
        logits = sess.run(None, {name: x})[0]
        p = np.exp(logits - logits.max(1, keepdims=True)); p /= p.sum(1, keepdims=True)
        return p[:, 0]                                            # model card: 0 = closed, 1 = open
    return predict


def dima806_vit():
    """dima806/closed_eyes_image_detection (ViT-base, Apache-2.0)."""
    import torch
    from transformers import AutoImageProcessor, AutoModelForImageClassification
    repo = "dima806/closed_eyes_image_detection"
    proc, model = AutoImageProcessor.from_pretrained(repo), AutoModelForImageClassification.from_pretrained(repo).eval()
    closed_id = [i for i, n in model.config.id2label.items() if "close" in n.lower()][0]
    def predict(crops):
        inputs = proc([cv2.cvtColor(c, cv2.COLOR_BGR2RGB) for c in crops], return_tensors="pt")
        with torch.no_grad():
            return model(**inputs).logits.softmax(-1)[:, closed_id].numpy()
    return predict


# ----------------------------------------------------------------------------- evaluation
def metrics(p_closed, truth_closed, threshold=0.5):
    pred = p_closed > threshold
    t = truth_closed.astype(bool)
    return {
        "accuracy": float((pred == t).mean()),
        "closed_recall": float(pred[t].mean()),              # closed eyes that were caught
        "open_recall": float((~pred[~t]).mean()),
        "false_open_rate": float((~pred[t]).mean()),         # closed eye called open: the dangerous mistake
        "false_closed_rate": float(pred[~t].mean()),         # open eye called closed: a false alarm
    }


def latency_ms(predict, crop, runs=100):
    for _ in range(5):
        predict([crop])
    times = []
    for _ in range(runs):
        t = time.perf_counter(); predict([crop]); times.append(time.perf_counter() - t)
    return 1000 * statistics.median(times)


def run_in_batches(predict, crops, size=64):
    return np.concatenate([predict(crops[i:i + size]) for i in range(0, len(crops), size)])


def compare():
    files, subject, closed = list_mrl()
    parts = split_subjects(subject)

    # fixed random test images from the held-out people, loaded at their original size
    rng = np.random.RandomState(1)
    test_idx = rng.choice(np.where(np.isin(subject, parts["test"]))[0], N_TEST_IMAGES, replace=False)
    test_crops = [cv2.cvtColor(cv2.imread(files[i], 0), cv2.COLOR_GRAY2BGR) for i in test_idx]
    test_truth = closed[test_idx]

    # choose the old model's output order on validation people (a fair chance for the baseline)
    val_idx = np.random.RandomState(2).choice(np.where(np.isin(subject, parts["val"]))[0], 600, replace=False)
    val_crops = [cv2.cvtColor(cv2.imread(files[i], 0), cv2.COLOR_GRAY2BGR) for i in val_idx]
    acc = {k: metrics(openvino_baseline(k)(val_crops), closed[val_idx])["accuracy"] for k in (0, 1)}
    closed_index = max(acc, key=acc.get)
    print("old model: validation accuracy by assumed 'closed' output", acc, "-> using", closed_index)

    web = np.load(WEBCAM_EVAL) if os.path.exists(WEBCAM_EVAL) else None
    web_crops, web_truth = (list(web["X"]), (web["y"] == 0).astype(np.float32)) if web is not None else (None, None)

    candidates = {
        "OpenVINO open-closed-eye-0001 (old)": (openvino_baseline(closed_index), "Apache-2.0", "0.05 M"),
        "MobileNetV2 (Michal M., HF)": (michal_mobilenetv2(), "CC BY-NC-ND 4.0", "4.4 M"),
        "ViT-base (dima806, HF)": (dima806_vit(), "Apache-2.0", "86 M"),
        "MobileNetV3 fine-tuned (ours)": (ours(), "own weights; torchvision BSD-3", "1.5 M"),
    }
    results = {}
    for name, (predict, licence, params) in candidates.items():
        row = {"licence": licence, "params": params}
        row["mrl"] = metrics(run_in_batches(predict, test_crops), test_truth)
        if web_crops is not None:
            row["webcam"] = metrics(run_in_batches(predict, web_crops), web_truth)
        row["ms_per_eye_cpu"] = latency_ms(predict, test_crops[0])
        results[name] = row
        print(name, json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in row["mrl"].items()}),
              "| webcam acc", round(row["webcam"]["accuracy"], 3) if web_crops is not None else "-",
              f"| {row['ms_per_eye_cpu']:.1f} ms")

    # speed of our model with the Apple Neural Engine / GPU path
    results["ours_coreml_ms"] = latency_ms(ours(providers=("CoreMLExecutionProvider", "CPUExecutionProvider")), test_crops[0])
    write_report(results, closed_index, len(web_crops) if web_crops is not None else 0, parts)


def write_report(results, closed_index, n_web, parts):
    os.makedirs(os.path.join(ROOT, "docs"), exist_ok=True)
    pct = lambda v: f"{100 * v:.1f}%"
    lines = [
        "# Eye-state model comparison", "",
        f"* **MRL test**: {N_TEST_IMAGES} random eye images from 8 held-out people (subject ids {sorted(int(s) for s in parts['test'])}), "
        "infrared, one eye per image. None of these people were used to train our model.",
        f"* **Webcam test**: {n_web} eye crops (colour) from one video of one person; labels from a 3 s deliberate closure and "
        "clearly open frames. Single subject, so only a sanity check of the colour-webcam domain.",
        "* `false open` = closed eye called open (dangerous). `false closed` = open eye called closed (false alarm).",
        "* ViT (dima806) was trained on a Kaggle eye dataset that is probably related to MRL, so its MRL numbers may be optimistic.",
        f"* Old model output order chosen on validation people: output {closed_index} = closed.", "",
        "| Model | Params | Licence | MRL acc | MRL false open | MRL false closed | Webcam acc | Webcam false open | Webcam false closed | CPU ms/eye |",
        "|---|---|---|---|---|---|---|---|---|---|"]
    for name, r in results.items():
        if name == "ours_coreml_ms":
            continue
        w = r.get("webcam")
        lines.append(f"| {name} | {r['params']} | {r['licence']} | {pct(r['mrl']['accuracy'])} | {pct(r['mrl']['false_open_rate'])} | "
                     f"{pct(r['mrl']['false_closed_rate'])} | " +
                     (f"{pct(w['accuracy'])} | {pct(w['false_open_rate'])} | {pct(w['false_closed_rate'])} | " if w else "- | - | - | ") +
                     f"{r['ms_per_eye_cpu']:.1f} |")
    lines += ["", f"Our model with ONNX Runtime CoreML provider: {results['ours_coreml_ms']:.1f} ms per eye."]
    with open(os.path.join(ROOT, "docs/eye_model_comparison.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    with open(os.path.join(ROOT, "docs/eye_model_comparison.json"), "w") as f:
        json.dump(results, f, indent=1)
    print("\n".join(lines))


if __name__ == "__main__":
    {"train": train, "compare": compare}[sys.argv[1]](*sys.argv[2:])
