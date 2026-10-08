# Eye-state model comparison

* **MRL test**: 6000 random eye images from 8 held-out people (subject ids [11, 12, 16, 17, 21, 23, 29, 32]), infrared, one eye per image. None of these people were used to train our model.
* **Webcam test**: 206 eye crops (colour) from one video of one person; labels from a 3 s deliberate closure and clearly open frames. Single subject, so only a sanity check of the colour-webcam domain.
* `false open` = closed eye called open (dangerous). `false closed` = open eye called closed (false alarm).
* ViT (dima806) was trained on a Kaggle eye dataset that is probably related to MRL, so its MRL numbers may be optimistic.
* Old model output order chosen on validation people: output 0 = closed.

| Model | Params | Licence | MRL acc | MRL false open | MRL false closed | Webcam acc | Webcam false open | Webcam false closed | CPU ms/eye |
|---|---|---|---|---|---|---|---|---|---|
| OpenVINO open-closed-eye-0001 (old) | 0.05 M | Apache-2.0 | 57.6% | 100.0% | 0.1% | 68.9% | 100.0% | 0.0% | 0.1 |
| MobileNetV2 (Michal M., HF) | 4.4 M | CC BY-NC-ND 4.0 | 52.4% | 0.0% | 82.6% | 90.8% | 0.0% | 13.4% | 6.6 |
| ViT-base (dima806, HF) | 86 M | Apache-2.0 | 99.2% | 1.3% | 0.5% | 100.0% | 0.0% | 0.0% | 48.1 |
| MobileNetV3 fine-tuned (ours) | 1.5 M | own weights; torchvision BSD-3 | 97.2% | 1.1% | 4.0% | 100.0% | 0.0% | 0.0% | 0.8 |

Our model with ONNX Runtime CoreML provider: 2.3 ms per eye.
