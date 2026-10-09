import numpy as np

from drowsiness.video.cnn_branch import crop_eye, input_confidence


def test_crop_reports_how_much_is_inside_the_frame():
    frame = np.zeros((100, 100, 3), np.uint8)
    crop, inside = crop_eye(frame, (50, 50), 20)
    assert crop.shape[:2] == (20, 20) and inside == 1.0
    crop, inside = crop_eye(frame, (0, 50), 20)  # half of the square is outside
    assert inside == 0.5 and crop.shape[1] == 10
    assert crop_eye(frame, (-50, 50), 20) == (None, 0.0)  # completely outside


def face(score=0.95, nose_x=100.0):
    return {"score": score, "nose": np.array([nose_x, 120.0])}


def confidence(p=(0.05, 0.05), brightness=120, nose_x=100.0, score=0.95, prev=None):
    eyes = np.array([[70.0, 100.0], [130.0, 100.0]])
    crops = [np.full((40, 40), brightness, np.uint8)] * 2
    return input_confidence(face(score, nose_x), eyes, crops, [1.0, 1.0], np.array(p), prev)


def test_clear_open_eyes_are_trusted():
    assert confidence()[0] > 0.9


def test_each_problem_lowers_confidence():
    assert confidence(brightness=5)[0] < 0.1  # dark
    assert confidence(nose_x=60.0)[0] < 0.1  # head turned: nose far from the eye centre
    assert confidence(score=0.6)[0] < 0.1  # the detector is unsure it is a face
    assert confidence(p=(0.05, 0.95))[0] < 0.1  # the two eyes disagree
    assert confidence(p=(0.5, 0.5))[0] < 0.1  # the network is undecided
    assert confidence(prev=np.array([[20.0, 100.0], [80.0, 100.0]]))[0] < 0.1  # eyes jumped between frames
