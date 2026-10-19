# Pose Trainer

Browser version of the Raspberry Pi pose trainer (`originClaude.py`). Train any pose with your webcam
(martial arts, yoga, dance, therapy, sport) and get live pose recognition and joint-by-joint coaching.

Everything runs in the browser: pose detection uses TensorFlow.js MoveNet (the same 17 COCO keypoints
as the YOLOv8-pose model on the Pi), so no server, Pi or Hailo accelerator is needed. Video never
leaves the device. Trained poses are stored in the browser's local storage. Use **Poses → Export /
Import** to back them up or move them to another device.

## Files

- `index.html`, `style.css`, `app.js`: the app
- `model/`: MoveNet SinglePose Lightning weights, hosted with the site
- `poses.json`: default pose library, built from the Pi's `training_data/`

## Run locally

Camera access needs `https://` or `localhost`:

```
python -m http.server 8000
```

Then open http://localhost:8000/.
