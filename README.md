# ML Human Activity Recognition 

Models for recognizing human activities.

---
## Overview

This project contains the implementation of a **GRU recurrent neural network** designed for **predicting recorded human activity** based on motion capture using a ZED camera, which provides a 34-point human body model.

**Key Features:**
- Uses GRU recurrent neural architectures for temporal sequence modeling.
- Input data: 34-point body joint positions captured frame by frame (read more about it [here](https://www.stereolabs.com/docs/body-tracking)).
- Output: predicted human activity classes or movement patterns.
- Testing dataset including 6 activity classes.
---

## Usage

1. **Clone the repository:**
```bash
git clone https://github.com/janoleskiuk/GRU-Pose-Classifier.git
cd GRU-Pose-Classifier
```

2. **Install required packages:**
```bash
python -m pip install -r requirements.txt
```
**Note**: Skip `pyzed` dependencies if you dont have access to ZED camera.

3. **Run**:

If you have ZED camera connected and set up run:
```bash
python predict.py
```

4. **Create and train on your custom data**:

After uploading your own raw data to `data/raw` update `classes.py` file and process it using `data_process.py` and `data_split.py`. Develop the model using delivered notebook.

---



