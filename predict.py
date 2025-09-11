import pyzed.sl as sl
import cv2
import numpy as np
import torch
from models.models import PoseGRU  
from data.classes import CLASS_MAPPING

# -----------------------------
# MODEL CONFIG
# -----------------------------

INPUT_DIM = 34 * 3  
SEQ_LEN = 15        
HIDDEN_DIM = 128
NUM_LAYERS = 2
NUM_CLASSES = 11
MODEL_PATH = "trained_models/pose_gru_11class.pt"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load trained model
model = PoseGRU(input_dim=INPUT_DIM, hidden_dim=HIDDEN_DIM, num_layers=NUM_LAYERS, num_classes=len(CLASS_MAPPING))
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()

# -----------------------------
# PREDICTOR
# -----------------------------

class RealtimePredictor:
    def __init__(self, model, SEQ_LEN):
        self.model = model
        self.seq_len = SEQ_LEN
        self.buffer = []

    def add_frame(self, keypoints):
        flat = keypoints.reshape(-1)  
        self.buffer.append(flat)
        if len(self.buffer) > self.seq_len:
            self.buffer.pop(0)

    def predict(self):
        if len(self.buffer) < self.seq_len:
            return None 
        seq = np.array(self.buffer, dtype=np.float32)  # [seq_len, input_dim]
        seq_tensor = torch.tensor(seq).unsqueeze(0).to(device)  # [1, seq_len, input_dim]
        with torch.no_grad():
            logits = self.model(seq_tensor)
            probs = torch.softmax(logits, dim=1)
            pred_class = torch.argmax(probs, dim=1).item()
        return pred_class, probs.cpu().numpy()

# -----------------------------
# ZED CAM CONFIG
# -----------------------------
BODY_IDX = 34
FRAMES = 2000

zed = sl.Camera()
init_params = sl.InitParameters()
init_params.camera_resolution = sl.RESOLUTION.HD720
init_params.depth_mode = sl.DEPTH_MODE.NEURAL
init_params.coordinate_units = sl.UNIT.METER
init_params.sdk_verbose = 1

if zed.open(init_params) != sl.ERROR_CODE.SUCCESS:
    print("Camera open failed")
    exit()

body_params = sl.BodyTrackingParameters()
body_params.detection_model = sl.BODY_TRACKING_MODEL.HUMAN_BODY_FAST
body_params.enable_tracking = True
body_params.enable_body_fitting = True
body_params.body_format = sl.BODY_FORMAT.BODY_34

if body_params.enable_tracking:
    positional_tracking_param = sl.PositionalTrackingParameters()
    positional_tracking_param.set_floor_as_origin = True
    zed.enable_positional_tracking(positional_tracking_param)

if zed.enable_body_tracking(body_params) != sl.ERROR_CODE.SUCCESS:
    print("Body tracking failed")
    zed.close()
    exit()

image = sl.Mat()
bodies = sl.Bodies()
body_runtime_param = sl.BodyTrackingRuntimeParameters()
body_runtime_param.detection_confidence_threshold = 40

cv2.namedWindow("ZED Body Tracking", cv2.WINDOW_NORMAL)
camera_info = zed.get_camera_information()
cv2.resizeWindow("ZED Body Tracking", camera_info.camera_configuration.resolution.width,
                                         camera_info.camera_configuration.resolution.height)

# -----------------------------
# PREDICTOR
# -----------------------------
predictor = RealtimePredictor(model, SEQ_LEN)

i = 0
while i < FRAMES:
    if zed.grab() == sl.ERROR_CODE.SUCCESS:
        zed.retrieve_image(image, sl.VIEW.LEFT)
        zed.retrieve_bodies(bodies, body_runtime_param)
        img_cv = image.get_data()

        if bodies.is_new and bodies.body_list:
            body = bodies.body_list[0]  
            keypoints_3d = np.array(body.keypoint)  # [34,3]
            predictor.add_frame(keypoints_3d)

            result = predictor.predict()
            if result is not None:
                pred_class, probs = result
                cv2.putText(img_cv, f"Predicted class: {pred_class}", (30,30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
        
        cv2.imshow("ZED Body Tracking", img_cv)
        key = cv2.waitKey(10)
        if key == 27:  # ESC
            break
        i += 1

zed.disable_body_tracking()
zed.close()
cv2.destroyAllWindows()

