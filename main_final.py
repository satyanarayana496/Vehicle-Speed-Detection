import cv2
import math
import os
import datetime
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import style
from ultralytics import YOLO

plt.rcParams.update({'font.size': 10})

# =====================================
# SETTINGS
# =====================================
SPEED_LIMIT_KMPH = 60
DISTANCE_METERS = 10   # Real road distance between two speed lines

# =====================================
# CREATE OUTPUT FOLDERS
# =====================================
base_dir = "TrafficRecord"
dataset_dir = os.path.join(base_dir, "dataset")

folders = [
    base_dir,
    dataset_dir,
    os.path.join(dataset_dir, "overspeed"),
    os.path.join(dataset_dir, "no_helmet"),
    os.path.join(dataset_dir, "multiple_violations"),
    os.path.join(dataset_dir, "all_violations")
]

for folder in folders:
    os.makedirs(folder, exist_ok=True)

# =====================================
# INIT REPORT FILE
# =====================================
report_path = os.path.join(base_dir, "SpeedRecord.txt")
with open(report_path, "w") as f:
    f.write("====================================================================================================\n")
    f.write("                               SMART TRAFFIC MONITORING REPORT                                      \n")
    f.write("====================================================================================================\n")
    f.write("ID\tTYPE\tSPEED(km/h)\tOVERSPEED\tHELMET\tVIOLATION\tIMAGE\n")
    f.write("====================================================================================================\n")


# =====================================
# TRACKER CLASS
# =====================================
class EuclideanDistTracker:
    def __init__(self, fps):
        self.center_points = {}
        self.id_count = 0
        self.fps = fps

        self.start_frame = np.zeros(5000, dtype=np.int32)
        self.end_frame = np.zeros(5000, dtype=np.int32)
        self.speed_kmph = np.zeros(5000, dtype=np.float32)

        self.capf = np.zeros(5000)
        self.finished = np.zeros(5000)

        self.count = 0
        self.exceeded = 0
        self.no_helmet_count = 0
        self.multi_violation_count = 0

        self.ids_DATA = []
        self.spd_DATA = []
        self.records = []

    def update(self, objects_rect, start_y1, start_y2, end_y1, end_y2, frame_count):
        objects_bbs_ids = []

        for rect in objects_rect:
            x, y, w, h, vehicle_type = rect
            cx = (x + x + w) // 2
            cy = (y + y + h) // 2

            same_object_detected = False

            for object_id, pt in self.center_points.items():
                dist = math.hypot(cx - pt[0], cy - pt[1])

                if dist < 70:
                    prev_cx, prev_cy = pt
                    self.center_points[object_id] = (cx, cy)
                    objects_bbs_ids.append([x, y, w, h, object_id, vehicle_type])
                    same_object_detected = True

                    # Vehicle moving from bottom to top
                    # Cross END line first (lower line)
                    if self.end_frame[object_id] == 0:
                        if prev_cy > end_y2 and cy <= end_y2:
                            self.end_frame[object_id] = frame_count
                            print(f"[END FIRST] ID {object_id} at frame {frame_count}")

                    # Cross START line second (upper line)
                    if self.end_frame[object_id] != 0 and self.start_frame[object_id] == 0:
                        if prev_cy > start_y2 and cy <= start_y2:
                            self.start_frame[object_id] = frame_count

                            frame_diff = abs(self.start_frame[object_id] - self.end_frame[object_id])

                            if frame_diff > 0:
                                time_sec = frame_diff / self.fps
                                speed = (DISTANCE_METERS / time_sec) * 3.6
                                self.speed_kmph[object_id] = speed
                                self.finished[object_id] = 1
                                print(f"[SPEED DONE] ID {object_id} | Time: {time_sec:.2f}s | Speed: {speed:.2f} km/h")

                    break

            if not same_object_detected:
                new_id = self.id_count
                self.center_points[new_id] = (cx, cy)
                objects_bbs_ids.append([x, y, w, h, new_id, vehicle_type])

                self.start_frame[new_id] = 0
                self.end_frame[new_id] = 0
                self.speed_kmph[new_id] = 0
                self.finished[new_id] = 0

                self.id_count += 1

        new_center_points = {}
        for obj in objects_bbs_ids:
            _, _, _, _, object_id, _ = obj
            center = self.center_points[object_id]
            new_center_points[object_id] = center

        self.center_points = new_center_points.copy()
        return objects_bbs_ids

    def getsp(self, object_id):
        if self.speed_kmph[object_id] > 0:
            return int(self.speed_kmph[object_id])
        return 0

    def is_done(self, object_id):
        return self.finished[object_id] == 1

    def capture(self, img, x, y, h, w, sp, object_id,
                no_helmet=False,
                vehicle_type="vehicle",
                helmet_status="N/A"):

        if self.capf[object_id] == 1:
            return

        overspeed = sp > SPEED_LIMIT_KMPH
        helmet_violation = (vehicle_type == "bike" and no_helmet)

        # Save ONLY violations
        if not overspeed and not helmet_violation:
            return

        self.capf[object_id] = 1

        x1 = max(0, int(x - 15))
        y1 = max(0, int(y - 15))
        x2 = min(img.shape[1], int(x + w + 15))
        y2 = min(img.shape[0], int(y + h + 15))

        if x2 <= x1 or y2 <= y1:
            return

        crop_img = img[y1:y2, x1:x2]

        if crop_img is None or crop_img.size == 0:
            return

        crop_img = cv2.resize(crop_img, (240, 240))

        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        violation_tags = []
        if overspeed:
            violation_tags.append("overspeed")
        if helmet_violation:
            violation_tags.append("nohelmet")

        if len(violation_tags) == 1:
            violation_name = violation_tags[0]
            if violation_name == "overspeed":
                save_folder = os.path.join(dataset_dir, "overspeed")
            else:
                save_folder = os.path.join(dataset_dir, "no_helmet")
        else:
            violation_name = "_".join(violation_tags)
            save_folder = os.path.join(dataset_dir, "multiple_violations")

        filename = f"id_{object_id}_{vehicle_type}_{sp}kmph_{violation_name}_{timestamp}.jpg"
        filepath = os.path.join(save_folder, filename)

        cv2.imwrite(filepath, crop_img)
        cv2.imwrite(os.path.join(dataset_dir, "all_violations", filename), crop_img)

        self.count += 1

        if overspeed:
            self.exceeded += 1
        if helmet_violation:
            self.no_helmet_count += 1
        if len(violation_tags) >= 2:
            self.multi_violation_count += 1

        with open(report_path, "a") as f:
            f.write(
                f"{object_id}\t{vehicle_type}\t{sp}\t"
                f"{'YES' if overspeed else 'NO'}\t"
                f"{helmet_status}\t"
                f"{violation_name}\t"
                f"{filename}\n"
            )

        self.records.append({
            "ID": object_id,
            "Vehicle_Type": vehicle_type,
            "Speed_kmph": sp,
            "Overspeed": "YES" if overspeed else "NO",
            "Helmet_Status": helmet_status,
            "Violation_Type": violation_name,
            "Image_Name": filename,
            "Timestamp": timestamp
        })

        self.ids_DATA.append(object_id)
        self.spd_DATA.append(sp)

        print(f"[SAVED] {filename}")

    def dataset(self):
        return self.ids_DATA, self.spd_DATA

    def save_csv(self):
        if len(self.records) > 0:
            df = pd.DataFrame(self.records)
            df.to_csv(os.path.join(base_dir, "traffic_data.csv"), index=False)

    def datavis(self, id_lst, spd_lst):
        if len(id_lst) == 0 or len(spd_lst) == 0:
            print("No data available for graph.")
            return

        x = id_lst
        y = spd_lst
        valx = [str(i) for i in x]

        plt.figure(figsize=(20, 5))
        style.use('dark_background')
        plt.axhline(y=SPEED_LIMIT_KMPH, color='r', linestyle='-', linewidth=3)
        plt.bar(x, y, width=0.5, linewidth=2, edgecolor='yellow', color='blue', align='center')
        plt.xlabel('Vehicle ID')
        plt.ylabel('Speed (km/h)')
        plt.xticks(x, valx) # Set x-ticks to be the IDs
        plt.legend(["Speed Limit"]) # Adds a small box explaining that the Red Line represents the Speed Limit.
        plt.title('SPEED OF VIOLATION VEHICLES\n')
        plt.savefig(os.path.join(base_dir, "datavis.png"), bbox_inches='tight', pad_inches=1)
        plt.close()

    def limit(self):
        return SPEED_LIMIT_KMPH

    def end(self):
        self.save_csv()

        with open(report_path, "a") as f:
            f.write("\n====================================================================================================\n")
            f.write("                                               SUMMARY                                              \n")
            f.write("====================================================================================================\n")
            f.write(f"Distance Between Lines (meters) : {DISTANCE_METERS}\n")
            f.write(f"Speed Limit (km/h)              : {SPEED_LIMIT_KMPH}\n")
            f.write(f"Total Violation Vehicles Saved  : {self.count}\n")
            f.write(f"Exceeded Speed Limit            : {self.exceeded}\n")
            f.write(f"No Helmet Violations (Bikes)    : {self.no_helmet_count}\n")
            f.write(f"Multiple Violations             : {self.multi_violation_count}\n")
            f.write("CSV Report Saved                : TrafficRecord/traffic_data.csv\n")
            f.write("====================================================================================================\n")
            f.write("                                                 END                                                \n")
            f.write("====================================================================================================\n")

        print("[INFO] Reports saved in TrafficRecord/")


# =====================================
# LOAD MODELS
# =====================================
vehicle_model = YOLO("yolov8n.pt")
print("Vehicle model loaded successfully!")

try:
    helmet_model = YOLO("helmet_best.pt")
    helmet_model_loaded = True
    print("Helmet model loaded successfully!")
except Exception as e:
    helmet_model_loaded = False
    print("Helmet model not loaded.")
    print("Reason:", e)
    print("Project will continue WITHOUT helmet detection.")

# =====================================
# VIDEO INPUT
# =====================================
cap = cv2.VideoCapture("resources/traffic.mp4")

if not cap.isOpened():
    print("[ERROR] Could not open video file.")
    exit()

fps = cap.get(cv2.CAP_PROP_FPS)
if fps == 0:
    fps = 30

print(f"[INFO] FPS detected: {fps}")

tracker = EuclideanDistTracker(fps)

total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
frame_count = 0
end_flag = 0

# =====================================
# VEHICLE CLASSES
# =====================================
vehicle_classes = {
    2: "car",
    3: "bike",
    5: "bus",
    7: "truck"
}

# =====================================
# ROI SETTINGS
# =====================================
ROI_Y1, ROI_Y2 = 100, 650
ROI_X1, ROI_X2 = 100, 1180

# =====================================
# SPEED LINES
# Vehicles move BOTTOM -> TOP
# lower line crossed first
# =====================================
start_line_y1 = 180
start_line_y2 = 200

end_line_y1 = 380
end_line_y2 = 400


# =====================================
# HELMET DETECTION
# =====================================
def detect_helmet_yolo(roi, x, y, w, h):
    if not helmet_model_loaded:
        return "UNKNOWN", False

    x1 = max(0, x)
    y1 = max(0, y)
    x2 = min(roi.shape[1], x + w)
    y2 = min(roi.shape[0], y + h)

    bike_crop = roi[y1:y2, x1:x2]

    if bike_crop is None or bike_crop.size == 0:
        return "UNKNOWN", False

    try:
        results = helmet_model(bike_crop, verbose=False)

        best_conf = 0
        best_label = None

        for r in results:
            for box in r.boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])

                if conf > best_conf:
                    best_conf = conf
                    best_label = cls

        # 0 = helmet, 1 = no_helmet
        if best_label == 0 and best_conf > 0.40:
            return "YES", False
        elif best_label == 1 and best_conf > 0.40:
            return "NO", True
        else:
            return "UNKNOWN", False

    except Exception as e:
        print("[HELMET ERROR]", e)
        return "UNKNOWN", False


# =====================================
# MAIN LOOP
# =====================================
while True:
    ret, frame = cap.read()
    if not ret:
        print("[INFO] Video ended.")
        break

    frame_count += 1

    frame = cv2.resize(frame, (1280, 720))
    roi = frame[ROI_Y1:ROI_Y2, ROI_X1:ROI_X2]
    roi_h, roi_w = roi.shape[:2]

    detections = []

    # VEHICLE DETECTION
    results = vehicle_model(roi, stream=True)

    for r in results:
        for box in r.boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])

            if cls in vehicle_classes and conf > 0.35:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                w = x2 - x1
                h = y2 - y1
                vehicle_type = vehicle_classes[cls]

                if w > 30 and h > 30:
                    detections.append([x1, y1, w, h, vehicle_type])

    # TRACKING
    boxes_ids = tracker.update(
        detections,
        start_line_y1, start_line_y2,
        end_line_y1, end_line_y2,
        frame_count
    )

    for box_id in boxes_ids:
        x, y, w, h, object_id, vehicle_type = box_id
        speed = tracker.getsp(object_id)

        helmet_status = "N/A"
        no_helmet = False

        if vehicle_type == "bike":
            helmet_status, no_helmet = detect_helmet_yolo(roi, x, y, w, h)

        is_overspeed = speed > tracker.limit()
        is_no_helmet = (vehicle_type == "bike" and helmet_status == "NO")

        info = []

        if speed > 0:
            info.append(f"{speed} km/h")

        if is_overspeed:
            info.append("OVERSPEED")

        if vehicle_type == "bike":
            if helmet_status == "YES":
                info.append("HELMET")
            elif helmet_status == "NO":
                info.append("NO HELMET")
            else:
                info.append("HELMET ?")
        else:
            info.append("HELMET: N/A")

        if len(info) == 0:
            label = f"ID:{object_id} {vehicle_type}"
            color = (0, 255, 0)
        else:
            label = f"ID:{object_id} {vehicle_type} | {' | '.join(info)}"
            color = (0, 0, 255) if (is_overspeed or is_no_helmet) else (0, 255, 0)

        cv2.rectangle(roi, (x, y), (x + w, y + h), color, 2)
        cv2.putText(roi, label, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

        # Save after speed calculated
        full_x = x + ROI_X1
        full_y = y + ROI_Y1

        if tracker.is_done(object_id) and tracker.capf[object_id] == 0:
            tracker.capture(
                frame,
                full_x,
                full_y,
                h,
                w,
                speed,
                object_id,
                no_helmet=no_helmet,
                vehicle_type=vehicle_type,
                helmet_status=helmet_status
            )

    # DRAW ROI
    cv2.rectangle(frame, (ROI_X1, ROI_Y1), (ROI_X2, ROI_Y2), (255, 255, 255), 2)

    # START line (upper)
    cv2.line(roi, (0, start_line_y1), (roi_w, start_line_y1), (255, 0, 0), 2)
    cv2.line(roi, (0, start_line_y2), (roi_w, start_line_y2), (255, 0, 0), 2)
    cv2.putText(roi, "START", (10, start_line_y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    # END line (lower)
    cv2.line(roi, (0, end_line_y1), (roi_w, end_line_y1), (255, 0, 0), 2)
    cv2.line(roi, (0, end_line_y2), (roi_w, end_line_y2), (255, 0, 0), 2)
    cv2.putText(roi, "END", (10, end_line_y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    # TOP BAR
    cv2.rectangle(frame, (0, 0), (1280, 40), (40, 40, 40), -1)

    d = datetime.datetime.now().strftime("%d-%m-%Y")
    t = datetime.datetime.now().strftime("%H:%M:%S")

    cv2.putText(frame, f"DATE: {d}", (20, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(frame, f"TIME: {t}", (220, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(frame, f"FPS: {fps:.2f}", (420, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(frame, f"FRAME: {frame_count}/{total_frames}", (560, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(frame, f"LIMIT: {tracker.limit()} km/h", (900, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    cv2.imshow("Traffic Monitoring System", frame)
    cv2.imshow("ROI Detection", roi)

    key = cv2.waitKey(1)

    if key == 13:  # ENTER
        print("[INFO] ENTER pressed. Stopping...")
        tracker.end()
        ids_lst, spd_lst = tracker.dataset()
        tracker.datavis(ids_lst, spd_lst)
        end_flag = 1
        break

# =====================================
# EXIT
# =====================================
if end_flag != 1:
    tracker.end()
    ids_lst, spd_lst = tracker.dataset()
    tracker.datavis(ids_lst, spd_lst)

cap.release()
cv2.destroyAllWindows()