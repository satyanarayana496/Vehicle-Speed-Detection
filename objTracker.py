import cv2
import math
import os
import datetime
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import style

plt.rcParams.update({'font.size': 10})

# =====================================
# SETTINGS
# =====================================
SPEED_LIMIT_KMPH = 60
DISTANCE_METERS = 10   # Real distance between lines

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

        self.capf = np.zeros(5000, dtype=np.int32)
        self.finished = np.zeros(5000, dtype=np.int32)

        self.count = 0
        self.exceeded = 0
        self.no_helmet_count = 0
        self.multi_violation_count = 0

        self.ids_DATA = []
        self.spd_DATA = []
        self.records = []

    # =====================================
    # TRACK UPDATE
    # =====================================
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

                    # =====================================
                    # TRAFFIC DIRECTION:
                    # Vehicle moves from BOTTOM -> TOP
                    # so lower line crossed first
                    # =====================================

                    # CROSS LOWER LINE FIRST (END line)
                    if self.end_frame[object_id] == 0:
                        if prev_cy > end_y2 and cy <= end_y2:
                            self.end_frame[object_id] = frame_count
                            print(f"[END FIRST] ID {object_id} at frame {frame_count}")

                    # CROSS UPPER LINE SECOND (START line)
                    if self.end_frame[object_id] != 0 and self.start_frame[object_id] == 0:
                        if prev_cy > start_y2 and cy <= start_y2:
                            self.start_frame[object_id] = frame_count

                            frame_diff = abs(self.start_frame[object_id] - self.end_frame[object_id])

                            if frame_diff > 0:
                                time_sec = frame_diff / self.fps
                                speed = (DISTANCE_METERS / time_sec) * 3.6
                                self.speed_kmph[object_id] = speed
                                self.finished[object_id] = 1

                                print(
                                    f"[SPEED DONE] ID {object_id} | "
                                    f"Time: {time_sec:.2f}s | "
                                    f"Speed: {speed:.2f} km/h"
                                )

                    break

            if not same_object_detected:
                new_id = self.id_count
                self.center_points[new_id] = (cx, cy)
                objects_bbs_ids.append([x, y, w, h, new_id, vehicle_type])

                self.start_frame[new_id] = 0
                self.end_frame[new_id] = 0
                self.speed_kmph[new_id] = 0
                self.finished[new_id] = 0
                self.capf[new_id] = 0

                self.id_count += 1

        # Keep only active tracked objects
        new_center_points = {}
        for obj in objects_bbs_ids:
            _, _, _, _, object_id, _ = obj
            center = self.center_points[object_id]
            new_center_points[object_id] = center

        self.center_points = new_center_points.copy()
        return objects_bbs_ids

    # =====================================
    # GET SPEED
    # =====================================
    def getsp(self, object_id):
        if self.speed_kmph[object_id] > 0:
            return int(self.speed_kmph[object_id])
        return 0

    # =====================================
    # SPEED FINISHED?
    # =====================================
    def is_done(self, object_id):
        return self.finished[object_id] == 1

    # =====================================
    # SAVE ONLY VIOLATION VEHICLES
    # =====================================
    def capture(self, img, x, y, h, w, sp, object_id,
                no_helmet=False,
                vehicle_type="vehicle",
                helmet_status="N/A"):

        if self.capf[object_id] == 1:
            return

        overspeed = sp > SPEED_LIMIT_KMPH
        helmet_violation = (vehicle_type == "bike" and no_helmet)

        # Save ONLY if violation exists
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

    # =====================================
    # DATA FOR GRAPH
    # =====================================
    def dataset(self):
        return self.ids_DATA, self.spd_DATA

    # =====================================
    # SAVE CSV
    # =====================================
    def save_csv(self):
        if len(self.records) > 0:
            df = pd.DataFrame(self.records)
            df.to_csv(os.path.join(base_dir, "traffic_data.csv"), index=False)

    # =====================================
    # GRAPH
    # =====================================
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
        plt.xticks(x, valx)
        plt.legend(["Speed Limit"])
        plt.title('SPEED OF VIOLATION VEHICLES\n')
        plt.savefig(os.path.join(base_dir, "datavis.png"), bbox_inches='tight', pad_inches=1)
        plt.close()

    # =====================================
    # SPEED LIMIT
    # =====================================
    def limit(self):
        return SPEED_LIMIT_KMPH

    # =====================================
    # END REPORT
    # =====================================
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