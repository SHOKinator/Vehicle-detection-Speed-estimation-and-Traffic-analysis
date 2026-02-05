import cv2
from deepsort_tracker import Tracker
from yolo_model import model
import random
import matplotlib.pyplot as plt
import numpy as np
import io
from PIL import Image

video_path = 'assets/enhanced_traffic.mp4'

cap = cv2.VideoCapture(video_path)
ret, frame = cap.read()
if not ret:
    print("Cannot read video")
    exit()

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

out = cv2.VideoWriter('result/output3.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

tracker = Tracker()
colors = [(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for _ in range(100)]

start_line = int(frame_height * 0.28)
d10 = int(frame_height * 0.31)
d20 = int(frame_height * 0.355)
d30 = int(frame_height * 0.43)
end_line = int(frame_height * 0.58)

track_times = {}
track_speeds = {}
distance_split = 10
distance_full = 45
frame_count = 0

class_names = {
    2: 'car',
    3: 'motorcycle',
    5: 'bus',
    7: 'truck'
}

vehicle_counts = {
    'car': 0,
    'motorcycle': 0,
    'bus': 0,
    'truck': 0
}

track_classes = {}

def diagram(counts):
    if sum(counts.values()) == 0:
        return np.zeros((200, 200, 3), dtype = np.uint8)
    labels = list(counts.keys())
    values = list(counts.values())

    fig, ax = plt.subplots(figsize=(3, 3), dpi = 100)
    ax.pie(values, labels=labels, autopct='%1.1f%%', startangle=90)
    ax.axis('equal')

    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    plt.close(fig)
    buf.seek(0)

    img_pil = Image.open(buf).convert('RGB')
    img_pil = img_pil.resize((200, 200))
    img_np = np.array(img_pil)
    img_cv2 = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

    return img_cv2

while ret:
    results = model(frame, classes=[2, 3, 5, 7])

    for result in results:
        detections = []
        for r in result.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = r
            if score > 0.5:
                class_id = int(class_id)
                class_name = class_names.get(class_id)
            if class_name:
                detections.append([int(x1), int(y1), int(x2), int(y2), score, class_name])
            else:
                continue

        tracker_input = [det[:5] for det in detections]  
        tracker.update(frame, tracker_input)

        for track in tracker.tracks:
            bbox = track.bbox
            x1, y1, x2, y2 = map(int, bbox)
            track_id = track.track_id
            for det in detections:
                dx1, dy1, dx2, dy2, _, class_name = det
                if abs(x1 - dx1) < 15 and abs(y1 - dy1) < 15:
                    track_classes[track_id] = class_name
                    break

            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2

            if track_id not in track_times:
                track_times[track_id] = {
                    'start_frame': None, 
                    'd10_frame': None,
                    'd20_frame': None,
                    'd30_frame': None,
                    'end_frame': None}

                    
            if start_line < center_y < start_line + 30:        
                if track_times[track_id]['start_frame'] is None:
                    track_times[track_id]['start_frame'] = frame_count
                    

            if d10 < center_y < d10 + 30:
                if track_times[track_id]['start_frame'] is not None and track_times[track_id]['d10_frame'] is None:
                    track_times[track_id]['d10_frame'] = frame_count 
                    time_diff = (frame_count - track_times[track_id]['start_frame']) / fps
                    speed_10 = ((distance_split) / time_diff) * 3.6 
                    track_speeds[f"{track_id}_10m"] = round(speed_10, 2)
            
            if d20 < center_y < d20 + 30:
                if track_times[track_id]['start_frame'] is not None and track_times[track_id]['d20_frame'] is None:
                    track_times[track_id]['d20_frame'] = frame_count
                    start_f = track_times[track_id]['start_frame']
                    end_f = track_times[track_id]['d20_frame']  
                    if end_f > start_f:
                        time_diff = (end_f - start_f) / fps
                        speed_20 = ((distance_split * 2) / time_diff) * 3.6 
                        track_speeds[f"{track_id}_20m"] = round(speed_20, 2)
                        track_speeds.pop(f"{track_id}_10m", None)
            
            if d30 < center_y < d30 + 30:
                if track_times[track_id]['start_frame'] is not None and track_times[track_id]['d30_frame'] is None:
                    track_times[track_id]['d30_frame'] = frame_count
                    start_f = track_times[track_id]['start_frame']
                    end_f = track_times[track_id]['d30_frame']  
                    if end_f > start_f:
                        time_diff = (end_f - start_f) / fps
                        speed_20 = ((distance_split * 3) / time_diff) * 3.6 
                        track_speeds[f"{track_id}_30m"] = round(speed_20, 2)
                        track_speeds.pop(f"{track_id}_20m", None)

            if end_line < center_y < end_line + 30:
                if track_times[track_id]['start_frame'] is not None and track_times[track_id]['end_frame'] is None:
                    track_times[track_id]['end_frame'] = frame_count
                    start_f = track_times[track_id]['start_frame']
                    end_f = track_times[track_id]['end_frame']
                    if end_f > start_f:
                        time_diff = (end_f - start_f) / fps
                        speed_45 = (distance_full / time_diff) * 3.6 
                        track_speeds[f'{track_id}_45m'] = round(speed_45, 2)
                        track_speeds.pop(f"{track_id}_30m", None)
                    class_name = track_classes.get(track_id)
                    if class_name:
                        vehicle_counts[class_name] += 1

            label = f"ID: {track_id}"
            if f"{track_id}_10m" in track_speeds:
                label += f" ~{track_speeds[f'{track_id}_10m']} km/h" 
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, colors[track_id % len(colors)], 1)
            if f"{track_id}_20m" in track_speeds:
                label += f" ~{track_speeds[f'{track_id}_20m']} km/h" 
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, colors[track_id % len(colors)], 1)
            if f"{track_id}_30m" in track_speeds:
                label += f" ~{track_speeds[f'{track_id}_30m']} km/h" 
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, colors[track_id % len(colors)], 1)
            if f"{track_id}_45m" in track_speeds:
                label += f" ~{track_speeds[f'{track_id}_45m']} km/h" 

            cv2.rectangle(frame, (x1, y1), (x2, y2), colors[track_id % len(colors)], 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, colors[track_id % len(colors)], 1)

    cv2.line(frame, (0, start_line), (frame_width, start_line), (0, 255, 0), 2)
    cv2.line(frame, (0, d10), (frame_width, d10), (0, 255, 0), 2)
    cv2.line(frame, (0, d20), (frame_width, d20), (0, 255, 0), 2)
    cv2.line(frame, (0, d30), (frame_width, d30), (0, 255, 0), 2)
    cv2.line(frame, (0, end_line), (frame_width, end_line), (0, 255, 0), 2)

    print("Vehicle counts:", vehicle_counts)
    chart = diagram(vehicle_counts)
    h_chart, w_chart, _ = chart.shape

    total_counts = sum(vehicle_counts.values())
    text = f"Total vehicles: {total_counts}"
    cv2.putText(frame, text, (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    if frame.shape[0] >= h_chart and frame.shape[1] >= w_chart:
        frame[0:h_chart, -w_chart:] = chart

    out.write(frame)

    ret, frame = cap.read()
    frame_count += 1

cap.release()
out.release()
