import cv2
import cvzone
import torch
import time
from ultralytics import YOLO

# Print GPU info
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA device count: {torch.cuda.device_count()}")
print(torch.cuda.get_device_name(0))

# Load and optimize YOLO model
try:
    model = YOLO('../Yolo-Weights/yolov8l.pt').cuda()
    model.fuse()
    model.eval()
except Exception as e:
    print(f"Failed to load YOLO model: {e}")
    exit(1)

# Vehicle classes and camera setup
vehicle_classes = {"car", "truck", "bus", "motorbike","ambulance"}
camera_sources = {
    "North": '../Yolo/videos/tra2.mp4',
    "South": '../Yolo/videos/tra2.mp4',
    "East": '../Yolo/videos/ambulance.mp4',
    "West": '../Yolo/videos/ambulance.mp4'
}

# Initialize cameras with optimized settings
caps = {}
for side, src in camera_sources.items():
    cap = cv2.VideoCapture(src)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 640)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)
    if cap.isOpened():
        caps[side] = cap
    else:
        print(f"Failed to open {side} camera")


def count_vehicles_and_draw_boxes(frame):
    frame = cv2.resize(frame, (640, 640))
    tensor = torch.from_numpy(frame).float().cuda()
    tensor = tensor.permute(2, 0, 1) / 255.0

    with torch.amp.autocast(device_type='cuda'), torch.no_grad():
        results = model(tensor[None])[0]

    count = 0
    boxes = results.boxes.xyxy.cpu().numpy()
    classes = results.boxes.cls.cpu().numpy()

    for box, cls in zip(boxes, classes):
        if model.names[int(cls)] in vehicle_classes:
            count += 1
            x1, y1, x2, y2 = map(int, box)
            cvzone.cornerRect(frame, (x1, y1, x2 - x1, y2 - y1), l=9)
            cvzone.putTextRect(frame, f"{model.names[int(cls)]}",
                               (x1, y1 - 10), scale=0.7, thickness=1)

    # Add vehicle count display
    cv2.putText(frame, f"Current Vehicles: {count}", (50, 150),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    return count, frame


def calculate_signal_time(vehicle_count):
    return max(15, min(120, vehicle_count * 5))


# Traffic control logic
traffic_order = ["North", "East", "South", "West"]
current_index = 0
signal_remaining = 0
next_side_ready = False
next_vehicle_count = 0
next_side = traffic_order[1]  # Initial next side

try:
    last_time = time.time()
    while True:
        current_time = time.time()
        elapsed = current_time - last_time
        last_time = current_time

        current_side = traffic_order[current_index]
        cap = caps.get(current_side)

        if cap and (success := cap.grab()):
            _, frame = cap.retrieve()
            if frame is not None:
                vehicle_count, frame = count_vehicles_and_draw_boxes(frame)
                green_time = calculate_signal_time(vehicle_count)

                # Update next signal info
                next_index = (current_index + 1) % 4
                next_side = traffic_order[next_index]
                if signal_remaining <= 5 and not next_side_ready:
                    if (cap_next := caps.get(next_side)) and cap_next.grab():
                        _, frame_next = cap_next.retrieve()
                        next_vehicle_count, _ = count_vehicles_and_draw_boxes(frame_next)
                        next_side_ready = True

                # Display all information
                cv2.putText(frame, f"Green Signal: {current_side} ({max(0, signal_remaining):.1f}s)",
                            (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(frame, f"Next: {next_side} ({next_vehicle_count} vehicles)",
                            (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

                # Create window before displaying
                cv2.namedWindow(f"Traffic - {current_side}", cv2.WINDOW_NORMAL)
                cv2.imshow(f"Traffic - {current_side}", frame)

        signal_remaining -= elapsed
        if signal_remaining <= 0:
            current_index = next_index
            signal_remaining = calculate_signal_time(next_vehicle_count)
            next_side_ready = False

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    for cap in caps.values():
        cap.release()
    cv2.destroyAllWindows()