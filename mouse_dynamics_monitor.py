import os
import sys
import time
import math
import threading
from collections import deque

import joblib
import numpy as np
import pandas as pd
from pynput import mouse
from sklearn.ensemble import IsolationForest


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(os.path.dirname(BASE_DIR))
if PROJECT_DIR not in sys.path:
    sys.path.append(PROJECT_DIR)

MODEL_FILE = os.path.join(BASE_DIR, "mouse_dynamics_model.pkl")
TRAINING_DATA_FILE = os.path.join(BASE_DIR, "mouse_dynamics_training.csv")
DETECTION_DATA_FILE = os.path.join(BASE_DIR, "mouse_detections.csv")


WINDOW_SIZE = 2000
MIN_TRAINING_WINDOWS = 1000
TRAINING_TARGET_ROWS = int(os.getenv("SENTRY_TRAINING_TARGET_ROWS", MIN_TRAINING_WINDOWS))
IDLE_CHECK_INTERVAL = 4
EVENT_WINDOW_SECONDS = 10
MIN_FEATURE_EVENTS = 5
MIN_MOVE_INTERVAL = 0.01
MIN_MOVE_DISTANCE = 1.0



FEATURE_COLUMNS = [
    "move_count",
    "click_count",
    "left_click_count",
    "right_click_count",
    "scroll_count",
    "event_rate",
    "path_length",
    "displacement",
    "straightness",
    "speed_mean",
    "speed_std",
    "speed_max",
    "acceleration_mean",
    "acceleration_std",
    "acceleration_max",
    "direction_change_mean",
    "direction_change_std",
    "click_interval_mean",
    "click_interval_std",
    "click_hold_mean",
    "click_hold_std",
    "double_click_count",
    "scroll_abs_total",
    "scroll_abs_mean",
    "scroll_interval_mean",
    "scroll_interval_std",
]

ANOMALY_INTERPRETATION = {
    
    "speed_mean": {
        "high": "Mouse movements are noticeably slower than normal",
        "low": "Mouse movements are unusually fast and erratic"
    },
    "speed_std": {
        "high": "Mouse speed is very inconsistent (jerky movements detected)",
        "low": "Mouse movements are unusually smooth/controlled"
    },
    "path_length": {
        "high": "User is taking longer paths to navigate",
        "low": "Navigation is unusually direct"
    },
    "straightness": {
        "high": "Mouse movements are unusually straight",
        "low": "Mouse movements are less direct than normal"
    },
    
    "click_interval_std": {
        "high": "Clicking pattern is very irregular (stress/fatigue indicator)",
        "low": "Clicking is unusually regular"
    },
    "click_hold_mean": {
        "high": "User is holding mouse button much longer than usual",
        "low": "User is releasing mouse button faster than usual"
    },
    "double_click_count": {
        "high": "Excessive double-clicking detected",
        "low": "Double-clicking reduced"
    },
    
    "scroll_interval_std": {
        "high": "Scrolling pattern is irregular",
        "low": "Scrolling is very regular"
    },
    
    "event_rate": {
        "high": "Mouse activity is unusually high",
        "low": "Mouse activity is reduced"
    },
}


mouse_events = deque()
pressed_buttons = {}
feature_vectors = []
feature_stats = {}
data_updated = False
baseline_ready = False
last_move_sample = None
detection_rows = 0





def prune_old_events(now):
    while mouse_events and mouse_events[0]["time"] < now - EVENT_WINDOW_SECONDS:
        mouse_events.popleft()

    while len(mouse_events) > WINDOW_SIZE:
        mouse_events.popleft()


def on_move(x, y):
    global data_updated, last_move_sample
    now = time.time()

    if last_move_sample is not None:
        last_time, last_x, last_y = last_move_sample
        dt = now - last_time
        distance = math.hypot(x - last_x, y - last_y)
        if dt < MIN_MOVE_INTERVAL and distance < MIN_MOVE_DISTANCE:
            return

    mouse_events.append({"time": now, "type": "move", "x": x, "y": y})
    last_move_sample = (now, x, y)
    prune_old_events(now)
    data_updated = True


def on_click(x, y, button, pressed):
    global data_updated
    now = time.time()
    button_name = str(button).split(".")[-1]

    mouse_events.append(
        {
            "time": now,
            "type": "click_press" if pressed else "click_release",
            "x": x,
            "y": y,
            "button": button_name,
        }
    )

    if pressed:
        pressed_buttons[button_name] = now
    else:
        pressed_buttons.pop(button_name, None)

    prune_old_events(now)
    data_updated = True


def on_scroll(x, y, dx, dy):
    global data_updated
    now = time.time()

    mouse_events.append(
        {
            "time": now,
            "type": "scroll",
            "x": x,
            "y": y,
            "dx": dx,
            "dy": dy,
        }
    )

    prune_old_events(now)
    data_updated = True


def safe_mean(values):
    return float(np.mean(values)) if values else 0.0


def safe_std(values):
    return float(np.std(values)) if values else 0.0


def extract_features_from_raw():
    events = list(mouse_events)
    if len(events) < MIN_FEATURE_EVENTS:
        return None

    features = {column: 0.0 for column in FEATURE_COLUMNS}

    move_events = [event for event in events if event["type"] == "move"]
    click_presses = [event for event in events if event["type"] == "click_press"]
    click_releases = [event for event in events if event["type"] == "click_release"]
    scroll_events = [event for event in events if event["type"] == "scroll"]

    features["move_count"] = len(move_events)
    features["click_count"] = len(click_presses)
    features["left_click_count"] = sum(1 for event in click_presses if event.get("button") == "left")
    features["right_click_count"] = sum(1 for event in click_presses if event.get("button") == "right")
    features["scroll_count"] = len(scroll_events)

    window_duration = max(events[-1]["time"] - events[0]["time"], 0.001)
    features["event_rate"] = len(events) / window_duration

    movement_steps = []
    speeds = []
    vectors = []
    times = []

    for previous, current in zip(move_events, move_events[1:]):
        dt = current["time"] - previous["time"]
        dx = current["x"] - previous["x"]
        dy = current["y"] - previous["y"]
        distance = math.hypot(dx, dy)

        if dt <= 0:
            continue

        movement_steps.append(distance)
        speeds.append(distance / dt)
        vectors.append((dx, dy))
        times.append(dt)

    if movement_steps:
        path_length = float(sum(movement_steps))
        first_move = move_events[0]
        last_move = move_events[-1]
        displacement = math.hypot(last_move["x"] - first_move["x"], last_move["y"] - first_move["y"])

        features["path_length"] = path_length
        features["displacement"] = float(displacement)
        features["straightness"] = float(displacement / path_length) if path_length else 0.0
        features["speed_mean"] = safe_mean(speeds)
        features["speed_std"] = safe_std(speeds)
        features["speed_max"] = float(max(speeds)) if speeds else 0.0

    accelerations = []
    for previous_speed, current_speed, dt in zip(speeds, speeds[1:], times[1:]):
        if dt > 0:
            accelerations.append(abs(current_speed - previous_speed) / dt)

    if accelerations:
        features["acceleration_mean"] = safe_mean(accelerations)
        features["acceleration_std"] = safe_std(accelerations)
        features["acceleration_max"] = float(max(accelerations))

    direction_changes = []
    for previous, current in zip(vectors, vectors[1:]):
        prev_length = math.hypot(previous[0], previous[1])
        curr_length = math.hypot(current[0], current[1])
        if prev_length == 0 or curr_length == 0:
            continue

        cosine = ((previous[0] * current[0]) + (previous[1] * current[1])) / (prev_length * curr_length)
        cosine = max(-1.0, min(1.0, cosine))
        direction_changes.append(abs(math.degrees(math.acos(cosine))))

    if direction_changes:
        features["direction_change_mean"] = safe_mean(direction_changes)
        features["direction_change_std"] = safe_std(direction_changes)

    click_times = [event["time"] for event in click_presses]
    if len(click_times) >= 2:
        click_intervals = np.diff(click_times)
        features["click_interval_mean"] = float(np.mean(click_intervals))
        features["click_interval_std"] = float(np.std(click_intervals))
        features["double_click_count"] = int(np.sum(click_intervals < 0.35))

    release_by_button = {}
    for event in click_releases:
        release_by_button.setdefault(event.get("button"), []).append(event)

    hold_times = []
    for press in click_presses:
        button_releases = release_by_button.get(press.get("button"), [])
        matching_release = next((event for event in button_releases if event["time"] >= press["time"]), None)
        if matching_release:
            hold_times.append(matching_release["time"] - press["time"])
            button_releases.remove(matching_release)

    if hold_times:
        features["click_hold_mean"] = safe_mean(hold_times)
        features["click_hold_std"] = safe_std(hold_times)

    scroll_times = [event["time"] for event in scroll_events]
    scroll_amounts = [abs(event.get("dx", 0)) + abs(event.get("dy", 0)) for event in scroll_events]
    if scroll_amounts:
        features["scroll_abs_total"] = float(sum(scroll_amounts))
        features["scroll_abs_mean"] = safe_mean(scroll_amounts)

    if len(scroll_times) >= 2:
        scroll_intervals = np.diff(scroll_times)
        features["scroll_interval_mean"] = float(np.mean(scroll_intervals))
        features["scroll_interval_std"] = float(np.std(scroll_intervals))

    return features


def update_feature_stats(dataframe):
    global feature_stats
    feature_stats = {}

    for column in dataframe.columns:
        feature_stats[column] = {
            "mean": dataframe[column].mean(),
            "std": dataframe[column].std(),
        }


def save_training_data():
    dataframe = pd.DataFrame(feature_vectors).reindex(columns=FEATURE_COLUMNS, fill_value=0.0)
    dataframe.to_csv(TRAINING_DATA_FILE, index=False)


def save_detection_row(feature_dict, prediction, score, reasons):
    global detection_rows
    detection_rows += 1

    row = {
        "row": detection_rows,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "prediction": "anomaly" if prediction == -1 else "normal",
        "score": score,
        "reasons": ", ".join(reasons),
    }
    row.update(feature_dict)

    pd.DataFrame([row]).to_csv(
        DETECTION_DATA_FILE,
        mode="a",
        header=not os.path.exists(DETECTION_DATA_FILE),
        index=False,
    )

    return detection_rows


def load_existing_data():
    global feature_vectors, baseline_ready, model, detection_rows

    if os.path.exists(TRAINING_DATA_FILE):
        dataframe = pd.read_csv(TRAINING_DATA_FILE)
        dataframe = dataframe.reindex(columns=FEATURE_COLUMNS, fill_value=0)
        feature_vectors = dataframe.to_dict("records")
        print(f"Loaded {len(feature_vectors)} training rows")

    if os.path.exists(DETECTION_DATA_FILE):
        detection_rows = len(pd.read_csv(DETECTION_DATA_FILE))
        

    if os.path.exists(MODEL_FILE):
        model = joblib.load(MODEL_FILE)
        baseline_ready = True
        print("Loaded existing model")

        if feature_vectors:
            update_feature_stats(pd.DataFrame(feature_vectors).reindex(columns=FEATURE_COLUMNS, fill_value=0.0))
    else:
        print(f"Will train model after {TRAINING_TARGET_ROWS} rows")
        print(f"(Capturing = {IDLE_CHECK_INTERVAL}s of mouse activity).")
        model = IsolationForest(contamination=0.05, random_state=42)


def train_baseline():
    global baseline_ready

    if len(feature_vectors) >= TRAINING_TARGET_ROWS and not baseline_ready:
        dataframe = pd.DataFrame(feature_vectors).reindex(columns=FEATURE_COLUMNS, fill_value=0.0)
        model.fit(dataframe)
        joblib.dump(model, MODEL_FILE)
        update_feature_stats(dataframe)

        baseline_ready = True
        print(f"[{time.strftime('%H:%M:%S')}] Model trained on {len(feature_vectors)} rows")
        print("Mouse anomaly detection active.")



def get_anomaly_reasons(feature_dict):
    reasons = []
    for feature_name, value in feature_dict.items():
        if feature_name not in feature_stats:
            continue
        
        mean = feature_stats[feature_name]["mean"]
        std = feature_stats[feature_name]["std"]
        
        if std > 0:
            z_score = (value - mean) / std
            if abs(z_score) > 2.0:
                direction = "high" if z_score > 0 else "low"
                if feature_name in ANOMALY_INTERPRETATION:
                    reason = ANOMALY_INTERPRETATION[feature_name].get(direction)
                    if reason:
                        reasons.append(reason)
    return reasons


def detect_anomaly(feature_dict):
    if not baseline_ready:
        return
    
    reasons = get_anomaly_reasons(feature_dict)

    dataframe = pd.DataFrame([feature_dict]).reindex(columns=FEATURE_COLUMNS, fill_value=0.0)
    prediction = model.predict(dataframe)[0]
    score = model.decision_function(dataframe)[0]
    save_detection_row(feature_dict, prediction, score, reasons)

    if prediction == -1:
        reason_text = ", ".join(reasons) if reasons else "mouse pattern drift"
        message = (
            f"[{time.strftime('%H:%M:%S')}] ANOMALY "
            f"(Score: {score:.1%}) - Reason: {reason_text}"
        )
        print(message)

    else:
        print(f"[{time.strftime('%H:%M:%S')}] NORMAL (Score: {score:.1%})")






def main_loop():
    global feature_vectors, data_updated

    while True:
        time.sleep(IDLE_CHECK_INTERVAL)

        if not data_updated:
            continue

        new_features = extract_features_from_raw()
        if new_features is not None:
            was_training_window = not baseline_ready

            if was_training_window:
                feature_vectors.append(new_features)

                if len(feature_vectors) > 5000:
                    feature_vectors = feature_vectors[-5000:]

                save_training_data()
                train_baseline()

            if baseline_ready and not was_training_window:
                detect_anomaly(new_features)

        data_updated = False


def main():
    load_existing_data()

    mouse.Listener(on_move=on_move, on_click=on_click, on_scroll=on_scroll).start()

    print("Mouse dynamics anomaly detection model active")
    threading.Thread(target=main_loop, daemon=True).start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nExit")


if __name__ == "__main__":
    main()
