import json
from datetime import datetime
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = PROJECT_DIR / "output"
DATA_FILE = PROJECT_DIR / "motion_data.json"
GRAPH_FILE = OUTPUT_DIR / "motion_statistics.png"
MIN_MOTION_AREA = 500

DATA_COLUMNS = ["date", "time", "motion_status", "motion_count"]


def create_project_files():
    """Create the output folder and an empty JSON file when needed."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if not DATA_FILE.exists():
        DATA_FILE.write_text("[]", encoding="utf-8")


def save_motion_data(records):
    """Write the current run's motion records to JSON."""
    try:
        with DATA_FILE.open("w", encoding="utf-8") as json_file:
            json.dump(records, json_file, indent=4)
        return True
    except (OSError, TypeError, ValueError) as error:
        print(f"Error: Could not save motion data: {error}")
        return False


def show_motion_statistics():
    """Read the JSON data with Pandas and display a summary."""
    try:
        data = pd.read_json(DATA_FILE)
    except (OSError, ValueError, TypeError) as error:
        print(f"Error: Could not read motion data with Pandas: {error}")
        return None

    if data.empty:
        data = pd.DataFrame(columns=DATA_COLUMNS)
    elif not all(column in data.columns for column in DATA_COLUMNS):
        print("Error: Motion data has an unexpected format.")
        return None
    else:
        motion_counts = pd.to_numeric(data["motion_count"], errors="coerce")
        valid_records = (
            data["date"].notna()
            & data["time"].notna()
            & data["motion_status"].isin(["MOTION DETECTED", "NO MOTION"])
            & motion_counts.notna()
            & (motion_counts >= 0)
        )
        if not valid_records.all():
            print("Warning: Invalid motion records were ignored.")
            data = data.loc[valid_records].copy()
            data["motion_count"] = motion_counts.loc[valid_records]

    detected = data["motion_status"] == "MOTION DETECTED"
    detected_count = int(detected.sum())
    no_motion_count = int(len(data) - detected_count)

    print("\nMotion Statistics:")
    if data.empty:
        print("No motion records were collected.")
    else:
        display_data = data[DATA_COLUMNS].rename(
            columns={
                "date": "Date",
                "time": "Time",
                "motion_status": "Motion Status",
                "motion_count": "Motion Count",
            }
        )
        print(display_data.to_string(index=False))

    print(f"\nTotal Records    : {len(data)}")
    print(f"Motion Detected  : {detected_count}")
    print(f"No Motion        : {no_motion_count}")
    return detected_count, no_motion_count


def create_motion_graph(statistics):
    """Save and display a bar graph of motion and no-motion records."""
    if statistics is None:
        return False

    detected_count, no_motion_count = statistics
    try:
        plt.figure(figsize=(7, 5))
        plt.bar(
            ["Motion Detected", "No Motion"],
            [detected_count, no_motion_count],
            color=["tomato", "steelblue"],
        )
        plt.title("Motion Statistics")
        plt.ylabel("Number of Records")
        plt.tight_layout()
        plt.savefig(GRAPH_FILE)
        print(f"Graph saved to: {GRAPH_FILE}")
        plt.show()
        return True
    except (OSError, ValueError, RuntimeError) as error:
        print(f"Error: Could not generate the graph: {error}")
        return False
    finally:
        plt.close()


def run_camera(records):
    """Capture webcam frames, detect motion, and collect one record per second."""
    camera = None
    camera_opened = False
    previous_gray = None
    last_recorded_time = None
    brightness_lookup = np.array(
        [((value / 255.0) ** 0.8) * 255 for value in range(256)],
        dtype=np.uint8,
    )

    try:
        camera = cv2.VideoCapture(0)
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        if not camera.isOpened():
            print("Camera Status : OFF")
            print("Error: Unable to access webcam.")
            print("Please check your camera connection.")
            return False

        camera_opened = True
        print("Camera Status : ON")
        print("Project Status: RUNNING")
        print("Project executed successfully!")
        print("\nFullscreen camera window opened. Press 'q' to exit.")
        window_name = "Simple Motion Detection System"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.setWindowProperty(
            window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN
        )

        while True:
            success, frame = camera.read()
            if not success or frame is None:
                print("Error: Could not read a frame from the webcam.")
                break

            now = datetime.now()
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            blurred_frame = cv2.GaussianBlur(gray_frame, (21, 21), 0)
            moving_rectangles = []

            if previous_gray is not None:
                difference = cv2.absdiff(previous_gray, blurred_frame)
                _, threshold_frame = cv2.threshold(
                    difference, 25, 255, cv2.THRESH_BINARY
                )
                threshold_frame = cv2.dilate(threshold_frame, None, iterations=2)
                contours = cv2.findContours(
                    threshold_frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                )[-2]

                for contour in contours:
                    if cv2.contourArea(contour) >= MIN_MOTION_AREA:
                        moving_rectangles.append(cv2.boundingRect(contour))
            previous_gray = blurred_frame
            motion_count = len(moving_rectangles)
            if motion_count:
                motion_status = "MOTION DETECTED"
                status_color = (0, 0, 255)
                for x, y, width, height in moving_rectangles:
                    cv2.rectangle(
                        frame, (x, y), (x + width, y + height), status_color, 2
                    )
            else:
                motion_status = "NO MOTION"
                status_color = (0, 255, 0)

            display_frame = cv2.LUT(frame, brightness_lookup)
            softened_frame = cv2.GaussianBlur(display_frame, (0, 0), 1.2)
            display_frame = cv2.addWeighted(
                display_frame, 1.25, softened_frame, -0.25, 0
            )

            panel_width = min(display_frame.shape[1] - 20, 460)
            panel_height = min(display_frame.shape[0] - 20, 190)
            panel = display_frame[
                10 : 10 + panel_height, 10 : 10 + panel_width
            ].copy()
            cv2.rectangle(
                panel, (0, 0), (panel_width, panel_height), (20, 20, 20), -1
            )
            cv2.addWeighted(
                panel,
                0.68,
                display_frame[10 : 10 + panel_height, 10 : 10 + panel_width],
                0.32,
                0,
                display_frame[10 : 10 + panel_height, 10 : 10 + panel_width],
            )
            overlay_lines = [
                ("MOTION DETECTION SYSTEM", (255, 255, 255), 0.65),
                ("Camera Status : ON", (0, 220, 255), 0.58),
                (f"Date          : {now.strftime('%d-%m-%Y')}", (255, 255, 255), 0.58),
                (f"Time          : {now.strftime('%H:%M:%S')}", (255, 255, 255), 0.58),
                (f"Motion Status : {motion_status}", status_color, 0.58),
                (f"Objects       : {motion_count}", (255, 255, 255), 0.58),
            ]
            for line_index, (text, color, font_scale) in enumerate(overlay_lines):
                cv2.putText(
                    display_frame,
                    text,
                    (20, 37 + line_index * 28),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale,
                    color,
                    2,
                    cv2.LINE_AA,
                )
            cv2.imshow(window_name, display_frame)

            timestamp = now.strftime("%d-%m-%Y %H:%M:%S")
            if timestamp != last_recorded_time:
                records.append(
                    {
                        "date": now.strftime("%d-%m-%Y"),
                        "time": now.strftime("%H:%M:%S"),
                        "motion_status": motion_status,
                        "motion_count": motion_count,
                    }
                )
                last_recorded_time = timestamp

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    except cv2.error as error:
        if camera_opened:
            print(f"Error: OpenCV camera operation failed: {error}")
        else:
            print("Camera Status : OFF")
            print("Error: Unable to access webcam.")
            print("Please check your camera connection.")
    except (OSError, ValueError) as error:
        if camera_opened:
            print(f"Error: Could not process a camera frame: {error}")
        else:
            print("Camera Status : OFF")
            print("Error: Unable to access webcam.")
            print("Please check your camera connection.")
    except KeyboardInterrupt:
        print("\nCamera stopped by the user.")
    finally:
        if camera is not None:
            try:
                camera.release()
            except cv2.error as error:
                print(f"Error: Could not release the webcam: {error}")
        try:
            cv2.destroyAllWindows()
        except cv2.error as error:
            print(f"Error: Could not close OpenCV windows: {error}")

    return camera_opened


def main():
    print("# ================================")
    print("MOTION DETECTION SYSTEM")
    print("Starting camera...")

    try:
        create_project_files()
    except OSError as error:
        print(f"Error: Could not create project files: {error}")
        return

    records = []
    camera_opened = run_camera(records)

    if not camera_opened:
        return

    data_saved = save_motion_data(records)
    statistics = show_motion_statistics()
    graph_saved = create_motion_graph(statistics)

    print("\n# ================================")
    print("PROGRAM COMPLETED")
    if data_saved:
        print("Motion data saved successfully.")
    if statistics is not None:
        print("Statistics generated successfully.")
    if graph_saved:
        print("Graph saved successfully.")
    print("Project executed successfully!")


if __name__ == "__main__":
    main()
