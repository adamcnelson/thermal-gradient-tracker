import argparse
import csv
import re
from pathlib import Path

import numpy as np


def rolling_median(values, window_size=11):
    values = np.array(values, dtype = float)

    if window_size < 1:
        return values
    
    if window_size % 2 == 0:
        window_size += 1
    
    half_window = window_size // 2
    smoothed = np.empty_like(values)

    for i in range(len(values)):
        start = max(0, i - half_window + 1)
        end = min(len(values), i + half_window + 1)
        smoothed[i] = np.median(values[start:end])

    return smoothed


def find_stationary_bouts(
    frame_numbers,
    x_positions,
    velocities,
    velocity_threshold=3.0,
    x_range_threshold=15.0,
    min_duration_s=5.0,
    fps=10,
    max_frame_gap=3,
):
    min_frames = int(min_duration_s * fps)

    bouts = []
    in_bout = False
    start_idx = None
    bout_x_values = []

    for i in range(len(velocities)):
        # Break a bout if there is a gap in tracked frames
        if i > 0 and frame_numbers[i] - frame_numbers[i - 1] > max_frame_gap:
            if in_bout:
                end_idx = i - 1
                duration_frames = frame_numbers[end_idx] - frame_numbers[start_idx] + 1

                if duration_frames >= min_frames:
                    bouts.append((start_idx, end_idx))
                
            in_bout = False
            start_idx = None 
            bout_x_values = []

        velocity_ok = velocities[i] <= velocity_threshold

        if not velocity_ok:
            if in_bout:
                end_idx = i - 1
                duration_frames = frame_numbers[end_idx] - frame_numbers[start_idx] + 1

                if duration_frames >= min_frames:
                    bouts.append((start_idx, end_idx))

            in_bout = False
            start_idx = None
            bout_x_values = []

            continue
        
        # Velocity is low enough at this point, so now check x coordinate stability
        if not in_bout:
            in_bout = True
            start_idx = i
            bout_x_values = [x_positions[i]]

        else:
            test_x_values = bout_x_values + [x_positions[i]]
            x_range = max(test_x_values) - min(test_x_values)

            if x_range <= x_range_threshold:
                bout_x_values.append(x_positions[i])
            else:
                end_idx = i - 1
                duration_frames = frame_numbers[end_idx] - frame_numbers[start_idx] + 1

                if duration_frames >= min_frames:
                    bouts.append((start_idx, end_idx))

                in_bout = True
                start_idx = i
                bout_x_values = [x_positions[i]]

    # Close the final bout if the video ends in a bout
    if in_bout:
        end_idx = len(velocities) - 1
        duration_frames = frame_numbers[end_idx] - frame_numbers[start_idx] + 1

        if duration_frames >= min_frames:
            bouts.append((start_idx, end_idx))
    
    return bouts


def parse_mouse_and_trial(csv_path):
    stem = Path(csv_path).stem

    # Remove the old tracking suffix (common ones/originals)
    for suffix in ["_tracked_edge20", "_tracked_antidrift", "_tracked"]:
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]

    # Detect whether this is Front or Back crop
    side_match = re.search(f"_(Front|Back)$", stem, flags=re.IGNORECASE)
    side = side_match.group(1).lower() if side_match else ""

    # Trial name = filename without Front/Back
    trial = re.sub(f"_(Front|Back)$", "", stem, flags=re.IGNORECASE)
    
    # MOUSE ID depends on crop side
    mouse = ""

    if side == "front":
        match = re.search(r"(\d{4}_F)", stem)

        if match:
            mouse = match.group(1)

    elif side == "back":
        match = re.search(r"(\d{4}_B)", stem)
        if match:
            mouse = match.group(1)

    # Fallback if the Front/Back logic fails
    if mouse == "":
        match = re.search(r"(\d{4}_[FB])", stem)
         
        if match:
            mouse = match.group(1)
        else:
            mouse = stem
    
    return mouse, trial


def read_tracking_csv(csv_path):
    frame_numbers = []
    x_positions = []
    velocities = []
    body_temps = []
    ambient_temps = []

    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f, skipinitialspace=True)

        for row in reader:
            frame_text = row.get("frame", row.get("frame_number", "")).strip()
            x_text = row.get("x", "").strip()
            velocity_text = row.get("velocity_pixels_per_frame", "").strip()
            body_text = row.get("Tb_median in C", "").strip()
            ambient_text = row.get("location_median in C", "").strip()

            if (
                frame_text == ""
                or x_text == ""
                or velocity_text == ""
                or body_text == ""
                or ambient_text == ""
            ):
                continue

            try:
                frame_numbers.append(int(frame_text))
                x_positions.append(float(x_text))
                velocities.append(float(velocity_text))
                body_temps.append(float(body_text))
                ambient_temps.append(float(ambient_text))
            except ValueError:
                continue

    return (
        np.array(frame_numbers),
        np.array(x_positions),
        np.array(velocities),
        np.array(body_temps),
        np.array(ambient_temps),
    )


def summarize_bouts_for_file(
    csv_path,
    fps=10,
    velocity_threshold=3.0,
    x_range_threshold=15,
    min_duration_s=20,
    smoothing_window=11,
):
    mouse, trial = parse_mouse_and_trial(csv_path)

    frame_numbers, x_positions, velocities, body_temps, ambient_temps = read_tracking_csv(csv_path)

    if len(frame_numbers) == 0:
        print(f"No usable rows in {csv_path}")
        return []

    x_smooth = rolling_median(x_positions, window_size=smoothing_window)
    velocity_smooth = rolling_median(velocities, window_size=smoothing_window)

    bouts = find_stationary_bouts(
        frame_numbers,
        x_smooth,
        velocity_smooth,
        velocity_threshold=velocity_threshold,
        x_range_threshold=x_range_threshold,
        min_duration_s=min_duration_s,
        fps=fps,
    )

    rows = []

    for bout_number, (start_idx, end_idx) in enumerate(bouts, start=1):
        start_frame = frame_numbers[start_idx]
        end_frame = frame_numbers[end_idx]

        bout_duration_s = (end_frame - start_frame + 1) / fps
        body_temp_C = np.median(body_temps[start_idx : end_idx + 1])
        ambient_temp_C = np.median(ambient_temps[start_idx : end_idx + 1])

        rows.append(
            {
                "mouse": mouse,
                "trial": trial,
                "bout_number": bout_number,
                "bout_duration_s": round(float(bout_duration_s), 3),
                "body_temp_C": round(float(body_temp_C), 3),
                "ambient_temp_C": round(float(ambient_temp_C), 3),
            }
        )
    
    print(f"{Path(csv_path).name}: {len(rows)} bouts")
    return rows


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--input-dir", required=True, help="Folder containing cleaned and tracked CSV files")
    parser.add_argument("--output-csv", required=True, help="Combined output bout summary CSV")
    parser.add_argument("--recursive", action="store_true", help="Search input folder recursively")

    parser.add_argument("--fps", type=float, default=10)
    parser.add_argument("--velocity-threshold", type=float, default=3.0),
    parser.add_argument("--x-range-threshold", type=float, default=15)
    parser.add_argument("--min-duration-s", type=float, default=20)
    parser.add_argument("--smoothing-window", type=int, default=11)

    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_csv = Path(args.output_csv)

    if args.recursive:
        csv_files = sorted(input_dir.rglob("*.csv"))
    else:
        csv_files = sorted(input_dir.glob("*.csv"))

    all_rows = []

    for csv_path in csv_files:
        rows = summarize_bouts_for_file(
            csv_path,
            fps=args.fps,
            velocity_threshold=args.velocity_threshold,
            x_range_threshold=args.x_range_threshold,
            min_duration_s=args.min_duration_s,
            smoothing_window=args.smoothing_window,
        )

        all_rows.extend(rows)

    output_csv.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "mouse",
        "trial",
        "bout_number",
        "bout_duration_s",
        "body_temp_C",
        "ambient_temp_C",
    ]

    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)

    
    print()
    print(f"Saved bout sumary CSV to {output_csv}")
    print(f"Total bout rows written: {len(all_rows)}")


if __name__ == "__main__":
    main()