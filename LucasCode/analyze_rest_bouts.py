import argparse
import csv
import numpy as np
import matplotlib.pyplot as plt



def summarize_velocity(csv_path):
    velocities = []
    total_rows = 0
    missing_velocity_rows = 0

    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)


        for row in reader:
            total_rows += 1

            velocity_text = row.get("velocity_pixels_per_frame", "").strip()

            if velocity_text == "":
                missing_velocity_rows +=1
                continue

            try:
                velocity = float(velocity_text)
            except ValueError:
                missing_velocity_rows += 1
                continue
            
            velocities.append(velocity)

    if len(velocities) == 0:
        print("No velocity values found.")
        return
    
    velocities = np.array(velocities)

    percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]

    print(f"Velocity Summary:")
    print(f"CSV: {csv_path}")
    print(f"Total rows: {total_rows}")
    print(f"Rows with numeric velocity: {len(velocities)}")
    print(f"Rows with missing velocity: {missing_velocity_rows}")
    print()

    for p in percentiles:
        value = np.percentile(velocities, p)
        print(f"p{p}: {value:.4f} pixels/frame")
    
    print()
    print(f"Min: {np.min(velocities):.4f} pixels/frame")
    print(f"Max: {np.max(velocities):.4f} pixels/frame")
    print(f"Mean: {np.mean(velocities):.4f} pixels/frame")

    candidate_thresholds = [0.1, 0.25, 0.5, 0.75, 1, 1.5, 2]

    print()
    print(f"Candidate Thresholds")

    for threshold in candidate_thresholds:
        count_below = np.sum(velocities <= threshold)
        percent_below = count_below / len(velocities) * 100

        print(
            f"<= {threshold:.2f} pixels/frame:"
            f"{count_below} frames"
            f"({percent_below:.2f}%)"
        )


def plot_velocity_over_time(csv_path, output_png="velocity_over_time.png"):

    """
    This function creates an image of a graph for velocity over time 
    for manula visual analysis
    """

    frame_numbers = []
    velocities = []

    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)

        for row in reader:
            frame_text = row.get("frame", "").strip()
            velocity_text = row.get("velocity_pixels_per_frame", "").strip()

            if frame_text == "" or velocity_text == "":
                continue

            try:
                frame_number = int(frame_text)
                velocity = float(velocity_text)
            except ValueError:
                continue
            
            frame_numbers.append(frame_number)
            velocities.append(velocity)

    if len(velocities) == 0:
        print("No velocity values found for plotting.")
        return
    
    zoom_max = np.percentile(velocities, 95)
            
    plt.figure(figsize=(12, 5))
    plt.ylim(0, zoom_max)
    plt.plot(frame_numbers, velocities, linewidth=0.8)
    plt.xlabel("Frame")
    plt.ylabel("Velocity (pixels/frame)")
    plt.title("Velocity over time")
    plt.tight_layout()
    plt.savefig(output_png, dpi=200)
    plt.close()

    print(f"Saved velocity graph to {output_png}")
    print(f"Plotted {len(velocities)} velocity values")


def plot_position_velocity_diagnostic(
    csv_path,
    output_png="position_velocity_diagnostic.png",
    fps=10,
    velocity_ylim_percentile=98,
    rest_velocity_threshold=3.0,
):
    times_s = []
    x_positions = []
    velocities = []


    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)

        for row in reader:
            frame_text = row.get(f"frame", row.get(f"frame_number", "")).strip()
            x_text = row.get("x", "").strip()
            velocity_text = row.get("velocity_pixels_per_frame", "").strip()

            if frame_text == "" or x_text == "" or velocity_text == "":
                continue

            try:
                frame_number = int(frame_text)
                x_position = float(x_text)
                velocity = float(velocity_text)

            except ValueError:
                continue

            times_s.append(frame_number / fps)
            x_positions.append(x_position)
            velocities.append(velocity)

    if len(velocities) == 0:
        print("No usable x/velocity values found for plotting.")
        return
        
    times_s = np.array(times_s)
    x_positions = np.array(x_positions)
    velocities = np.array(velocities)
    x_smooth = rolling_median(x_positions, window_size=11)
    velocity_smooth = rolling_median(velocities, window_size=11)

    bouts = find_stationary_bouts(
        times_s,
        x_smooth,
        velocity_smooth,
        velocity_threshold=3.0,
        x_range_threshold=15.0,
        min_duration_s=20.0,
        fps=fps,
    )

    velocity_ymax = 15

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(12, 6),
        sharex=True,
    )

    # Panel 1: x position over time
    """ >>>UNSMOOTHED OLD VERSION
    axes[0].plot(
        times_s,
        x_positions,
        marker=".",
        linestyle="None",
        markersize=1.5,
        label="raw x"
    )
    """
    axes[0].plot(
        times_s,
        x_smooth,
        linewidth=1.0,
        label="smoothed x"
    )
    axes[0].set_ylabel("centroid x (px)")
    axes[0].set_title("Mouse x position over time")
    axes[0].legend()

    # Panel 2: Velocity over time
    """   >>  UNSMOOTHED OLD VERSION
    axes[1].plot(
        times_s,
        velocities,
        marker=".",
        linestyle="None",
        markersize=1.5,
    )
    """
    axes[1].plot(
        times_s,
        velocity_smooth,
        marker=".",
        linestyle="None",
        markersize=1.0,
        label="smoothed velocity"
    )
    axes[1].axhline(
        rest_velocity_threshold,
        linestyle="--",
        linewidth=1.2,
        label=f"velocity threshold = {rest_velocity_threshold} px/frame",
    )
    axes[1].legend()
    axes[1].set_ylabel("Velocity (px/frame)")
    axes[1].set_ylim(0, velocity_ymax)
    axes[1].set_title("Velocity over time")
    axes[1].set_xlabel("Time (s)")

    for start_idx, end_idx in bouts:
        start_time = times_s[start_idx]
        end_time = times_s[end_idx]

        axes[0].axvspan(start_time, end_time, alpha=0.2)
        axes[1].axvspan(start_time, end_time, alpha=0.2)

    plt.tight_layout()
    plt.savefig(output_png, dpi=200)
    plt.close()

    print(f"Saved diagnostic plot to {output_png}")
    print(f"Plotted {len(velocities)} points.")
    below_threshold = np.sum(velocities <= rest_velocity_threshold)
    percent_below = below_threshold / len(velocities) * 100

    print(f"Candidate rest threshold: {rest_velocity_threshold} px/frame")
    print(f"Frames at or below threshold: {below_threshold} / {len(velocities)} ({percent_below:.2f}%)")

    total_bout_time_s = 0
    for start_idx, end_idx in bouts:
        total_bout_time_s += times_s[end_idx] - times_s[start_idx]

    total_time_s = times_s[-1] - times_s[0]
    percent_bout_time = total_bout_time_s / total_time_s * 100

    print(f"Detected bouts: {len(bouts)}")
    print(f"Total bout time: {total_bout_time_s:.1f} s")
    print(f"Percent bout time: {percent_bout_time:.2f}%")


def rolling_median(values, window_size=11):
    """
    Smooth a 1D signal with a rolling median

    This is useful for removing short tracking jumps/outliers
    without changing CSV data
    """
    values = np.array(values, dtype=float)

    if window_size < 1:
        return values
    
    if window_size % 2 == 0:
        window_size += 1
    
    half_window = window_size // 2
    smoothed = np.empty_like(values)

    for i in range(len(values)):
        start = max(0, i - half_window)
        end = min(len(values), i + half_window + 1)
        smoothed[i] = np.median(values[start:end])
    
    return smoothed
    

def find_stationary_bouts(
    times_s, 
    x_positions,
    velocities,
    velocity_threshold=3.0,
    x_range_threshold=15.0,
    min_duration_s=20.0,
    fps=10,
):
    min_frames = int(min_duration_s * fps)

    bouts = []
    in_bout = False
    start_idx = None
    bout_x_values = []

    for i in range(len(velocities)):
        velocity_ok = velocities[i] <= velocity_threshold

        if not velocity_ok:
            if in_bout:
                end_idx = i - 1
                duration_frames = end_idx - start_idx + 1
            
                if duration_frames >= min_frames:
                    bouts.append((start_idx, end_idx))
                
            in_bout = False
            start_idx = None
            bout_x_values = []
            continue

        # Velocity is low enough, so check whether the position is also stable
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
                #Current point makes the position too spread out
                #End the previous bout before this point
                end_idx = i - 1
                duration_frames = end_idx - start_idx + 1

                if duration_frames >= min_frames:
                    bouts.append((start_idx, end_idx))
                
                #Start a new bout/check at the current point
                in_bout = True
                start_idx = i 
                bout_x_values = [x_positions[i]]
    
    #End final bout if video ends during a bout
    if in_bout:
        end_idx = len(velocities) - 1
        duration_frames = end_idx - start_idx + 1

        if duration_frames >= min_frames:
            bouts.append((start_idx, end_idx))

    return bouts


def main():

    """
    This is where all of the above functions are called
    """

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Tracked CSV file")
    parser.add_argument("--output", default="velocity_over_time.png", help="Output PNG") 
    parser.add_argument("--fps", type=float, default=10, help="Frames per second")
    args = parser.parse_args()

    #summarize_velocity(args.input)
    #plot_velocity_over_time(args.input, args.output)
    plot_position_velocity_diagnostic(args.input, args.output, fps=args.fps)


if __name__ == "__main__":
    main()