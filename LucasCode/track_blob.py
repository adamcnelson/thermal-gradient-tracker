import csv
import cv2
import argparse
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.seq_io import SeqReader


def inspect_seq(seq_path, max_frames=5):
    """
    Open a .seq file and print basic info from the first few frames.
    This is simply a test before actual tracking
    """
    count = 0

    with SeqReader(seq_path) as reader:
        for frame_idx, hdr, frame in reader.frames():
            print(f"Frame {frame_idx}")
            print(f"  shape: {frame.shape}")
            print(f" dtype: {frame.dtype}")
            print(f" min value: {frame.min()}")
            print(f" max value: {frame.max()}")
            print(f" mean value: {frame.mean()}")

            count += 1

            if count>= max_frames:
                break
        
    print(f"Read {count} frame(s).")


def save_frame_preview(seq_path, output_png="frame_preview.png", frame_number = 0):
    """
    Save the frist frame of the .seq file as a viewable PNG.
    This helps us see what Python is seeing/reading
    """
    with SeqReader(seq_path) as reader:
        for frame_idx, hdr, frame in reader.frames():
            if frame_idx != frame_number:
                continue

            frame_float = frame.astype(np.float32)

            #normalize raw thermal values to 0-255 for display only
            display = frame_float - frame_float.min()
            display = display / display.max()
            display = display * 255
            display = display.astype(np.uint8)

            cv2.imwrite(output_png, display)

            print(f"Saved frame {frame_idx} preview to {output_png}")
            print(f"Original frame shape: {frame.shape}")
            print(f"Original raw min: {frame.min()}")
            print(f"Original frame max: {frame.max()}")

            return
        
    print(f"Frame {frame_number} was not found.")


def save_background_preview(seq_path, output_png="background_preview.png", max_frames =200, every_n=50):
    """
    Sample frames throughout the video and make a median background image.
    The goal is to capture the stable termal gradient/arena, NOT the moving blob (mouse).
    """
    frames = []
    read_count = 0

    with SeqReader(seq_path) as reader:
        for frame_idx, hdr, frame in reader.frames():
            if read_count % every_n == 0:
                frames.append(frame.astype(np.float32))

            read_count += 1

            if len(frames) >= max_frames:
                break
    
    if not frames:
        print(f"No frames found.")
        return

    background = np.median(np.stack(frames), axis=0)

    #Normalize background to 0-25 for display only
    display = background - background.min()
    display = display / display.max()
    display = display * 255
    display = display.astype(np.uint8)

    cv2.imwrite(output_png, display)

    print(f"Saved background preview to {output_png}")
    print(f"Used {len(frames)} samples frames")
    print(f"Background shape: {background.shape}")
    print(f"Background min: {background.min()}")
    print(f"Background max: {background.max()}")


def save_difference_preview(seq_path, output_png="difference_preview.png", frame_number=1000, max_frames=100):
    """
    Build a background from the first max_frames frames,
    then compare one selected frame to that background.
    Save the absolute difference as a preview image.
    """

    
    # --- Build Background ---
    frames = []
    read_count = 0

    with SeqReader(seq_path) as reader:
        for frame_idx, hdr, frame in reader.frames():
            if read_count < max_frames:
                frames.append(frame.astype(np.float32))
                read_count += 1
            else:
                break
    
    if not frames:
        print(f"No frames found for background.")
        return
    
    background = np.median(np.stack(frames), axis=0)

    # ---Find the chosen frame---
    frames = []
    read_count = 0

    with SeqReader(seq_path) as reader:
        for frame_idx, hdr, frame in reader.frames():
            if read_count != frame_number:
                read_count += 1
                continue

            frame_float = frame.astype(np.float32)

            #Absolute difference between burrent frame and background
            diff = np.abs(frame_float - background)

            #Normalize background to 0-25 for display only
            display = diff - diff.min()
            display = display / display.max()
            display = display * 255
            display = display.astype(np.uint8)

            cv2.imwrite(output_png, display)

            print(f"Saved difference preview for frame {read_count} to {output_png}")
            print(f"Diff min: {diff.min()}")
            print(f"Diff max: {diff.max()}")

            return

    print(f"Frame {frame_number} was not found.")


def save_mask_preview(seq_path, output_png="mask_preview.png", frame_number=1000, max_frames=100):

    """
    Creates a mask from the absolute difference image
    """

    # --- Build Background ---
    frames = []
    read_count = 0

    with SeqReader(seq_path) as reader:
        for frame_idx, hdr, frame in reader.frames():
            if read_count < max_frames:
                frames.append(frame.astype(np.float32))
                read_count += 1
            else:
                break
    
    if not frames:
        print(f"No frames found for background.")
        return
    
    background = np.median(np.stack(frames), axis=0)

# --- Find selected frame ---
    read_count = 0

    with SeqReader(seq_path) as reader:
        for frame_idx, hdr, frame in reader.frames():
            if read_count !=frame_number:
                read_count += 1
                continue

            # Difference from background
            frame_float = frame.astype(np.float32)
            diff = np.abs(frame_float - background)

            # Threshold difference into black/white mask
            threshold_value = float(np.percentile(diff, 88))

            print(f"Diff min: {diff.min()}")
            print(f"Diff max: {diff.max()}")
            print(f"Diff mean: {diff.mean()}")
            print(f"Threshold value: {threshold_value}")

            mask = (diff >= threshold_value).astype(np.uint8) * 255

            mouse_mask, centroid, area = keep_largest_blob(mask, min_area=100)

            if mouse_mask is None:
                print(f"No mouse blob found.")
                cv2.imwrite(output_png, mask)
            else:
                cv2.imwrite(output_png, mouse_mask)
                print(f"Mouse centroid: {centroid}")
                print(f"Mouse area: {area}")

            print(f"Mask min: {mask.min()}")
            print(f"Mask max: {mask.max()}")
            print(f"White pixels: {np.sum(mask == 255)}")

            # Save mask
            cv2.imwrite(output_png, mask)
            
            print(f"Saved mask preview for counted frame {read_count} to {output_png}")
            print(f"Threshold value: {threshold_value}")

            return
    
    print(f"Frame {frame_number} was not found.")


def keep_largest_blob(mask, min_area=150):
    """
    Find all white blobs in the mask and keep the largest one
    Small blobs are then ignored/removed
    """

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)

    #print(f"Number of Labels found: {num_labels}")

    best_label = None
    best_area = 0

    # Start with 1 because label 0 is the black background
    for label in range(1, num_labels):
        area = stats[label, cv2.CC_STAT_AREA]
        #print(f"Checking label {label}, area {area}")

        if area >= min_area and area > best_area:
            #print(f"New best label: {label}, area {area}")
            best_label = label
            best_area = area 

    #print(f"Final best_label: {best_label}")
    #print(f"Final best_area: {best_area}")

    if best_label is None:
        return None, None, 0
        
    mouse_mask = (labels == best_label).astype(np.uint8) * 255
    x, y = centroids[best_label]

    return mouse_mask, (x, y), best_area


def choose_blob_with_continuity(
    mask,
    last_centroid=None,
    min_area=100,
    max_jump=50,
):
    """
    Choose the mouse blob from a binary b/w mask

    If there is no previous centroid, choose the largest blob.
    If there is a previous centroid, prefer the closer of acceptable blobs.
    This helps prevent the tracker from jumping between candidate blobs
    """

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)

    candidates = []

    for label in range(1, num_labels):  #Label 0 is background
        area = stats[label, cv2.CC_STAT_AREA]

        if area < min_area:
            continue
        
        x, y = centroids[label]

        if last_centroid is None:
            distance = None
        else:
            dx = x - last_centroid[0]
            dy = y - last_centroid[1]
            distance = float(np.sqrt(dx**2 + dy**2))
        
        candidates.append(
            {
                "label": label,
                "area": area,
                "centroid": (float(x), float(y)),
                "distance": distance,
            }
        )
    
    if len(candidates) == 0:
        return None, None, None, "missing"

    # First detection: Choose largest blob
    if last_centroid is None:
        chosen = max(candidates, key=lambda c: c["area"])
        status = "largest_first"

    else:
        # Prefer blobs closer to the previouscentroid
        close_candidates = [
            c for c in candidates
            if c["distance"] is not None and c["distance"] >= max_jump
        ]

        if len(close_candidates) > 0:
            chosen = max(close_candidates, key=lambda c: c["area"])
            status = "largest_nearby"
        else:
            # If nothing is cose, fall back to the largest blob 
            chosen = max(candidates, key=lambda c: c["area"])
            status = "largest_fallback"
        
    chosen_mask = np.zeros_like(mask, dtype=np.uint8)
    chosen_mask[labels == chosen["label"]] = 255

    return chosen_mask, chosen["centroid"], chosen["area"], status


def clean_mouse_mask(mouse_mask, close_size=3, open_size=3):
    """
    Clean the selected blob for each frame, make it a cleaner shape
    """

    if mouse_mask is None:
        return None
    
    close_kernel = np.ones((close_size, close_size), np.uint8)
    open_kernel = np.ones((open_size, open_size), np.uint8)

    #fill small gaps in the blob
    cleaned = cv2.morphologyEx(mouse_mask, cv2.MORPH_CLOSE, close_kernel)

    #remove tiny protrusions
    cleaned = cv2.morphologyEx(mouse_mask, cv2.MORPH_OPEN, open_kernel)

    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return mouse_mask
    
    largest_contour = max(contours, key=cv2.contourArea)

    filled = np.zeros_like(mouse_mask)
    cv2.drawContours(filled, [largest_contour], -1, 255, thickness=-1)

    return filled


def local_fallback_blob(diff, previous_centroid, search_radius=20, threshold_percentile=65, min_area=100):
    """
    Try to recover the mouse on frames where it is undetected by using last location and lowering threshold
    """
    if previous_centroid is None:
        return None, None, 0

    height, width = diff.shape
    prev_x, prev_y = previous_centroid
    prev_x = int(prev_x)
    prev_y = int(prev_y)

    #Define small search window
    x0 = max(prev_x - search_radius, 0)
    x1 = min(prev_x - search_radius +1, width)

    y0 = max(prev_y - search_radius, 0)
    y1 = min(prev_y + 1, height)

    roi = diff[y0:y1, x0:x1]

    if roi.size == 0:
        return None, None, 0

    #Lower the threshold to see a better outline
    threshold_value = float(np.percentile(roi, threshold_percentile))
    roi_mask = (roi >= threshold_value).astype(np.uint8) * 255

    #Connect broken larger speckles
    kernel = np.ones((3, 3), np.uint8)
    roi_mask = cv2.morphologyEx(roi_mask, cv2.MORPH_CLOSE, kernel)

    #Find largest blob
    roi_mouse_mask, roi_centroid, area = keep_largest_blob(roi_mask, min_area=min_area)

    if roi_mouse_mask is None:
        return None, None, 0
    
    # Convert local ROI mask back into a full size mask
    full_mouse_mask = np.zeros(diff.shape, dtype=np.uint8)
    full_mouse_mask[y0:y1, x0:x1] = roi_mouse_mask

    roi_x, roi_y = roi_centroid
    global_centroid = (roi_x + x0, roi_y + y0)

    return full_mouse_mask, global_centroid, area


def save_overlay_preview(seq_path, output_png="overlay_preivew.png", frame_number=1000, max_frames=100):

    """
    Shows/saves a preview of the outline and centroid created for the blob
    Good for testing on low-contrast areas and general tests
    """

    # --- Build Background ---
    frames = []
    read_count = 0

    with SeqReader(seq_path) as reader:
        for frame_idx, hdr, frame in reader.frames():
            if read_count < max_frames:
                frames.append(frame.astype(np.float32))
                read_count += 1
            else:
                break
    
    if not frames:
        print(f"No frames found for background.")
        return
    
    background = np.median(np.stack(frames), axis=0)

# --- Find selected frame ---
    read_count = 0

    with SeqReader(seq_path) as reader:
        for frame_idx, hdr, frame in reader.frames():
            if read_count !=frame_number:
                read_count += 1
                continue

            # Difference from background
            frame_float = frame.astype(np.float32)
            diff = np.abs(frame_float - background)

            # Threshold difference into black/white mask
            threshold_value = float(np.percentile(diff, 96.5))

            print(f"Diff min: {diff.min()}")
            print(f"Diff max: {diff.max()}")
            print(f"Diff mean: {diff.mean()}")
            print(f"Threshold value: {threshold_value}")

            mask = (diff >= threshold_value).astype(np.uint8) * 255

            #Keep only the largest blob for tracking purposes
            mouse_mask, centroid, area = keep_largest_blob(mask, min_area=100)

            """
            if mouse_mask is not None:
                mouse_mask = clean_mouse_mask(mouse_mask)
                mouse_mask, centroid, area = keep_largest_blob(mouse_mask, min_area=100)
            """

            if mouse_mask is None:
                print(f"No mouse blob found.")
                cv2.imwrite(output_png, mask)
                return 
            
            temps = calculate_temperatures(frame, background, mouse_mask)

            print(f"Tb mean: {temps['Tb_mean']}"),
            print(f"Tb median: {temps['Tb_median']}"),
            print(f"Tb_p95: {temps['Tb_p95']}"),

            print(f"location_mean: {temps['location_mean']}"),
            print(f"location_median: {temps['location_median']}"),
            print(f"location_p95: {temps['location_p95']}"),
            
            print(f"Mask min: {mask.min()}")
            print(f"Mask max: {mask.max()}")
            print(f"White pixels: {np.sum(mask == 255)}")

            low = np.percentile(frame_float, 1)
            high = np.percentile(frame_float, 97)

            display = np.clip(frame_float, low, high)
            diaply = display - low
            display = display / (high - low)
            display = display * 255
            display = display.astype(np.uint8)

            #convert grayscale image to color so we can draw
            overlay = cv2.cvtColor(display, cv2.COLOR_GRAY2BGR)
            
            #Fine and draw the outline of the mouse blob
            contours, _ = cv2.findContours(mouse_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(overlay, contours, -1, (0, 255, 0), 1)

            x, y = centroid 
            cv2.circle(overlay, (int(x), int(y)), 3, (0, 0, 255), -1)

            cv2.imwrite(output_png, overlay)

            print(f"Saved overlay preview to {output_png}")


            # Save mask
            print(f"Threshold value: {threshold_value}")
            print(f"Mouse Centroid: {centroid}")
            print(f"Mouse area: {area}")

            return
    
    print(f"Frame {frame_number} was not found.")


def raw_to_celsius(pixel_values):
    
    # Convert raw pixel values to degrees Celsius
    # Straight from 'temperatre_extraction.py'
    raw = np.asarray(pixel_values, dtype=np.float64)

    return (
        -2.52760392e-7 * raw**2
        + 0.013976934 * raw
        - 116.29242
    )

def calculate_temperatures(frame, background, mouse_mask):
    """
    Calculate the mouse Tb and assay-location temp using the same mouse mask
    Calculates location temperature my using background image with same overlay drawing (outline & centroid)
    """

    mouse_pixels_raw = frame[mouse_mask > 0]
    location_pixels_raw = background[mouse_mask > 0]

    if mouse_pixels_raw.size == 0:
        return None

    mouse_pixels = raw_to_celsius(mouse_pixels_raw)
    location_pixels = raw_to_celsius(location_pixels_raw)

    temps = {
        "Tb_mean": float(np.mean(mouse_pixels)),
        "Tb_median": float(np.median(mouse_pixels)),
        "Tb_p95": float(np.percentile(mouse_pixels, 95)),
        "Tb_max": float(np.max(mouse_pixels)),

        "location_mean": float(np.mean(location_pixels)),
        "location_median": float(np.median(location_pixels)),
        "location_p95": float(np.percentile(location_pixels, 95)),
        "location_max": float(np.max(location_pixels)),
    }

    return temps

def save_one_frame_csv(seq_path, output_csv="one_frame_csv", frame_number=1000, max_frames=100):
    """Save tracking + temperature for one frame into a CSV rather than printing
    Just a test before a full-video tracking run
    """

    # --- Build Background ---
    frames = []
    read_count = 0

    with SeqReader(seq_path) as reader:
        for frame_idx, hdr, frame in reader.frames():
            if read_count < max_frames:
                frames.append(frame.astype(np.float32))
                read_count += 1
            else:
                break
    
    if not frames:
        print(f"No frames found for background.")
        return
    
    background = np.median(np.stack(frames), axis=0)

# --- Find selected frame ---
    read_count = 0

    with SeqReader(seq_path) as reader:
        for frame_idx, hdr, frame in reader.frames():
            if read_count !=frame_number:
                read_count += 1
                continue

        # Difference from background
        frame_float = frame.astype(np.float32)
        diff = np.abs(frame_float - background)

        # Threshold difference into black/white mask
        threshold_value = float(np.percentile(diff, 96.5))

        mask = (diff >= threshold_value).astype(np.uint8) * 255

        #Keep only the largest blob for tracking purposes
        mouse_mask, centroid, area = keep_largest_blob(mask, min_area=100)



        if mouse_mask is None:
            print(f"No mouse blob found.")
            cv2.imwrite(output_png, mask)
            return
    
        temps = calculate_temperatures(frame, background, mouse_mask)

        x, y = centroid

        # write one CSV row
        with open(output_csv, "w", newline="") as f:
            writer = csv.writer(f)

            writer.writerow([
                "frame",
                "x",
                "y",
                "area_pixels",
                "Tb_mean in C",
                "Tb_median in C",
                "Tb_p95 in C",
                "Tb_max in C",
                "location_mean in C",
                "location_median in C",
                "location_p95 in C",
                "location_max in C",
            ])

            writer.writerow([
                read_count,
                float(x),
                float(y),
                area,
                temps["Tb_mean"],
                temps["Tb_median"],\
                temps["Tb_p95"],
                temps["Tb_max"],
                temps["location_mean"],
                temps["location_median"],
                temps["location_p95"],
                temps["location_max"],
            ])
        
        print(f"Saved one-frame CSV to {output_csv}")
        print(f"Frame: {read_count}")
        print(f"Centroid: {centroid}")
        print(f"Area: {area}")
        print(f"Tb mean: {temps['Tb_mean']}")
        print(f"Location mean: {temps['location_mean']}")
        print(hdr.keys())

        return

    print(f"Frame {frame_number} was not found")


def build_sampled_background(seq_path, num_background_frames=300):
    """
    Build a Median background image from frames sampled evenly across the entire video
    """

    total_frames = 0

    with SeqReader(seq_path) as reader:
        for frame_idx, hdr, frame in reader.frames():
            total_frames += 1

    if total_frames == 0:
        raise ValueError("No Frames Found in SEQ file")

    #Pick evenly spaced frame numbers
    num_to_sample = min(num_background_frames, total_frames)

    sampled_frame_numbers = np.linspace(
        0,
        total_frames - 1,
        num_to_sample,
        dtype=int,
    )

    sampled_frame_numbers = set(sampled_frame_numbers)

    #Collect sampled frames
    sampled_frames = []

    with SeqReader(seq_path) as reader:
        for frame_number, (frame_idx, hdr, frame) in enumerate(reader.frames()):
            if frame_number in sampled_frame_numbers:
                sampled_frames.append(frame.astype(np.float32))
    
    #median per pixel
    background = np.median(np.stack(sampled_frames), axis=0)

    print(f"Built sampled background from {len(sampled_frames)} frames across {total_frames} total frames.")

    return background


def build_initial_background(seq_path, max_background_frames=100):
    """
    Build a tracking background from the first frames of the video.

    This is better for mouse detection, because the mouse has not yet been dropped in
    so early no-mouse frames correctly look like no mouse
    """
    frames = []

    with SeqReader(seq_path) as reader:
        for frame_number, (frame_idx, hdr, frame), in enumerate(reader.frames()):
            if frame_number >= max_background_frames:
                break
            
            frames.append(frame.astype(np.float32))
        
    if len(frames) == 0:
        raise ValueError("No frames found for initial tracking background")

    background = np.median(np.stack(frames), axis=0)

    print(f"Built tracking background from first {len(frames)} frames")

    return background


def track_video_to_csv(seq_path, output_csv="tracking_output.csv", max_background_frames=100):

     # --- Build Background ---
    
    read_count = 0
    total_frames = 0
    detected_frames = 0
    local_fallback_frames = 0
    missing_frames = 0

    last_centroid = None
    last_centroid_frame = None
    missing_count = 0

    rejected_jump_frames = 0
    started_tracking = False
    skipped_initial_frames = 0

    """
    This is the old background code
    with SeqReader(seq_path) as reader:
        for frame_idx, hdr, frame in reader.frames():
            if read_count < max_background_frames:
                frames.append(frame.astype(np.float32))
                read_count += 1
            else:
                break
    """
    #New background code:
    tracking_background = build_initial_background(
        seq_path,
        max_background_frames=100,
    )

    temperature_background = build_sampled_background(
        seq_path,
        num_background_frames=300,
    )
    

    with open(output_csv, "w", newline="") as f:
        writer = csv.writer(f)

        writer.writerow([
            "frame",
            "x",
            "y",
            "area_pixels",
            "velocity_pixels_per_frame",
            "Tb_mean in C",
            "Tb_median in C",
            "Tb_p95 in C",
            "Tb_max in C",
            "location_mean in C",
            "location_median in C",
            "location_p95 in C",
            "location_max in C",
            "tracking_status in C",
            ])
    
        read_count = 0

        with SeqReader(seq_path) as reader:
            for frame_idx, hdr, frame in reader.frames():
                frame_number = read_count
                read_count += 1

                # Difference from background
                frame_float = frame.astype(np.float32)
                diff = np.abs(frame_float - tracking_background)

                # Threshold difference into black/white mask
                threshold_value = float(np.percentile(diff, 96.5))

                mask = (diff >= threshold_value).astype(np.uint8) * 255

                # Remove partial edges because of false blobs/artifacts
                # Edge_margin crops the video an additional X pixels to remove outliers
                edge_margin = 20

                # Save original mask in case we want a fallback later
                original_mask = mask.copy()

                # Remove consistent edge artifacts before choosing mouse blob
                mask[:, :edge_margin] = 0
                mask[:, -edge_margin:] = 0

                #Keep only the largest blob for tracking purposes
                mouse_mask, centroid, area = keep_largest_blob(mask, min_area=100)

                if mouse_mask is None and not started_tracking:
                    skipped_initial_frames += 1
                    
                    continue
                """
                if (
                    centroid is not None 
                    and last_centroid is not None
                    and last_centroid_frame is not None
                    and frame_number - last_centroid_frame <= 10
                ):
                    jump_distance = np.sqrt(
                        (centroid[0] - last_centroid[0]) ** 2
                        + (centroid[1] - last_centroid[1]) ** 2 
                    )

                    if jump_distance > 40:
                        centroid = None
                        mouse_mask = None
                        area = None
                        tracking_status = "rejected_jump"
                        rejected_jump_frames += 1
                """

                if mouse_mask is None:
                    mouse_mask, centroid, area = local_fallback_blob(
                        diff, 
                        last_centroid,
                        search_radius=30,
                        threshold_percentile=85,
                        min_area=10,
                    )
                
                    if mouse_mask is None:
                        writer.writerow([
                            frame_number,
                            "",
                            "",
                            0,
                            "",
                            "",
                            "",
                            "",
                            "",
                            "",
                            "",
                            "",
                            "",
                            "missing",
                        ])

                        missing_count += 1
                        missing_frames += 1
                        
                        continue

                    tracking_status = "local_fallback"

                else:
                    tracking_status = "detected"
                
                if mouse_mask is not None and not started_tracking:
                    started_tracking = True
                    print(f"First mouse detection at frame {read_count}")
                    

                if tracking_status == "detected":
                    detected_frames += 1
                elif tracking_status == "local_fallback":
                    local_fallback_frames += 1

                temps = calculate_temperatures(frame, temperature_background, mouse_mask)
                x, y = centroid
                velocity = ""

                if last_centroid is not None and last_centroid_frame is not None:
                    dx = float(x) - float(last_centroid[0])
                    dy = float(y) - float(last_centroid[1])

                    frame_gap = frame_number - last_centroid_frame

                    if frame_gap > 0:
                        velocity = float(np.sqrt(dx**2 + dy**2) / frame_gap)

                writer.writerow([
                    frame_number,
                    float(x),
                    float(y),
                    area,
                    velocity,
                    temps["Tb_mean"],
                    temps["Tb_median"],
                    temps["Tb_p95"],
                    temps["Tb_max"],
                    temps["location_mean"],
                    temps["location_median"],
                    temps["location_p95"],
                    temps["location_max"],
                    tracking_status,
                ])
                if centroid is not None:
                    x, y = centroid
                    last_centroid = centroid
                    last_centroid_frame = frame_number
                missing_count = 0

                if read_count % 1000 == 0:
                    print(f"Processed through frame {frame_number + 1}")

                total_frames += 1
    
    
    print(f"Done. Saved tracking CSV to {output_csv}")
    
    tracked_frames = detected_frames + local_fallback_frames
    total_recorded_frames = tracked_frames + missing_frames

    if total_frames > 0:
        tracked_percent = tracked_frames / total_recorded_frames * 100
    else:
        tracked_percent = 0
    

    print(f"Tracking summary:")
    print(f"  Skipped {skipped_initial_frames} initial frames with no mouse.")
    print(f"  Total recorded frames: {total_frames}")
    print(f"  Detected frames: {detected_frames}")
    print(f"  Local fallback frames: {local_fallback_frames}")
    print(f"  Missing frames: {missing_frames}")
    print(f"  rejected_jump_frames: {rejected_jump_frames}")
    print(f"  Tracked percent: {tracked_percent:.2f}%")


def print_calibration_raw_values(seq_path):
    target_frames = [999, 1000, 1001]

    cold_x = 381
    cold_y = 0

    hot_x = 30
    hot_y = 0

    with SeqReader(seq_path) as reader:
        for frame_number, (frame_idx, hdr, frame) in enumerate(reader.frames()):
            if frame_number in target_frames:
                cold_raw = frame[cold_y, cold_x]
                hot_raw = frame[hot_y, hot_x]

                print(f"Frame {frame_number}")
                print(f"  Cold raw at x={cold_x}, y={cold_y}: {cold_raw}")
                print(f"  Hot raw at x={hot_x}, y={hot_y}: {hot_raw}")
                
            if frame_number > max(target_frames):
                break


def check_celsius_validation_points(seq_path):
    """
    Compares FLIR celsius Values to python/seq raw pixel values
    To check the current conversion
    """

    # Input the numbers from your desired SEQ file as shown two lines below
    points = [
        # frame, label, x, y, FLIR Celsius
        (6559, "cold", 381, 28, 6.92),
        (6559, "cool", 305, 17, 16.05),
        (6559, "mid", 228, 11, 22.40),
        (6559, "warm", 147, 29, 28.78),
        (6559, "hot", 51, 18, 45.74),
    ]

    points_by_frame = {}

    for frame_number, label, x, y, flir_c in points:
        if frame_number not in points_by_frame:
            points_by_frame[frame_number] = []

        points_by_frame[frame_number].append((label, x, y, flir_c))

    last_frame_needed = max(points_by_frame.keys())

    with SeqReader(seq_path) as reader:
        for frame_number, (frame_idx, hdr, frame) in enumerate(reader.frames()):
            if frame_number in points_by_frame:
                print(f"\nFrame {frame_number}")

                for label, x, y, flir_c in points_by_frame[frame_number]:
                    raw = float(frame[y, x])
                    predicted_c = float(raw_to_celsius(raw))
                    error = predicted_c - flir_c
                    print(
                        f"{label}: "
                        f"x={x}, y={y}, "
                        f"raw={raw}, "
                        f"FLIR={flir_c:.2f}, "
                        f"predicted={predicted_c:.2f}, "
                        f"error={error:.2f}"
                    )
            if frame_number > last_frame_needed:
                break


def plot_background_drift(
    seq_path,
    output_png="background_drift_new.png",
    x=25,
    y=20,
    #roi_radius=2,
    sample_every=10,
):
    """
    This is the old background code
    background_frames = []

    with SeqReader(seq_path) as reader:
        for frame_number, (frame_idx, hdr, frame) in enumerate(reader.frames()):
            if frame_number < max_background_frames:
                background_frames.append(frame.astype(np.float32))
            else:
                break
    
    background = np.median(np.stack(background_frames), axis=0)
    """
    #New background code:
    background = build_sampled_background(
        seq_path,
        num_background_frames=300,
    )

    """
    height, width = background.shape
    
    x1 = max(0, y - roi_radius)
    x2 = min(height, y + roi_radius + 1)

    y1 = max(0, y - roi_radius)
    y2 = min(height, y + roi_radius + 1)
    """

    background_roi_raw = background[y, x]
    background_roi_c = raw_to_celsius(background_roi_raw)
    #background_mean_c = float(np.mean(background_roi_c))
    

    frame_numbers = []
    roi_temps_c = []
    drift_from_background_c = []

    with SeqReader(seq_path) as reader:
        for frame_number, (frame_idx, hdr, frame) in enumerate(reader.frames()):
            if frame_number % sample_every != 0:
                continue

            roi_raw = frame[y, x]
            roi_c = raw_to_celsius(roi_raw)
            #roi_mean_c = float(np.mean(roi_c))

            frame_numbers.append(frame_number)
            #roi_temps_c.append(roi_mean_c)
            drift_from_background_c.append(roi_c - background_roi_c)

    plt.figure(figsize=(10, 5))
    plt.plot(frame_numbers, drift_from_background_c)
    plt.axhline(0, linestyle="--")
    plt.xlabel("Frame")
    plt.ylabel("ROI Drift from background image (degrees C)")
    plt.title(f"Background drift check at x={x}, y={y}, ")
    plt.tight_layout()
    plt.savefig(output_png, dpi=200)
    plt.close()

    print(f"Saved background drift graph to {output_png}")
    #print(f"Background ROI mean: {background_mean_c:.2f} degrees C")
    print(f"Background point raw: {float(background_roi_raw):.2f}")
    print(f"Background point C: {background_roi_c:.2f}")
    print(f"Min drift: {min(drift_from_background_c):.2f} degrees C")
    print(f"Max drift: {max(drift_from_background_c):.2f} degrees C")


def main():
    parser = argparse.ArgumentParser(description="Track mouse blob in a cropped .seq file.")

    parser.add_argument("--input", required=True, help="Path to the cropped .seq file")
    parser.add_argument("--output", required=True, help="Path to output CSV file")
    parser.add_argument("--frame", type=int, default=1000, help="Frame number to preview")
    
    args = parser.parse_args()

    #These are the function calls with different arguments, rewrite or comment out the versions you do not want called

    #plot_background_drift(args.input, output_png="background_drift_new.png", x=25, y=20, sample_every=10)
    #check_celsius_validation_points(args.input)
    track_video_to_csv(args.input, args.output)
    #save_overlay_preview(args.input, args.output, args.frame, max_frames=100)

if __name__ == "__main__":
    main()