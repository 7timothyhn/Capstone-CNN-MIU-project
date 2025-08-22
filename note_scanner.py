import cv2
import numpy as np
import tensorflow as tf
import argparse
import os
import shutil
import sys
from datetime import datetime

# Configure GPU memory growth
gpus = tf.config.experimental.list_physical_devices("GPU")
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

# Complete class names mapping
CLASS_NAMES = {
    0: "12-8-Time",
    1: "2-2-Time",
    2: "2-4-Time",
    3: "3-4-Time",
    4: "3-8-Time",
    5: "4-4-Time",
    6: "6-8-Time",
    7: "9-8-Time",
    8: "Barline",
    9: "C-Clef",
    10: "Common-Time",
    11: "Cut-Time",
    12: "Dot",
    13: "Double-Flat",
    14: "Double-Sharp",
    15: "Eighth-Grace-Note",
    16: "Eighth-Note",
    17: "Eighth-Rest",
    18: "F-Clef",
    19: "Flat",
    20: "G-Clef",
    21: "Half-Note",
    22: "Multiple-Half-Notes",
    23: "Natural",
    24: "Onehundred-Twenty-Eight-Note",
    25: "Quarter-Note",
    26: "Quarter-Rest",
    27: "Sharp",
    28: "Sixteenth-Note",
    29: "Sixteenth-Rest",
    30: "Sixty-Four-Note",
    31: "Sixty-Four-Rest",
    32: "Thirty-Two-Note",
    33: "Thirty-Two-Rest",
    34: "Whole-Half-Rest",
    35: "Whole-Note",
}


def save_current_script(destination_dir):
    """Save a copy of this script to output directory"""
    try:
        script_path = os.path.abspath(sys.argv[0])
        dest_path = os.path.join(destination_dir, "note_scanner.py")
        shutil.copy2(script_path, dest_path)
        print(f"Saved script copy to: {dest_path}")
    except Exception as e:
        print(f"Error saving script copy: {e}")


def create_output_structure(sheet_path, classifier_path):
    """Create organized output directory structure"""
    base_dir = "Generated_Observations"
    counter = 1
    while os.path.exists(f"{base_dir}_{counter}"):
        counter += 1

    output_dir = f"{base_dir}_{counter}"
    os.makedirs(output_dir, exist_ok=True)

    # Save input files
    shutil.copy2(sheet_path, os.path.join(output_dir, "sheet_music_used.png"))
    shutil.copy2(classifier_path, os.path.join(output_dir, "model_used.h5"))

    # Create Observations directory and subdirectories
    observations_dir = os.path.join(output_dir, "Observations")
    os.makedirs(observations_dir, exist_ok=True)

    # Create all subdirectories inside Observations
    os.makedirs(os.path.join(observations_dir, "symbol_results"), exist_ok=True)
    os.makedirs(os.path.join(observations_dir, "deconstruction"), exist_ok=True)
    os.makedirs(os.path.join(observations_dir, "staff_lines"), exist_ok=True)
    os.makedirs(os.path.join(observations_dir, "reconstruction"), exist_ok=True)

    # Create empty classification report in parent directory
    parent_report_path = os.path.join(
        os.path.dirname(output_dir), f"classification_report_{counter}.txt"
    )
    open(parent_report_path, "w").close()

    # Save current version of the script
    save_current_script(output_dir)

    return output_dir


def detect_staff_lines(image):
    """Detect staff lines and save visualization"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (50, 1))
    detected_lines = cv2.morphologyEx(
        binary, cv2.MORPH_OPEN, horizontal_kernel, iterations=2
    )

    # Save staff line visualization
    staff_visualization = image.copy()
    contours, _ = cv2.findContours(
        detected_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    cv2.drawContours(staff_visualization, contours, -1, (0, 255, 0), 2)
    cv2.imwrite(
        os.path.join(output_dir, "Observations", "staff_lines", "detected_staff.png"),
        staff_visualization,
    )

    staff_lines = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        staff_lines.append(y + h // 2)

    staff_lines.sort()
    return [staff_lines[i : i + 5] for i in range(0, len(staff_lines), 5)]


def remove_staff_lines(image):
    """Remove staff lines and save intermediate steps"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    cv2.imwrite(
        os.path.join(output_dir, "Observations", "deconstruction", "01_binary.png"),
        binary,
    )

    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (50, 1))
    detected_lines = cv2.morphologyEx(
        binary, cv2.MORPH_OPEN, horizontal_kernel, iterations=2
    )
    cv2.imwrite(
        os.path.join(
            output_dir, "Observations", "deconstruction", "02_staff_lines.png"
        ),
        detected_lines,
    )

    cleaned = cv2.inpaint(image, detected_lines, 3, cv2.INPAINT_TELEA)
    cv2.imwrite(
        os.path.join(output_dir, "Observations", "deconstruction", "03_cleaned.png"),
        cleaned,
    )

    return cleaned


def determine_pitch(y_pos, staves):
    """Determine musical pitch based on vertical position"""
    note_sequence = [
        "E5",
        "D5",
        "C5",
        "B4",
        "A4",
        "G4",
        "F4",
        "E4",
        "D4",
        "C4",
        "B3",
        "A3",
        "G3",
        "F3",
        "E3",
        "D3",
        "C3",
    ]
    for staff in staves:
        if len(staff) < 5:
            continue

        staff_height = staff[-1] - staff[0]
        line_spacing = staff_height / 4
        position = (y_pos - staff[0]) / (line_spacing / 2)
        note_index = round(position) + 2
        note_index = max(0, min(note_index, len(note_sequence) - 1))

        return note_sequence[note_index]

    return "Unknown"


def process_symbol(image, contour, model, symbol_id, staves):
    """Process and classify an individual symbol"""
    try:
        x, y, w, h = cv2.boundingRect(contour)
        symbol_img = image[y : y + h, x : x + w]

        # Save original symbol
        symbol_path = os.path.join(
            output_dir, "Observations", "symbol_results", f"symbol_{symbol_id}.png"
        )
        cv2.imwrite(symbol_path, symbol_img)

        # Classification - resize to 192x96 to match model input
        resized = cv2.resize(symbol_img, (96, 192))
        input_tensor = tf.convert_to_tensor([resized], dtype=tf.float32)
        prediction = model.predict(input_tensor, verbose=0)
        class_id = np.argmax(prediction)
        class_name = CLASS_NAMES.get(class_id, f"Unknown_{class_id}")
        confidence = np.max(prediction)

        # Pitch determination for notes
        pitch = ""
        if class_name in ["Whole-Note", "Half-Note", "Quarter-Note", "Eighth-Note"]:
            pitch = determine_pitch(y + h // 2, staves)
            if w > h * 1.5:
                pitch = "Unknown"
            elif h < 5 or w < 5:
                pitch = "Unknown"

        # Save processed symbol
        processed_path = os.path.join(
            output_dir, "Observations", "symbol_results", f"processed_{symbol_id}.png"
        )
        cv2.imwrite(processed_path, resized)

        return x, y, w, h, class_id, class_name, confidence, resized, pitch

    except Exception as e:
        print(f"Error processing symbol {symbol_id}: {e}")
        return None


def generate_report(results, output_dir):
    """Generate comprehensive classification report"""
    # Main report path inside output directory
    report_path = os.path.join(output_dir, "classification_report.txt")

    # Parent directory report path
    counter = os.path.basename(output_dir).split("_")[-1]
    parent_report_path = os.path.join(
        os.path.dirname(output_dir), f"classification_report_{counter}.txt"
    )

    # Write to both files
    for path in [report_path, parent_report_path]:
        with open(path, "w") as f:
            f.write("MUSIC SYMBOL CLASSIFICATION REPORT\n")
            f.write("=" * 80 + "\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 80 + "\n\n")

            # Summary statistics
            valid_results = [r for r in results if r is not None]
            note_count = sum(
                1
                for r in valid_results
                if r[5] in ["Whole-Note", "Half-Note", "Quarter-Note", "Eighth-Note"]
            )
            pitch_count = sum(1 for r in valid_results if r[8] not in ["", "Unknown"])

            f.write(f"Total Symbols Detected: {len(valid_results)}\n")
            f.write(f"Notes Detected: {note_count}\n")
            f.write(f"Pitches Determined: {pitch_count}/{note_count}\n\n")

            # Pitch distribution
            if note_count > 0:
                f.write("PITCH DISTRIBUTION:\n")
                pitch_counts = {}
                for r in valid_results:
                    if r[8] not in ["", "Unknown"]:
                        pitch_counts[r[8]] = pitch_counts.get(r[8], 0) + 1

                for pitch, count in sorted(pitch_counts.items()):
                    f.write(f"{pitch}: {count}\n")
                f.write("\n")

            # Detailed symbol listing
            f.write("DETECTED SYMBOLS:\n")
            f.write("-" * 80 + "\n")
            f.write(
                f"{'ID':<5}{'Class':<20}{'Pitch':<10}{'Position':<15}{'Confidence':<10}{'Size':<10}\n"
            )
            f.write("-" * 80 + "\n")

            for i, r in enumerate(valid_results):
                x, y, w, h, _, class_name, confidence, _, pitch = r
                position_str = f"({x},{y})"
                f.write(
                    f"{i:<5}{class_name[:20]:<20}{pitch if pitch else 'N/A':<10}{position_str:<15}{confidence:.2f}{f'{w}x{h}':<10}\n"
                )


def main():
    parser = argparse.ArgumentParser(description="Sheet Music Classifier")
    parser.add_argument("--sheet", required=True, help="Input sheet music image")
    parser.add_argument("--classifier", required=True, help="Classifier model file")
    parser.add_argument(
        "--output", help="Output image name (default: result_from_note_scanner.png)"
    )
    args = parser.parse_args()

    # Setup output directory
    global output_dir
    output_dir = create_output_structure(args.sheet, args.classifier)
    result_image = os.path.join(
        output_dir, args.output if args.output else "result_of_note_scanner.png"
    )

    # Load and verify image
    original = cv2.imread(args.sheet)
    if original is None:
        print(f"Error: Could not load image {args.sheet}")
        return

    # Detect staff lines
    staves = detect_staff_lines(original)
    if not staves:
        print("Warning: No staff lines detected - using fallback positions")
        staves = [[y for y in range(100, 500, 50)]]

    # Remove staff lines
    cleaned = remove_staff_lines(original)

    # Symbol detection
    gray = cv2.cvtColor(cleaned, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=lambda c: cv2.boundingRect(c)[0])

    # Load model
    try:
        model = tf.keras.models.load_model(args.classifier)
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # Process symbols with 2-letter abbreviations
    results = []
    output_img = original.copy()
    for i, contour in enumerate(contours):
        if cv2.contourArea(contour) < 20:
            continue

        result = process_symbol(original, contour, model, i, staves)
        if not result:
            continue

        x, y, w, h, _, class_name, confidence, _, pitch = result

        # Draw bounding box
        color = (0, 255, 0) if confidence > 0.8 else (0, 0, 255)
        cv2.rectangle(output_img, (x, y), (x + w, y + h), color, 2)

        # Create 2-letter abbreviation + pitch label
        abbrev = class_name.replace("-", "")[:2].title()
        label = f"{abbrev} {pitch}" if pitch else abbrev

        # Calculate text position
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        tx = x + (w - tw) // 2
        ty = y - 5

        # Draw label with background
        cv2.rectangle(
            output_img, (tx - 2, ty - th - 2), (tx + tw + 2, ty + 2), (0, 0, 0), -1
        )
        cv2.putText(
            output_img,
            label,
            (tx, ty),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

        results.append(result)

    # Save final results and reconstruction
    cv2.imwrite(result_image, output_img)
    cv2.imwrite(
        os.path.join(output_dir, "Observations", "reconstruction", "final_output.png"),
        output_img,
    )
    generate_report(results, output_dir)

    print(f"\nAnalysis complete. Results saved to: {output_dir}")
    print(
        f"Classification report: {os.path.join(output_dir, 'classification_report.txt')}"
    )
    print(f"Result image: {result_image}")


if __name__ == "__main__":
    main()
