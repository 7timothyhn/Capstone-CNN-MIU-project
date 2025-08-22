#!/usr/bin/env python3
"""
music_pipeline.py - Complete pipeline from sheet music to audio
"""

import os
import subprocess
import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Complete music processing pipeline")
    parser.add_argument("--sheet", required=True, help="Input sheet music image")
    parser.add_argument(
        "--classifier", required=True, help="Classifier model file (.h5)"
    )
    parser.add_argument(
        "--output-wav", help="Output WAV filename (default: generated_music.wav)"
    )
    args = parser.parse_args()

    # Step 1: Run note_scanner to generate classification report
    print("\n=== Running Sheet Music Analysis ===")
    note_scanner_cmd = [
        "python",
        "note_scanner.py",
        "--sheet",
        args.sheet,
        "--classifier",
        args.classifier,
    ]

    try:
        subprocess.run(note_scanner_cmd, check=True)
    except subprocess.CalledProcessError as e:
        print("Error during sheet music analysis:", e)
        return

    # Find the generated classification report
    output_dir = max(
        [d for d in os.listdir() if d.startswith("Gernerated_Observations_")],
        key=os.path.getmtime,
    )
    report_path = os.path.join(output_dir, "classification_report.txt")

    if not os.path.exists(report_path):
        print("Error: Could not find generated classification report")
        return

    # Step 2: Run midi_creation to generate WAV file
    print("\n=== Generating Audio from Analysis ===")
    output_wav = args.output_wav if args.output_wav else "generated_music.wav"
    midi_creation_cmd = [
        "python",
        "midi_creation.py",
        report_path,
        "--base-name",
        Path(output_wav).stem,
    ]

    try:
        subprocess.run(midi_creation_cmd, check=True)
        print(f"\nSuccessfully generated audio file: {output_wav}")
    except subprocess.CalledProcessError as e:
        print("Error during audio generation:", e)


if __name__ == "__main__":
    main()
