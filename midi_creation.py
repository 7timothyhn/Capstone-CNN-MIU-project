#!/usr/bin/env python3
"""
midi_creation_complete.py - Full pipeline for MIDI rendering via sfizz
"""

import re
import os
import subprocess
import argparse
from pathlib import Path
from mido import Message, MidiFile, MidiTrack

# Configuration
SFZ_PATH = r"C:\Users\Timothy\OneDrive\Documents\CPtest\Chapter_4_Exe\NoctSalamanderGrandPianoV4.1_48khz24bit\Noct-SalamanderGrandPiano_flat_withoutNoise.sfz"
SFIZZ_RENDER = r"C:\Users\Timothy\OneDrive\Documents\CPtest\Chapter_4_Exe\tools\sfizz\bin\Release\sfizz_render.exe"

DURATION_MAP = {"whole": 1920, "half": 960, "quarter": 480, "eighth": 240}


def get_next_filename(base_name="Midi_attempt", extension=".wav"):
    """Find the next available filename in sequence"""
    counter = 1
    while True:
        output_file = f"{base_name}{counter}{extension}"
        if not os.path.exists(output_file):
            return output_file
        counter += 1


def parse_report(report_path):
    """Extract and normalize note data from classification report"""
    notes = []
    with open(report_path) as f:
        for line in f:
            match = re.match(r"^\d+\s+([\w-]+)\s+(\S+)\s+\((\d+),", line)
            if match and match.group(2) != "N/A":
                symbol_class = match.group(1).lower().replace("-note", "")  # Normalize!
                notes.append(
                    {
                        "class": symbol_class,  # 'quarter', 'half', etc.
                        "pitch": match.group(2),  # e.g. 'E5'
                        "x_pos": int(match.group(3)),
                    }
                )
    return notes


def pitch_to_midi(pitch_name):
    """Convert pitch name to MIDI note number"""
    notes = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
    note_letter = pitch_name[0].upper()
    sharp = pitch_name[1] == "#"
    octave = int(pitch_name[2:] if sharp else pitch_name[1:])
    note_index = notes.index(note_letter + "#" if sharp else note_letter)
    return note_index + 12 * (octave + 1)


def create_midi_messages(notes):
    messages = []
    last_time = 0

    for note in notes:
        midi_note = pitch_to_midi(note["pitch"])
        duration_type = note.get("class", "quarter").lower()
        note_duration = DURATION_MAP.get(duration_type, 480)

        # Note ON: schedule with relative delay
        messages.append(Message("note_on", note=midi_note, velocity=100, time=0))

        # Note OFF: occurs after note_duration
        messages.append(
            Message("note_off", note=midi_note, velocity=0, time=note_duration)
        )
    return messages


def write_midi_file(messages, output_path="temp.mid"):
    mid = MidiFile()
    track = MidiTrack()
    mid.tracks.append(track)

    for message in messages:
        track.append(message)
    mid.save(output_path)
    return output_path


def render_audio_with_sfizz(midi_file, output_wav):
    """Render audio from MIDI using sfizz_render"""
    cmd = [SFIZZ_RENDER, "--sfz", SFZ_PATH, "--midi", midi_file, "--wav", output_wav]
    try:
        print("Executing:", " ".join(cmd))
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(result.stdout)
    except subprocess.CalledProcessError as e:
        print("Error during rendering:")
        print("Return code:", e.returncode)
        print("Output:", e.stdout)
        print("Error:", e.stderr)
        raise


def main():
    parser = argparse.ArgumentParser(description="Generate MIDI WAV from report")
    parser.add_argument("report", help="Classification report file")
    parser.add_argument(
        "--base-name", default="Midi_attempt", help="Base output filename"
    )
    args = parser.parse_args()

    notes = parse_report(args.report)
    if not notes:
        print("No valid notes found in report")
        return

    output_wav = get_next_filename(args.base_name)
    print(f"Creating {output_wav} with {len(notes)} notes...")

    try:
        midi_messages = create_midi_messages(notes)
        midi_path = write_midi_file(midi_messages)
        render_audio_with_sfizz(midi_path, output_wav)
        print(f"Successfully created: {output_wav}")
    except Exception as e:
        print(f"Failed to create audio: {e}")


if __name__ == "__main__":
    main()
