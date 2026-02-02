"""Test Script to Simulate LSL Streams.

1. Gaze Stream (4 channels: Gaze_X, Gaze_Y, PupilDiameter, GazeConfidence) - 60 Hz
2. EEG Stream (21 channels: Standard 10-20 system with Fp1, Fp2, F7, F3, Fz, F4, F8, T7,
    C3, Cz, C4, T8, P7, P3, Pz, P4, P8, O1, O2, A1, A2)
3. Physiological Stream (5 channels: HeartRate, SkinConductance, BreathingRate, 
    BloodPressure, BodyTemperature) - 100 Hz
4. Audio Marker Stream (Markers) - single-channel string markers (e.g., 'beep', 
    'speech_start')

Run this script before starting your GUI application to ensure that the streams are 
available, with proper channel metadata so that your data_inlet code can read them.
"""

import sys
import threading
import time

import numpy as np
from pylsl import StreamInfo, StreamOutlet


def create_gaze_stream() -> None:
    """Creates a Gaze LSL stream with four channels.

    1. Gaze_X (deg)
    2. Gaze_Y (deg)
    3. PupilDiameter (mm)
    4. GazeConfidence (percent)
    """
    channel_labels = ["Gaze_X", "Gaze_Y", "PupilDiameter", "GazeConfidence"]
    info = StreamInfo(
        name="GazeStream",
        type="Gaze",
        channel_count=len(channel_labels),
        channel_format="float32",
        nominal_srate=60,
        source_id="gaze_stream_001",
    )

    # Populate channel labels and units in the LSL 'channels' node
    channels_node = info.desc().append_child("channels")
    units = ["deg", "deg", "mm", "percent"]
    for lbl, unit in zip(channel_labels, units):
        chan = channels_node.append_child("channel")
        chan.append_child_value("label", lbl)
        chan.append_child_value("unit", unit)

    outlet = StreamOutlet(info)
    print("GazeStream created and sending data...")

    while True:
        # Simulate gaze coordinates between -30 and 30 degrees
        gaze_x = np.random.uniform(-30, 30)
        gaze_y = np.random.uniform(-30, 30)
        # Simulate pupil diameter between 3 and 7 mm
        pupil_diameter = np.random.uniform(3, 7)
        # Simulate gaze confidence as a percentage (0-100)
        gaze_confidence = np.random.uniform(0, 100)
        sample = [gaze_x, gaze_y, pupil_diameter, gaze_confidence]
        outlet.push_sample(sample)
        time.sleep(1.0 / 60.0)  # 60 Hz


def create_eeg_stream() -> None:
    """Creates an EEG LSL stream with standard 10-20 system channels.

    Based on international 10-20 system with 19 recording electrodes + 2 reference 
    electrodes. Standard clinical EEG setup with realistic voltage ranges.
    Units: microvolts (µV)
    """
    # Standard 10-20 system with 19 recording electrodes + 2 reference (A1, A2)
    eeg_channels = [
        "Fp1",
        "Fp2",
        "F7",
        "F3",
        "Fz",
        "F4",
        "F8",
        "T7",
        "C3",
        "Cz",
        "C4",
        "T8",
        "P7",
        "P3",
        "Pz",
        "P4",
        "P8",
        "O1",
        "O2",
        "A1",
        "A2",
    ]

    info = StreamInfo(
        name="EEGStream",
        type="EEG",
        channel_count=len(eeg_channels),
        channel_format="float32",
        nominal_srate=250,  # Standard clinical sampling rate
        source_id="eeg_stream_001",
    )

    # Populate channel labels and units
    channels_node = info.desc().append_child("channels")
    for lbl in eeg_channels:
        chan = channels_node.append_child("channel")
        chan.append_child_value("label", lbl)
        chan.append_child_value("unit", "µV")

    outlet = StreamOutlet(info)
    print("EEGStream created and sending data...")

    while True:
        # Simulate realistic EEG data with proper voltage ranges
        eeg_data = []
        for i, channel in enumerate(eeg_channels):
            # Base noise level: realistic EEG ranges from 10-100 µV
            if channel in ["A1", "A2"]:  # Reference electrodes
                # Reference electrodes typically have lower amplitude
                voltage = np.random.normal(0, 5)
            else:
                # Mix of different frequency bands with realistic amplitudes
                # Alpha waves (8-12 Hz): higher amplitude in occipital/parietal
                alpha_component = 0
                if channel in ["O1", "O2", "P3", "P4", "Pz", "P7", "P8"]:
                    alpha_component = 30 * np.sin(2 * np.pi * 10 * time.time())

                # Beta waves (13-30 Hz): prominent in frontal areas
                beta_component = 0
                if channel in ["Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8"]:
                    beta_component = 15 * np.sin(2 * np.pi * 20 * time.time())

                # Theta waves (4-7 Hz): present in temporal areas
                theta_component = 0
                if channel in ["T7", "T8"]:
                    theta_component = 20 * np.sin(2 * np.pi * 6 * time.time())

                # Background noise and mixed activity
                background = np.random.normal(0, 25)  # Increased noise level

                # Combine components
                voltage = (
                    alpha_component + beta_component + theta_component + background
                )

                # Add occasional spikes/artifacts (more realistic)
                if np.random.random() < 0.001:  # 0.1% chance of artifact
                    voltage += np.random.normal(0, 100)

            eeg_data.append(voltage)

        outlet.push_sample(eeg_data)
        time.sleep(0.004)  # 250 Hz sampling rate


def create_physiological_stream() -> None:
    """Creates a Physiological LSL stream with five channels.

    1. HeartRate (bpm)
    2. SkinConductance (µS)
    3. BreathingRate (breaths/min)
    4. BloodPressure (mmHg)
    5. BodyTemperature (°C)
    """
    physio_channels = [
        ("HeartRate", "bpm"),
        ("SkinConductance", "µS"),
        ("BreathingRate", "breaths/min"),
        ("BloodPressure", "mmHg"),
        ("BodyTemperature", "°C"),
    ]
    info = StreamInfo(
        name="PhysioStream",
        type="Physiological",
        channel_count=len(physio_channels),
        channel_format="float32",
        nominal_srate=100,
        source_id="physio_stream_001",
    )

    # Populate channel labels and units
    channels_node = info.desc().append_child("channels")
    for lbl, unit in physio_channels:
        chan = channels_node.append_child("channel")
        chan.append_child_value("label", lbl)
        chan.append_child_value("unit", unit)

    outlet = StreamOutlet(info)
    print("PhysioStream created and sending data...")

    while True:
        # Simulate physiological data:
        heart_rate = np.random.uniform(60, 100)  # bpm
        skin_conductance = np.random.uniform(0.5, 5.0)  # µS
        breathing_rate = np.random.uniform(12, 20)  # breaths/min
        blood_pressure = np.random.uniform(80, 120)  # mmHg (simulated mean pressure)
        body_temperature = np.random.uniform(36.5, 37.5)  # °C
        sample = [
            heart_rate,
            skin_conductance,
            breathing_rate,
            blood_pressure,
            body_temperature,
        ]
        outlet.push_sample(sample)
        time.sleep(1.0 / 100.0)  # 100 Hz


def create_event_markers_stream() -> None:
    """Creates an Audio Marker LSL stream that sends string markers.

    This is a single-channel marker stream (e.g., event codes, audio markers).
    Samples are strings describing the marker (e.g., 'start', 'beep', 'stop').
    """
    # Single-channel marker stream with string samples
    info = StreamInfo(
        name="EventMarkerStream",
        type="Markers",
        channel_count=1,
        channel_format="string",
        source_id="event_marker_stream_001",
    )

    # Optional: add a description node for marker types
    channels_node = info.desc().append_child("channels")
    chan = channels_node.append_child("channel")
    chan.append_child_value("label", "Marker")
    chan.append_child_value("unit", "label")

    outlet = StreamOutlet(info)
    print("EventMarkerStream created and sending marker strings...")

    marker_types = [
        "Onset_Experiment",
        "Onset_Event1",
        "Offset_Event1",
        "Onset_Event2",
        "Offset_Event2",
        "Onset_Event3",
        "Offset_Event3",
        "Onset_Event4",
        "Offset_Event4",
        "Offset_Experiment",
    ]
    time.sleep(15)
    while True:
        time.sleep(30)
        if len(marker_types) > 0:
            outlet.push_sample([str(marker_types[0])])
            marker_types.pop(0)
        else:
            print("All markers sent. Stopping EventMarkerStream.")
            break


def create_microphone_stream() -> None:
    """Creates a Microphone LSL stream that simulates audio input.

    This is a single-channel stream that sends string markers indicating audio events.
    """
    channel_labels = ["Left", "Right"]
    info = StreamInfo(
        name="MicrophoneStream",
        type="Audio",
        channel_count=len(channel_labels),
        channel_format="float32",
        nominal_srate=44100,
        source_id="microphone_stream_001",
    )

    # Populate channel labels and units in the LSL 'channels' node
    channels_node = info.desc().append_child("channels")
    for lbl in channel_labels:
        chan = channels_node.append_child("channel")
        chan.append_child_value("label", lbl)
        chan.append_child_value("unit", "dB")

    outlet = StreamOutlet(info)
    print("MicrophoneStream created and sending data...")

    sample_rate = 44100
    chunk_size = 1024
    freq_left = 220.0
    freq_right = 224.0
    noise_level = 0.015

    dt = 1.0 / sample_rate
    chunk_duration = chunk_size / sample_rate
    t = 0.0

    print("Streaming synthetic stereo microphone via LSL at 44.1 kHz...")

    while True:
        times = t + np.arange(chunk_size) * dt

        # Left channel
        left = 0.6 * np.sin(2 * np.pi * freq_left * times) + 0.3 * np.sin(
            2 * np.pi * 2 * freq_left * times
        )

        # Right channel
        right = 0.6 * np.sin(2 * np.pi * freq_right * times) + 0.3 * np.sin(
            2 * np.pi * 2 * freq_right * times
        )

        # Shared speech-like envelope
        envelope = 0.5 * (1 + np.sin(2 * np.pi * 2.2 * times))

        left = envelope * left + noise_level * np.random.randn(chunk_size)
        right = envelope * right + noise_level * np.random.randn(chunk_size)

        stereo_chunk = np.column_stack([left, right])
        stereo_chunk = np.clip(stereo_chunk, -1.0, 1.0).astype(np.float32)
        # print(stereo_chunk)

        outlet.push_chunk(stereo_chunk.tolist())

        t += chunk_size * dt
        time.sleep(chunk_duration)


def create_camera_stream() -> None:
    """Single-channel marker stream with string samples."""
    info = StreamInfo(
        name="CameraFrameStream",
        type="Frames",
        channel_count=1,
        channel_format="int32",
        nominal_srate=30,
        source_id="camera_frame_stream_001",
    )
    # Optional: add a description node for marker types
    channels_node = info.desc().append_child("channels")
    chan = channels_node.append_child("channel")
    chan.append_child_value("label", "Frame")
    chan.append_child_value("unit", "frame_number")

    frame_counter = 1
    outlet = StreamOutlet(info)
    print("CameraFrameStream created and sending frame numbers...")
    while True:
        outlet.push_sample([frame_counter])
        frame_counter += 1
        time.sleep(1.0 / 30.0)  # 30 Hz


if __name__ == "__main__":
    import threading

    # Create threads for each stream to run concurrently
    gaze_thread = threading.Thread(target=create_gaze_stream, daemon=True)
    eeg_thread = threading.Thread(target=create_eeg_stream, daemon=True)
    physio_thread = threading.Thread(target=create_physiological_stream, daemon=True)
    event_markers_thread = threading.Thread(
        target=create_event_markers_stream, daemon=True
    )
    microphone_thread = threading.Thread(target=create_microphone_stream, daemon=True)
    webcam_thread = threading.Thread(target=create_camera_stream, daemon=True)

    gaze_thread.start()
    eeg_thread.start()
    physio_thread.start()
    webcam_thread.start()
    # time.sleep(15)
    event_markers_thread.start()
    microphone_thread.start()

    print("All test streams are running. Press Ctrl+C to exit.")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Test streams stopped.")
        sys.exit(0)
