import json
import os

STATE_FILE = "downloader_state.json"

def load_state():
    """Loads timestamps and converts keys back to integers."""
    if not os.path.exists(STATE_FILE):
        return {}
    try:
        with open(STATE_FILE, 'r') as f:
            data = json.load(f)
            # JSON keys are always strings, so we convert them back to ints
            return {int(k): v for k, v in data.items()}
    except (ValueError, json.JSONDecodeError):
        return {} # Return empty if file is corrupted

def save_state(data):
    """Saves the current dictionary to a JSON file."""
    with open(STATE_FILE, 'w') as f:
        json.dump(data, f, indent=4)
