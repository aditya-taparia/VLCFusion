import json
from datetime import datetime, timedelta
import numpy as np

# --- Constants ---
SCENARIOS = ['02003', '02005', '02007', '02009', '02011', '02013', '02015', '02017', '02019']
# Classes from your 'frame_and_time data'
CLASSES_FILTER = [1, 2, 5, 6, 9, 10, 11, 12, 13, 14, 15]

TGTTYPE_TO_ID = {
    'PICKUP': 1, 'SUV': 2, 'BTR70': 5, 'BRDM2': 6, 'BMP2': 9,
    'T62': 10, 'T72': 11, 'ZSU23': 12, '2S3': 13, 'MTLB': 14, 'D20': 15,
}

# From your 'dataset creation code' (Note: T62 (10) is commented out there for tgtid_to_categories)
TGTID_TO_CATEGORIES = {
    1: 0,  # Pickup
    2: 1,  # SUV
    5: 2,  # BTR70
    6: 3,  # BRDM2
    9: 4,  # BMP2
    # 10: 5, # T62 (commented out in original, implies class 5 is T72)
    11: 5, # T72 (was 6 if T62 was present, now 5)
    12: 6, # ZSU23 (was 7, now 6)
    13: 7, # 2S3 (was 8, now 7)
    14: 8, # MTLB (was 9, now 8)
    15: 9  # D20 (was 10, now 9)
}
# Ensure TGTTYPE_TO_ID keys for TGTID_TO_CATEGORIES are present
VALID_TGTTYPES_FOR_CATEGORIES = {k for k, v in TGTTYPE_TO_ID.items() if v in TGTID_TO_CATEGORIES}


TGTID_TO_LABELS = {
    1: 'Pickup', 2: 'SUV', 5: 'BTR70', 6: 'BRDM2', 9: 'BMP2',
    10: 'T62', 11: 'T72', 12: 'ZSU23', 13: '2S3', 14: 'MTLB', 15: 'D20'
}

DISTANCE_TAG_MAPPING = {
    "2003": "1000", "2005": "1500", "2007": "2000", "2009": "2500",
    "2011": "3000", "2013": "3500", "2015": "4000", "2017": "4500", "2019": "5000",
}

DIMENSIONS = { # length, width, height (in meters)
    'PICKUP': [5.41, 1.8, 1.68], 'SUV': [4.57, 1.73, 1.73],
    'BTR70': [7.62, 2.79, 2.16], 'BTR': [7.62, 2.79, 2.16], # BTR alias for BTR70
    'BRDM2': [5.72, 2.29, 2.03], 'BMP2': [6.73, 3.15, 2.45],
    'T72': [6.59, 3.59, 2.21], 'ZSU23': [5.94, 2.88, 2.34],
    '2S3': [6.71, 3.23, 2.72], 'MTLB': [6.25, 3.1, 2.3], 'D20': [8.35, 2.4, 1.8],
    'T62': [9.34, 3.3, 2.2] # Added T62 dimensions, assuming this might be needed.
}

# --- Utility Functions ---
def read_json_file(file_path):
    """Reads a JSON file and returns its content."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data
    except FileNotFoundError:
        print(f"Error: JSON file not found at {file_path}")
        return None
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {file_path}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred while reading {file_path}: {e}")
        return None

def julian_to_datetime(year, julian_day, hour, minute, second, millisecond):
    """Converts Julian date components to a Python datetime object."""
    try:
        # Convert the Julian day to a proper month and day
        date_part = datetime(int(year), 1, 1) + timedelta(days=int(julian_day) - 1)
        # Add time
        final_dt = datetime(
            date_part.year, date_part.month, date_part.day,
            int(hour), int(minute), int(second), int(millisecond) * 1000
        )
        return final_dt
    except ValueError as e:
        print(f"Error converting Julian date: Y={year} D={julian_day} H={hour} M={minute} S={second} MS={millisecond}. Error: {e}")
        return None # Or raise error

def convert_to_serializable_for_json(obj):
    """Converts numpy types to Python native types for JSON serialization."""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.datetime64):
        # Convert numpy.datetime64 to ISO 8601 string format
        # To convert to a Python datetime object first:
        # timestamp = (obj - np.datetime64('1970-01-01T00:00:00Z')) / np.timedelta64(1, 's')
        # return datetime.utcfromtimestamp(timestamp).isoformat() + 'Z'
        return str(obj) # Simpler string representation
    elif isinstance(obj, (datetime, timedelta)):
        return obj.isoformat()
    else:
        # For other types, attempt to convert to string, or raise TypeError
        try:
            return str(obj)
        except Exception:
            raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")

