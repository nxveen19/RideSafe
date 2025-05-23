import re
import json
import os
import glob
from datetime import datetime
from typing import List, Dict, Any, Tuple

def is_valid_license_plate(plate: str) -> bool:
    """
    Validate if a license plate matches the format: two letters (state code), 
    two digits, and an alphanumeric sequence.
    
    Args:
        plate (str): License plate to validate
        
    Returns:
        bool: True if valid, False otherwise
    """
    # Pattern: two letters followed by two digits followed by alphanumeric sequence
    pattern = r'^[A-Z]{2}\d{2}[A-Z0-9]+$'
    return bool(re.match(pattern, plate))

def extract_license_plates_from_json(json_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Extract license plates from JSON data with the correct structure.
    
    Args:
        json_data (dict): JSON data containing license plate information
        
    Returns:
        list: List of license plate entries with text and confidence
    """
    license_plates = []
    
    # Check if the JSON has the expected structure
    if "detections" in json_data and "license_plates" in json_data["detections"]:
        for plate in json_data["detections"]["license_plates"]:
            # Only include plates that have text and confidence
            if "text" in plate and "confidence" in plate:
                license_plates.append({
                    "text": plate["text"],
                    "confidence": plate["confidence"],
                    "timestamp": plate.get("timestamp", ""),
                    "frame_id": plate.get("frame_id", 0)
                })
    
    return license_plates

def get_license_plates_from_directory(directory: str) -> List[Dict[str, Any]]:
    """
    Get all license plates from all JSON files in a directory.
    
    Args:
        directory (str): Directory containing JSON files
        
    Returns:
        list: List of license plate entries with text and confidence
    """
    all_plates = []
    
    # Check if directory exists
    if not os.path.exists(directory):
        print(f"Directory not found: {directory}")
        return all_plates
    
    # Get all JSON files in the directory
    json_files = [f for f in os.listdir(directory) if f.endswith('.json')]
    
    if not json_files:
        print(f"No JSON files found in directory: {directory}")
        return all_plates
    
    # Process each JSON file
    for json_file in json_files:
        file_path = os.path.join(directory, json_file)
        try:
            with open(file_path, 'r') as f:
                json_data = json.load(f)
                
            # Extract license plates from this file
            plates = extract_license_plates_from_json(json_data)
            
            # Add source file information to each plate
            for plate in plates:
                plate["source_file"] = json_file
            
            all_plates.extend(plates)
            
        except Exception as e:
            print(f"Error processing file {json_file}: {e}")
    
    return all_plates

def get_unique_license_plates(plates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Get unique license plates, keeping the one with highest confidence for each plate.
    
    Args:
        plates (list): List of license plate entries
        
    Returns:
        list: List of unique license plates with highest confidence
    """
    # Group plates by text
    plate_groups = {}
    
    for plate in plates:
        text = plate["text"]
        if text not in plate_groups or plate["confidence"] > plate_groups[text]["confidence"]:
            plate_groups[text] = plate
    
    # Convert back to list
    return list(plate_groups.values())

def match_license_plate_with_user(plate: str, users: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Match a license plate with a user in the database.
    
    Args:
        plate (str): License plate to match
        users (list): List of users from the database
        
    Returns:
        dict: Matched user or None if not found
    """
    for user in users:
        if user.get("license_plate") == plate:
            return user
    
    return None

def scan_output_directories(base_dir, directories):
    """
    Scan multiple output directories for license plate data.
    
    Args:
        base_dir (str): Base directory path
        directories (list): List of directory names to scan
        
    Returns:
        dict: Dictionary mapping directory names to lists of license plates
    """
    results = {}
    
    for directory in directories:
        dir_path = os.path.join(base_dir, directory)
        if not os.path.exists(dir_path):
            print(f"Directory not found: {dir_path}")
            continue
            
        # Find all JSON files in the directory
        json_files = glob.glob(os.path.join(dir_path, "tracking_*.json"))
        
        all_plates = []
        for json_file in json_files:
            plates = extract_license_plates_from_json(json_file)
            all_plates.extend(plates)
        
        # Remove duplicates based on plate number
        unique_plates = {}
        for plate in all_plates:
            plate_num = plate['plate']
            if plate_num not in unique_plates or plate['confidence'] > unique_plates[plate_num]['confidence']:
                unique_plates[plate_num] = plate
        
        results[directory] = list(unique_plates.values())
    
    return results

def match_license_plates_with_users(plates, db_utils):
    """
    Match license plates with users in the database.
    
    Args:
        plates (list): List of license plate dictionaries
        db_utils (module): Database utilities module
        
    Returns:
        list: List of matched users with their license plates
    """
    matches = []
    
    for plate_data in plates:
        plate_number = plate_data['plate']
        user = db_utils.get_user_by_license_plate(plate_number)
        
        if user:
            matches.append({
                'user': user,
                'plate_data': plate_data
            })
    
    return matches 