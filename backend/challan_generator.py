import os
import json
import logging
from datetime import datetime
from dotenv import load_dotenv

from db_connection import get_database, add_user, get_user_by_license_plate, add_challan
from license_plate_utils import (
    is_valid_license_plate, 
    get_license_plates_from_directory,
    get_unique_license_plates,
    match_license_plate_with_user
)
from twilio_utils import send_challan_sms, send_challan_whatsapp

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("challan_generator.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Get output directories from environment variables
OUTPUT_DIRS = os.getenv("OUTPUT_DIRS", "").split(",")
# Filter out empty strings in case there are trailing commas
OUTPUT_DIRS = [dir_path.strip() for dir_path in OUTPUT_DIRS if dir_path.strip()]

# def add_test_user():
#     """
#     Add a test user to the database for testing purposes.
#     """
#     test_user = {
#         "name": "akash pandey",
#         "license_plate": "GA11B7547",
#         "phone_number": "+919082223606",
#         "email": "pandeynaveen2002@gmail.com",
#         "address": "123 Test Street, Test City"
#     }
    
#     user_id = add_user(test_user)
#     if user_id:
#         logger.info(f"Test user added with ID: {user_id}")
#     else:
#         logger.error("Failed to add test user")

def process_license_plates():
    """
    Process license plates from all output directories and generate challans.
    """
    all_plates = []
    
    # Process each output directory
    for directory in OUTPUT_DIRS:
        logger.info(f"Processing directory: {directory}")
        
        # Get all license plates from this directory
        plates = get_license_plates_from_directory(directory)
        logger.info(f"Found {len(plates)} license plates in {directory}")
        
        all_plates.extend(plates)
    
    # Get unique license plates (highest confidence for each)
    unique_plates = get_unique_license_plates(all_plates)
    logger.info(f"Found {len(unique_plates)} unique license plates")
    
    # Process each unique license plate
    for plate_data in unique_plates:
        plate_text = plate_data["text"].strip().upper()
        
        # Validate license plate format
        if not is_valid_license_plate(plate_text):
            logger.warning(f"Invalid license plate format: {plate_text}")
            continue
        
        # Find user with this license plate
        user = get_user_by_license_plate(plate_text)
        
        if user:
            logger.info(f"Found user for license plate {plate_text}: {user['name']}")
            
            # Generate challan
            challan_data = {
                "user_id": str(user["_id"]),
                "license_plate": plate_text,
                "violation_type": "No Helmet",
                "fine_amount": 500,
                "location": "Main Street",
                "timestamp": datetime.now().isoformat(),
                "status": "Pending",
                "source_file": plate_data.get("source_file", "Unknown")
            }
            
            # Add challan to database
            challan_id = add_challan(challan_data)
            
            if challan_id:
                logger.info(f"Challan generated with ID: {challan_id}")
                
                # Send notification to user
                if "phone_number" in user:
                    # Send SMS
                    sms_sent = send_challan_sms(user["phone_number"], challan_data)
                    if sms_sent:
                        logger.info(f"SMS sent to {user['phone_number']}")
                    else:
                        logger.error(f"Failed to send SMS to {user['phone_number']}")
                    
                    # Send WhatsApp message
                    whatsapp_sent = send_challan_whatsapp(user["phone_number"], challan_data)
                    if whatsapp_sent:
                        logger.info(f"WhatsApp message sent to {user['phone_number']}")
                    else:
                        logger.error(f"Failed to send WhatsApp message to {user['phone_number']}")
            else:
                logger.error(f"Failed to generate challan for license plate {plate_text}")
        else:
            logger.warning(f"No user found for license plate {plate_text}")

if __name__ == "__main__":
    logger.info("Starting challan generator")
    
    # Uncomment the following line to add a test user
    # add_test_user()
    
    # Process license plates and generate challans
    process_license_plates()
    
    logger.info("Challan generator completed") 