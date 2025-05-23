# DashCop E-Challan System

This system automatically processes license plate data from video analysis, matches it with user records in a MongoDB database, and sends e-challans via Twilio API.

## Features

- MongoDB integration for storing user and challan data
- License plate validation and matching
- Automated e-challan generation
- SMS and WhatsApp notifications via Twilio API
- Comprehensive logging

## Setup

1. Install the required dependencies:

```bash
pip install pymongo python-dotenv twilio
```

2. Create a `.env` file in the backend directory with the following configuration:

```
# MongoDB Connection
MONGO_URI=mongodb://localhost:27017/

# Twilio API Credentials
TWILIO_ACCOUNT_SID=your_account_sid_here
TWILIO_AUTH_TOKEN=your_auth_token_here
TWILIO_PHONE_NUMBER=your_twilio_phone_number_here

# Application Configuration
OUTPUT_DIRS=C:/path/to/output,C:/path/to/video2,C:/path/to/video3

# Logging Configuration
LOG_LEVEL=INFO
```

3. Make sure MongoDB is running on your system.

## Usage

### Adding a Test User

To add a test user to the database, uncomment the `add_test_user()` function call in `challan_generator.py` and run the script:

```bash
python challan_generator.py
```

### Processing License Plates and Sending Challans

To process license plates from the output directories and send challans:

```bash
python challan_generator.py
```

## System Architecture

The system consists of the following components:

1. **Database Connection (`db_connection.py`)**
   - Handles MongoDB connection and operations
   - Provides functions for adding and retrieving users and challans

2. **License Plate Utilities (`license_plate_utils.py`)**
   - Validates license plate format
   - Extracts license plates from JSON files
   - Matches license plates with users

3. **Twilio Utilities (`twilio_utils.py`)**
   - Handles Twilio API integration
   - Formats and sends SMS and WhatsApp messages

4. **Challan Generator (`challan_generator.py`)**
   - Main module that ties everything together
   - Processes license plates and generates challans
   - Sends notifications to users

## License Plate Format

The system validates license plates in the following format:
- Two letters (state code)
- Two digits
- Alphanumeric sequence (vehicle identifier)

Example: `MH12AB1234`

## Troubleshooting

- Check the `challan_generator.log` file for detailed logs
- Ensure MongoDB is running and accessible
- Verify Twilio credentials are correct
- Make sure the output directories contain valid JSON files with license plate data 