import os
from twilio.rest import Client
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Twilio credentials
TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN")
TWILIO_PHONE_NUMBER = os.getenv("TWILIO_PHONE_NUMBER")

def initialize_twilio_client():
    """
    Initialize and return a Twilio client.
    
    Returns:
        twilio.rest.Client: Twilio client or None if initialization fails
    """
    if not all([TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN, TWILIO_PHONE_NUMBER]):
        print("Twilio credentials not properly configured. Please check your .env file.")
        return None
    
    try:
        client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
        return client
    except Exception as e:
        print(f"Error initializing Twilio client: {e}")
        return None

def send_challan_sms(phone_number, challan_data):
    """
    Send an e-challan SMS to the user's phone number.
    
    Args:
        phone_number (str): User's phone number
        challan_data (dict): Challan data including violation details
        
    Returns:
        bool: True if SMS sent successfully, False otherwise
    """
    client = initialize_twilio_client()
    if not client:
        return False
    
    try:
        # Format the message
        message = format_challan_message(challan_data)
        
        # Send the message
        message = client.messages.create(
            body=message,
            from_=TWILIO_PHONE_NUMBER,
            to=phone_number
        )
        
        print(f"Challan SMS sent to {phone_number}. SID: {message.sid}")
        return True
    except Exception as e:
        print(f"Error sending challan SMS: {e}")
        return False

def format_challan_message(challan_data):
    """
    Format the challan data into a readable SMS message.
    
    Args:
        challan_data (dict): Challan data including violation details
        
    Returns:
        str: Formatted message
    """
    # Extract data
    license_plate = challan_data.get('license_plate', 'Unknown')
    fine_amount = challan_data.get('fine_amount', 'Unknown')
    
    # Update the message content
    message = (
        f"NOTICE: You are caught with no helmet.\n"
        f"License Plate: {license_plate}\n"
        f"You have to pay a challan of ₹{fine_amount}.\n"
        f"Please clear your dues at the earliest."
    )
    
    # message = (
    #     f"Dear Vehicle Owner,\n\n"
    #     f"We have detected a traffic violation associated with your vehicle "
    #     f"(License Plate: {license_plate}). The violation is 'No Helmet'.\n\n"
    #     f"A challan of ₹{fine_amount} has been issued. Kindly ensure payment is made promptly "
    #     f"to avoid further penalties.\n\n"
    #     f"Thank you for your cooperation.\n\n"
    #     f"You can pay the challan at your nearest traffic police station or online at https://www.ridesafe.com.\n\n"
    #     f"Regards,\nTraffic Enforcement Department"
    # )
    
    return message

def send_challan_whatsapp(phone_number, challan_data):
    """
    Send an e-challan WhatsApp message to the user's phone number.
    
    Args:
        phone_number (str): User's phone number
        challan_data (dict): Challan data including violation details
        
    Returns:
        bool: True if WhatsApp message sent successfully, False otherwise
    """
    client = initialize_twilio_client()
    if not client:
        return False
    
    try:
        # Format the message
        message = format_challan_message(challan_data)
        
        # Format the phone number for WhatsApp (remove + if present)
        if phone_number.startswith('+'):
            phone_number = phone_number[1:]
        
        # Add 'whatsapp:' prefix to the phone number
        whatsapp_number = f"whatsapp:+{phone_number}"
        
        # Send the message
        message = client.messages.create(
            body=message,
            from_=f"whatsapp:{TWILIO_PHONE_NUMBER}",
            to=whatsapp_number
        )
        
        print(f"Challan WhatsApp message sent to {phone_number}. SID: {message.sid}")
        return True
    except Exception as e:
        print(f"Error sending challan WhatsApp message: {e}")
        return False