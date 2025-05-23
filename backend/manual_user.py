import os
import logging
from datetime import datetime
from dotenv import load_dotenv

from db_connection import get_database, add_user, get_user_by_license_plate

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("manual_user.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def add_manual_user():
    """
    Add a user manually to the database.
    """
    print("\n===== MANUAL USER ENTRY =====")
    
    # Get user details
    name = input("Enter user name: ").strip()
    license_plate = input("Enter license plate (e.g., GA11B7547): ").strip().upper()
    
    # Check if user already exists
    existing_user = get_user_by_license_plate(license_plate)
    if existing_user:
        print(f"User with license plate {license_plate} already exists.")
        update_user = input("Would you like to update this user? (y/n): ").strip().lower()
        
        if update_user != 'y':
            print("Operation cancelled.")
            return
    
    # Get additional user details
    phone_number = input("Enter phone number (with country code, e.g., +919082223606): ").strip()
    email = input("Enter email: ").strip()
    address = input("Enter address: ").strip()
    
    # Create user data
    user_data = {
        "name": name,
        "license_plate": license_plate,
        "phone_number": phone_number,
        "email": email,
        "address": address,
        "created_at": datetime.now().isoformat()
    }
    
    # Add user to database
    user_id = add_user(user_data)
    
    if user_id:
        print(f"User added successfully with ID: {user_id}")
    else:
        print("Failed to add user.")

def list_all_users():
    """
    List all users in the database.
    """
    db = get_database()
    if db is None:
        print("Failed to connect to database.")
        return
    
    users = list(db.users.find())
    
    if not users:
        print("No users found in the database.")
        return
    
    print("\n===== ALL USERS =====")
    for user in users:
        print(f"ID: {user['_id']}")
        print(f"Name: {user['name']}")
        print(f"License Plate: {user['license_plate']}")
        print(f"Phone: {user.get('phone_number', 'N/A')}")
        print(f"Email: {user.get('email', 'N/A')}")
        print(f"Address: {user.get('address', 'N/A')}")
        print("-" * 30)

def search_user():
    """
    Search for a user by license plate.
    """
    print("\n===== SEARCH USER =====")
    
    license_plate = input("Enter license plate to search: ").strip().upper()
    
    user = get_user_by_license_plate(license_plate)
    
    if user:
        print("\n===== USER FOUND =====")
        print(f"ID: {user['_id']}")
        print(f"Name: {user['name']}")
        print(f"License Plate: {user['license_plate']}")
        print(f"Phone: {user.get('phone_number', 'N/A')}")
        print(f"Email: {user.get('email', 'N/A')}")
        print(f"Address: {user.get('address', 'N/A')}")
    else:
        print(f"No user found with license plate {license_plate}")

def main_menu():
    """
    Display the main menu and handle user input.
    """
    while True:
        print("\n===== MANUAL USER MANAGEMENT =====")
        print("1. Add a new user")
        print("2. List all users")
        print("3. Search for a user")
        print("4. Exit")
        
        choice = input("Enter your choice (1-4): ").strip()
        
        if choice == '1':
            add_manual_user()
        elif choice == '2':
            list_all_users()
        elif choice == '3':
            search_user()
        elif choice == '4':
            print("Exiting. Goodbye!")
            break
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main_menu() 