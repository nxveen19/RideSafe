import os
from pymongo import MongoClient
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# MongoDB connection string
MONGO_URI = os.getenv("MONGO_URI")

def get_database():
    """
    Create a connection to MongoDB and return the database object.
    
    Returns:
        pymongo.database.Database: MongoDB database object
    """
    try:
        client = MongoClient(MONGO_URI)
        db = client["ridesafe"]
        return db
    except Exception as e:
        print(f"Error connecting to MongoDB: {e}")
        return None

def get_user_collection():
    """
    Get the users collection from the database.
    
    Returns:
        pymongo.collection.Collection: Users collection or None if database connection fails
    """
    db = get_database()
    if db is not None:
        return db.users
    return None

def get_challan_collection():
    """
    Get the challans collection from the database.
    
    Returns:
        pymongo.collection.Collection: Challans collection or None if database connection fails
    """
    db = get_database()
    if db is not None:
        return db.challans
    return None

def add_user(user_data):
    """
    Add a new user to the database.
    
    Args:
        user_data (dict): User data to add
        
    Returns:
        str: ID of the added user or None if operation fails
    """
    try:
        users = get_user_collection()
        if users is not None:
            result = users.insert_one(user_data)
            return str(result.inserted_id)
        return None
    except Exception as e:
        print(f"Error adding user: {e}")
        return None

def add_challan(challan_data):
    """
    Add a new challan to the database.
    
    Args:
        challan_data (dict): Challan data to add
        
    Returns:
        str: ID of the added challan or None if operation fails
    """
    try:
        challans = get_challan_collection()
        if challans is not None:
            result = challans.insert_one(challan_data)
            return str(result.inserted_id)
        return None
    except Exception as e:
        print(f"Error adding challan: {e}")
        return None

def get_user_by_license_plate(license_plate):
    """
    Get a user by license plate number.
    
    Args:
        license_plate (str): License plate number to search for
        
    Returns:
        dict: User data or None if not found
    """
    try:
        users = get_user_collection()
        if users is not None:
            return users.find_one({"license_plate": license_plate})
        return None
    except Exception as e:
        print(f"Error getting user by license plate: {e}")
        return None

def get_challans_by_user_id(user_id):
    """
    Get all challans for a user.
    
    Args:
        user_id (str): User ID to search for
        
    Returns:
        list: List of challans or empty list if none found
    """
    try:
        challans = get_challan_collection()
        if challans is not None:
            return list(challans.find({"user_id": user_id}))
        return []
    except Exception as e:
        print(f"Error getting challans by user ID: {e}")
        return [] 