#!/usr/bin/env python3
import os
import time
import subprocess
import logging
import sys

# Set up logging to both file and console
logger = logging.getLogger('display_monitor')
logger.setLevel(logging.INFO)

# Create file handler
try:
    file_handler = logging.FileHandler('display_monitor.log')
    file_handler.setLevel(logging.INFO)
    file_formatter = logging.Formatter('%(asctime)s - %(message)s')
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)
except PermissionError:
    print("Warning: Could not create log file. Logging to console only.")

# Create console handler
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO)
console_formatter = logging.Formatter('%(asctime)s - %(message)s')
console_handler.setFormatter(console_formatter)
logger.addHandler(console_handler)

def check_display():
    """Check if display is connected"""
    try:
        display = os.environ.get('DISPLAY')
        return bool(display)
    except Exception as e:
        logger.error(f"Error checking display: {e}")
        return False

def restart_service():
    """Restart the facial recognition service"""
    try:
        subprocess.run(['sudo', 'systemctl', 'restart', 'facial_recognition.service'])
        logger.info("Service restarted successfully")
    except Exception as e:
        logger.error(f"Error restarting service: {e}")

def main():
    logger.info("Display monitor started")
    last_display_state = check_display()
    
    while True:
        current_display_state = check_display()
        
        # If display state changed
        if current_display_state != last_display_state:
            if current_display_state:
                logger.info("Display connected, restarting service")
                restart_service()
            else:
                logger.info("Display disconnected")
            last_display_state = current_display_state
        
        time.sleep(5)  # Check every 5 seconds

if __name__ == "__main__":
    main() 