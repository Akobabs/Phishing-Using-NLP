# mini_ids_check.py

# List of known malicious IP addresses
malicious_ips = ["123.45.67.89", "10.10.10.10", "192.0.2.1"]

def is_valid_ip(ip):
    """Check if the input is a valid IP address format without using modules."""
    if not ip:
        return False
    # Split by dots
    parts = ip.split(".")
    # Check if there are exactly 4 parts
    if len(parts) != 4:
        return False
    # Check each part is a number between 0 and 255
    for part in parts:
        # Ensure part is numeric and not empty
        if not part or not part.isdigit():
            return False
        num = int(part)
        if num < 0 or num > 255:
            return False
    return True

while True:
    # Prompt the user to enter an IP address
    user_ip = input("Enter an IP address to check (or 'quit' to exit): ").strip()
    '''
    if user_ip.lower() == "quit":
        print("Exiting program.")
        break
    
    if not user_ip:
        print("Error: No IP address entered.")
        continue
    
    if not is_valid_ip(user_ip):
        print("Error: Invalid IP address format.")
        continue
    
    # Check if the IP is in the malicious list
    if user_ip in malicious_ips:
        print("ALERT: Malicious IP detected!")
    else:
        print("IP is clean.")
        '''
    # Check for empty input
    if not user_ip:
        print("Error: No IP address entered.")
    elif user_ip.lower() == "quit":
        print("Exiting program.")
        break
    elif not is_valid_ip(user_ip):
        print("Error: Invalid IP address format.")
    elif user_ip in malicious_ips:
        print("ALERT: Malicious IP detected!")
    else:
        print("IP is clean.")