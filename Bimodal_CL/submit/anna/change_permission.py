# Change persmissions of the directories located in the same directory as this script
import os
import stat


def change_permissions():
    # Get the directory of this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    print(f"Script directory: {script_dir}")

    # Find all directories in the script directory
    directories = [
        d for d in os.listdir(script_dir) if os.path.isdir(os.path.join(script_dir, d))
    ]
    print(f"Directories found: {directories}")

    for directory in directories:
        directory_path = os.path.join(script_dir, directory)
        print(f"Changing permissions for: {directory_path}")

        current_mode = os.stat(directory_path).st_mode
        print(f"Current mode: {oct(current_mode)}")
        # Define the bits to add for the group
        #    - stat.S_IRGRP: Group read
        #    - stat.S_IWGRP: Group write
        #    - stat.S_IXGRP: Group execute (handles 'X' for a directory)
        #    - stat.S_ISGID: Set-group-ID bit ('s')
        permissions_to_add = stat.S_IRGRP | stat.S_IWGRP | stat.S_IXGRP | stat.S_ISGID
        print(f"Permissions to add: {oct(permissions_to_add)}")

        # Calculate the new mode using bitwise OR (|)
        new_mode = current_mode | permissions_to_add
        print(f"New mode: {oct(new_mode)}")

        # 4. Apply the new mode
        # os.chmod(directory_path, new_mode)


change_permissions()
