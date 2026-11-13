"""
Main launcher.
Mode 1: Victim Detection (existing system - unchanged)
Mode 2: Security Monitoring (new - lives entirely in mode2_security/)
"""

import subprocess
import sys
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def main():
    print("=" * 50)
    print(" SELECT MODE")
    print(" 1. Victim Detection")
    print(" 2. Security Monitoring")
    print("=" * 50)

    choice = input("Enter 1 or 2: ").strip()

    if choice == "1":
        # Update this filename only if your Mode 1 entry point is different
        script = os.path.join(BASE_DIR, "rescue_system.py")
        subprocess.run([sys.executable, script])

    elif choice == "2":
        script = os.path.join(BASE_DIR, "mode2_security", "security_monitor.py")
        subprocess.run([sys.executable, script])

    else:
        print("Invalid choice. Please enter 1 or 2.")


if __name__ == "__main__":
    main()