"""
main.py
Main entry point script demonstrating end-to-end usage across modular imports.
"""

from registry import registry
from process import process_material
from exceptions import PiezoValidationError, PiezoNotFoundError


def main():
    print("--- 1. Registering Materials ---")
    q_freq = process_material("Quartz", (100.0, 500.0), create_new=True)
    pzt_freq = process_material("PZT-5H", (1000.0, 5000.0), create_new=True)

    print(f"Quartz generated frequency : {q_freq:.2f} kHz")
    print(f"PZT-5H generated frequency: {pzt_freq:.2f} kHz")
    print(f"Registered materials       : {registry.list_materials()}\n")

    print("--- 2. Retrieving Existing Material ---")
    fetched_freq = process_material("Quartz", create_new=False)
    print(f"Retrieved Quartz frequency : {fetched_freq:.2f} kHz\n")

    print("--- 3. Testing Exception Handling ---")
    try:
        process_material("BadRange", (500.0, 1000.0), create_new=True)
    except PiezoValidationError as e:
        print(f"Validation failure caught: {e}")

    try:
        process_material("UnknownMat", create_new=False)
    except PiezoNotFoundError as e:
        print(f"Lookup failure caught    : {e}")


if __name__ == "__main__":
    main()