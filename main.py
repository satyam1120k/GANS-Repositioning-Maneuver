import importlib
import sys


def main():
    print("AI-Based BPPV Maneuver Guider")
    print("-----------------------------")

    # Prompt for affected ear
    while True:
        ear = input("Which ear is affected? (Left/Right): ").strip().lower()
        if ear in ['left', 'right']:
            break
        print("Invalid input. Please enter 'Left' or 'Right'.")

    # Prompt for language
    while True:
        language = input("Which language do you prefer? (English/Hindi): ").strip().lower()
        if language in ['english', 'hindi']:
            break
        print("Invalid input. Please enter 'English' or 'Hindi'.")

    # Determine the script to run
    script_name = f"{ear}_ear_{language}"
    try:
        # Import the module dynamically
        module = importlib.import_module(script_name)
        print(f"Starting BPPV Maneuver for {ear.capitalize()} Ear in {language.capitalize()}...")
        # Assuming each script has a `run` function to execute the main logic
        module.run()
    except ImportError:
        print(f"Error: The script for {ear.capitalize()} Ear in {language.capitalize()} is not found.")
        print("Please ensure the file '{script_name}.py' exists in the same directory.")
        sys.exit(1)
    except AttributeError:
        print(f"Error: The script '{script_name}.py' does not have a 'run' function.")
        sys.exit(1)

if __name__ == "__main__":
    main()

