import subprocess

def install_requirements():
    try:
        subprocess.run(["pip", "install", "-r", "requirements.txt"], check=True)
        print("Requirements installed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Error installing requirements: {e}")

def run_script(script_name):
    try:
        subprocess.run(["python", script_name], check=True)
        print(f"Done running: {script_name}")
    except subprocess.CalledProcessError as e:
        print(f"Error running {script_name}: {e}")

if __name__ == "__main__":
    install_requirements()
    
    scripts = ["store_bank_policy.py", "store_client_policy.py", "Compliance.py"]
    
    for script in scripts:
        run_script(script)
