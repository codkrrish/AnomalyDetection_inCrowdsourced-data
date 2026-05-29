import subprocess
import json

data = {
    "jobFamily": "Software Engineer",
    "level": "senior",
    "baseSalary": 200000,
    "yearsOfExperience": 5,
    "company": "google",
    "region": "US"
}

data_str = json.dumps(data)

cmd = [
    r"D:\Obj_detection\env\python.exe",
    "infer.py",
    "--data",
    data_str,
    "--save"
]

result = subprocess.run(cmd, capture_output=True, text=True)
print("STDOUT:")
print(result.stdout)
print("STDERR:")
print(result.stderr)
