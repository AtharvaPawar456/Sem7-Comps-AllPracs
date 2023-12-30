import requests

url = "https://github.com/AtharvaPawar456/Sem7-Comps-AllPracs/raw/main/ML%20-%20Machine%20Learning/exp%20-%204/main.py"

response = requests.get(url)

if response.status_code == 200:
    with open("main.py", "wb") as file:
        file.write(response.content)
    print("main.py downloaded successfully.")
else:
    print(f"Failed to download main.py. Status code: {response.status_code}")
