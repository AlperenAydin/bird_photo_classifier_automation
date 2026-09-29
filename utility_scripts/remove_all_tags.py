import os 
import requests 

IMMICH_URL = os.getenv("IMMICH_URL", "http://immich-server:2283/api").rstrip("/")
IMMICH_API_KEY = os.getenv("IMMICH_API_KEY", "")

HEADERS = {
    "x-api-key": IMMICH_API_KEY,
    "Accept": "application/json",
}


def get_all_tags():
    """
    Fetches all tags from immich server
    """
    # Get tags and find the relevant tag
    response = requests.get(f"{IMMICH_URL}/tags", headers=HEADERS)
    return response.json()


def delete_tag(tagId: str): 
    url = f"{IMMICH_URL}/tags/{tagId}"
    requests.delete(url, headers=HEADERS)
    
    
def main():
    for tag in get_all_tags():
        delete_tag(tag["id"])
        

if __name__ == "__main__":
    main()