import requests
import os 

IMMICH_URL = os.getenv("IMMICH_URL", "http://immich-server:2283/api").rstrip("/")
IMMICH_API_KEY = os.getenv("IMMICH_API_KEY", "")

IMMICH_BATCH_SIZE = os.getenv("IMMICH_BATCH_SIZE", "500",)

HEADERS = {
    "x-api-key": IMMICH_API_KEY,
    "Accept": "application/json",
}

def verify_environment_variables():
    if not IMMICH_URL:
        raise ValueError("IMMICH_URL environment variable is required.")
    if not IMMICH_API_KEY:
        raise ValueError("IMMICH_API_KEY environment variable is required.")
    

def get_untagged_assets():
    """
    Fetches assets from Immich. 
    You can filter by metadata/tags or check if specific EXIF fields are present.
    """
    url = f"{IMMICH_URL}/search/metadata"
    # Query Immich for images (customize payload to filter specific tags/albums if needed)
    payload = {
        "size": int(IMMICH_BATCH_SIZE),
        "filter": {
            "type": {
                "eq": "IMAGE"
            },
            "hasTags": {
                "eq": False
                },
        },
    }
    
    response = requests.post(url, headers=HEADERS, json=payload)
    response.raise_for_status()
    assets = response.json().get("assets", {}).get("items", [])
            
    return assets

def trigger_immich_metadata_refresh(asset_ids: list[str]):
    """Notifies Immich to re-read updated EXIF metadata from the disk."""
    if not asset_ids:
        return
    url = f"{IMMICH_URL}/assets/jobs"
    payload = {
        "assetIds": asset_ids,
        "name": "refresh-metadata"
    }
    resp = requests.post(url, headers=HEADERS, json=payload)
    resp.raise_for_status()
    
def main():
    assets = get_untagged_assets()
    print(f"{assets[0]}")
    print(f"{assets[0].get("exifInfo")}")
    print(f"{assets[0].get("tags")}")
    print(len(assets))
    
            
if __name__ == "__main__":
    main()