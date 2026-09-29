import requests
import os

IMMICH_URL = os.getenv("IMMICH_URL", "http://immich-server:2283/api").rstrip("/")
IMMICH_API_KEY = os.getenv("IMMICH_API_KEY", "")

IMMICH_BATCH_SIZE = os.getenv(
    "IMMICH_BATCH_SIZE",
    "500",
)

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
    Fetches untagged assets from Immich.
    """
    url = f"{IMMICH_URL}/search/metadata"
    # Query Immich for images (customize payload to filter specific tags/albums if needed)
    payload = {
        "size": int(IMMICH_BATCH_SIZE),
        "filter": {
            "type": {"eq": "IMAGE"},
            "hasTags": {"eq": False},
        },
    }

    response = requests.post(url, headers=HEADERS, json=payload)
    response.raise_for_status()
    assets = response.json().get("assets", {}).get("items", [])

    return assets


def get_unidentified_tagged_assets():
    """
    Fetches assets with the "Nature|Species|unidentified_locally" from Immich.
    You can filter by metadata/tags or check if specific EXIF fields are present.
    """
    # Get tags and find the relevant tag
    response = requests.get(f"{IMMICH_URL}/tags", headers=HEADERS)
    tags = response.json()
    tags = list(
        filter(lambda t: t["value"] == "Nature/Species/unidentified_locally", tags)
    )
    if len(tags) == 0:
        return []
    unidentified_tag = tags[0]

    # Query Immich for images (customize payload to filter specific tags/albums if needed)
    url = f"{IMMICH_URL}/search/metadata"
    payload = {
        "size": int(IMMICH_BATCH_SIZE),
        "filter": {
            "type": {"eq": "IMAGE"},
            "tagIds": {"any": [unidentified_tag["id"]]},
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
    payload = {"assetIds": asset_ids, "name": "refresh-metadata"}
    resp = requests.post(url, headers=HEADERS, json=payload)
    resp.raise_for_status()


def main():
    assets = get_untagged_assets()
    print(len(assets))

    tag = get_unidentified_tagged_assets()
    print(len(tag))


if __name__ == "__main__":
    main()
