import os
import time
import immich_api
from bird_identifier_pipeline import BirdIdentifierPipeline
import exif_writer
import logging
import sys

CHECK_INTERVAL = int(os.getenv("CHECK_INTERVAL_SECONDS", "300"))
OVERWRITE_EXIF_DATA = os.getenv("IMMICH_TAGGER_OVERWRITE_TAGS", "False") == True

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)


def identify_and_write_exif(pipeline: BirdIdentifierPipeline, file_path: str):
    """
    Placeholder for your custom identifier & EXIF modification logic.
    Modify file_path in place or perform API-based tagging.
    """
    if not OVERWRITE_EXIF_DATA:
        tags = exif_writer.read_exif_data(file_path)
        if "XMP:HierarchicalSubject" in tags:
            logging.info(f"{file_path} already tagged, skipping")
            return
    exif_writer.write_species_tag(pipeline, file_path)


def run_sync_cycle(pipeline: BirdIdentifierPipeline):
    logging.info("Checking for untagged assets...")
    try:
        untagged_assets = immich_api.get_untagged_assets()
        logging.info(f"Found {len(untagged_assets)} candidate asset(s).")

        updated_ids = []
        for asset in untagged_assets:
            asset_id = asset["id"]
            original_path = asset.get("originalPath")

            # If the library is mounted into the container at the same path:
            if original_path and os.path.exists(original_path):
                identify_and_write_exif(pipeline, original_path)
                updated_ids.append(asset_id)
            else:
                logging.info(f"File path not accessible locally: {original_path}")

        if updated_ids:
            logging.info(f"Refreshing Immich metadata for {len(updated_ids)} assets...")
            immich_api.trigger_immich_metadata_refresh(updated_ids)

    except Exception as e:
        logging.info(f"Error during execution: {e}")


def main():
    ## Verify the environment variables for the Immich API
    immich_api.verify_environment_variables()

    # Initialize once (loads weights into VRAM/RAM)
    pipeline = BirdIdentifierPipeline(
        yolo_model_name="yolov8n.pt",
        birder_model_name="mvit_v2_t_il-all",
        yolo_conf_threshold=0.30,
    )

    logging.info(f"Starting EXIF tagger service. Interval: {CHECK_INTERVAL}s")
    while True:
        run_sync_cycle(pipeline)
        logging.info(f"Cycle done, will wait for {CHECK_INTERVAL}s")
        time.sleep(CHECK_INTERVAL)


if __name__ == "__main__":
    main()
