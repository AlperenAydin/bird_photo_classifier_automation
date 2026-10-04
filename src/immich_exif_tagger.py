import os
import time
import immich_api
from bird_identifier_pipeline import BirdIdentifierPipeline
from llm_identifier_pipeline import LLMIdentifierPipeline
import exif_writer
import logging
import sys

CHECK_INTERVAL = int(os.getenv("CHECK_INTERVAL_SECONDS", "300"))
OVERWRITE_EXIF_DATA = os.getenv("IMMICH_TAGGER_OVERWRITE_TAGS", "False") == "True"
USE_REMOTE_IDENTIFICATION = os.getenv("USE_REMOTE_IDENTIFICATION", "False") == "True"

DETECTION_CONFIDENCE = int(os.getenv("BIRD_DETECTION_CONFIDENCE", 80))
SPECIES_IDENTIFICATION_CONFIDENCE = int(
    os.getenv("BIRD_SPECIES_IDENTIFICATION_CONFIDENCE", 80)
)


API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(filename)s:%(lineno)d - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)


def process_images_locally(pipeline: BirdIdentifierPipeline, file_path: str):
    if not OVERWRITE_EXIF_DATA:
        tags = exif_writer.read_exif_data(file_path)
        if "XMP:HierarchicalSubject" in tags:
            logging.info(f"{file_path} already tagged, skipping")
            return

    results = pipeline.process_image(file_path, top_k=3)
    tags = {"IPTC:Keywords": [], "XMP:HierarchicalSubject": []}

    for r in results:
        if r.detection_confidence * 100 < DETECTION_CONFIDENCE:
            continue
        if r.top_species[0][1] < SPECIES_IDENTIFICATION_CONFIDENCE:
            continue

        species_name = r.top_species[0][0].lower()
        tags["IPTC:Keywords"].append(f"species:{species_name.replace(' ',"_")}")
        tags["XMP:HierarchicalSubject"].append(
            f"Nature|Species|{species_name.capitalize()}"
        )

    if len(tags["IPTC:Keywords"]) == 0:
        tags = {
            "IPTC:Keywords": "species:unidentified_locally",
            "XMP:HierarchicalSubject": "Nature|Species|unidentified_locally",
        }

    exif_writer.write_exif_data(file_path, tags)


def process_images_remotely(pipeline: LLMIdentifierPipeline, file_path: str):
    if not OVERWRITE_EXIF_DATA:
        tags = exif_writer.read_exif_data(file_path)
        if "XMP:HierarchicalSubject" in tags:
            logging.info(f"{file_path} already tagged, skipping")
            return

    results = pipeline.identify_species(file_path)
    print(results)
    # If there is no creature, we should tag it as such.
    if not results.is_organism:
        exif_writer.write_exif_data(
            file_path,
            tags={
                "IPTC:Keywords": "species:no_animal",
                "XMP:HierarchicalSubject": "Nature|Species|No animal",
            },
        )
        return
    # If the confidence level is low, mark it as unidentified
    if int(results.confidence) < SPECIES_IDENTIFICATION_CONFIDENCE:
        exif_writer.write_exif_data(
            file_path,
            tags={
                "IPTC:Keywords": "species:unidentified_remotely",
                "XMP:HierarchicalSubject": "Nature|Species|unidentified_remotely",
            },
        )
        return

    species_name = results.common_name.lower()
    tags = {
        "IPTC:Keywords": f"species:{species_name.replace(' ',"_")}",
        "XMP:HierarchicalSubject": f"Nature|Species|{species_name.capitalize()}",
    }
    exif_writer.write_exif_data(file_path, tags)


def run_local_cycle(pipeline: BirdIdentifierPipeline):
    logging.info("Checking for untagged assets...")
    try:
        untagged_assets = immich_api.get_untagged_assets()
        logging.info(f"Found {len(untagged_assets)} untagged candidate asset(s).")

        updated_ids = []
        for asset in untagged_assets:
            asset_id = asset["id"]
            original_path = asset.get("originalPath")

            # If the library is mounted into the container at the same path:
            if original_path and os.path.exists(original_path):
                process_images_locally(pipeline, original_path)
                updated_ids.append(asset_id)
            else:
                logging.info(f"File path not accessible locally: {original_path}")

        if updated_ids:
            logging.info(f"Refreshing Immich metadata for {len(updated_ids)} assets...")
            immich_api.trigger_immich_metadata_refresh(updated_ids)
    except Exception as e:
        logging.error(f"Error during execution: {e}", exc_info=True)


def run_remote_cycle():
    pipeline = LLMIdentifierPipeline(API_KEY, GEMINI_MODEL)
    logging.info("Checking for unidentified assets...")
    try:
        unidentified_tagged_assets = immich_api.get_unidentified_tagged_assets()
        logging.info(
            f"Found {len(unidentified_tagged_assets)} previously unidentified candidate asset(s)."
        )
        max_assets_to_identify = min(10, len(unidentified_tagged_assets))

        updated_ids = []
        for asset in unidentified_tagged_assets[:max_assets_to_identify]:
            asset_id = asset["id"]
            original_path = asset.get("originalPath")

            # If the library is mounted into the container at the same path:
            if original_path and os.path.exists(original_path):
                process_images_remotely(pipeline, original_path)
                updated_ids.append(asset_id)
            else:
                logging.info(f"File path not accessible locally: {original_path}")

        if updated_ids:
            logging.info(
                f"Refreshing Immich metadata for {len(updated_ids)} previously unidentified assets..."
            )
            immich_api.trigger_immich_metadata_refresh(updated_ids)
    except Exception as e:
        logging.error(f"Error during execution: {e}", exc_info=True)


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
        run_local_cycle(pipeline)
        if USE_REMOTE_IDENTIFICATION:
            run_remote_cycle()
        logging.info(f"Cycle done, will wait for {CHECK_INTERVAL}s")
        time.sleep(CHECK_INTERVAL)


if __name__ == "__main__":
    main()
