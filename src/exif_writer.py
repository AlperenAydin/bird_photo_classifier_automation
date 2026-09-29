from exiftool import ExifToolHelper
from pathlib import Path
from bird_identifier_pipeline import BirdIdentifierPipeline, BirdDetectionResult

import argparse
from pathlib import Path
import sys
import os

DETECTION_CONFIDENCE = int(os.getenv("BIRD_DETECTION_CONFIDENCE", 80))
SPECIES_IDENTIFICATION_CONFIDENCE = int(os.getenv("BIRD_SPECIES_IDENTIFICATION_CONFIDENCE", 80))

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract and process a file path passed via command-line flags."
    )
    parser.add_argument(
        "-f",
        "--folder",
        dest="folder_path",
        type=Path,
        required=True,
        help="Path to the target folder.",
    )
    return parser.parse_args()

def read_exif_data(file_path: str):
    with ExifToolHelper() as et: 
        tags_list = et.get_tags([file_path], ["IPTC:Keywords", "XMP:HierarchicalSubject"])
    tags = {}
    [ tags.update(entry) for entry in tags_list]
    return tags

def write_exif_data(file_path: str, tags: dict):    
    """Write tags (e.g. copyright, artist, keywords) into EXIF/IPTC/XMP."""
    with ExifToolHelper() as et:
        et.set_tags(
            file_path,
            tags=tags,
            params=["-overwrite_original"]  # Avoids creating .original backups
        )

def write_species_tag(pipeline: BirdIdentifierPipeline, file_path: str):
    
    results = pipeline.process_image(file_path, top_k=3)
    tags = {"IPTC:Keywords": [], 
            "XMP:HierarchicalSubject" : []} 
    for r in results:
        if r.detection_confidence < DETECTION_CONFIDENCE or r.top_species[0][1] < SPECIES_IDENTIFICATION_CONFIDENCE:
            continue
        tags["IPTC:Keywords"].append(f"species:{r.top_species[0][0].lower().replace(' ',"_")}")
        tags["XMP:HierarchicalSubject"].append(f"Nature|Species|{r.top_species[0][0].lower().capitalize()}")
    
    if len(tags["IPTC:Keywords"]) == 0:
        tags = {"IPTC:Keywords": "species:unidentified_locally", 
                "XMP:HierarchicalSubject" : "Nature|Species|unidentified_locally"}
        
    write_exif_data(file_path, tags)
    
def main():
    # Initialize once (loads weights into VRAM/RAM)
    pipeline = BirdIdentifierPipeline(
        yolo_model_name="yolov8n.pt",
        birder_model_name="mvit_v2_t_il-all",
        yolo_conf_threshold=0.30,
    )
    
    # Run inference on target image
    args = parse_args()
    folder_path: Path = args.folder_path.resolve()

    # Validate file existence and type
    if not folder_path.exists():
        print(f"Error: Path '{folder_path}' does not exist.", file=sys.stderr)
        sys.exit(1)

    # Find all .jpg and .jpeg files (case-insensitive)
    valid_extensions = {".jpg", ".jpeg"}
    image_files = sorted(
        [p for p in folder_path.iterdir() if p.suffix.lower() in valid_extensions]
    ) if folder_path.exists() else []
    
    for file_path in image_files:
        tags = read_exif_data(file_path)
        if "XMP:HierarchicalSubject" in tags:
            print(f"{file_path} already tagged, skipping")
            continue
        write_species_tag(pipeline, file_path)


            
if __name__ == "__main__":
    main()