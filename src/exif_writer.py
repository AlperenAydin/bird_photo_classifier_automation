from exiftool import ExifToolHelper
from pathlib import Path
from bird_identifier_pipeline import BirdIdentifierPipeline, BirdDetectionResult

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
    tags = {"IPTC:Keywords": "species:unknown", 
            "XMP:HierarchicalSubject" : "Nature|Species|unknown"} 
    for r in results:
        confidence = r.detection_confidence**0.2 * (r.top_species[0][1]/100)**0.8
        if confidence < 0.7:
            continue   
        tags["IPTC:Keywords"] = f"species:{r.top_species[0][0].lower().replace(' ',"_")}"
        tags["XMP:HierarchicalSubject"] = f"Nature|Species|{r.top_species[0][0]}"
    if tags:
        write_exif_data(file_path, tags)
    
def main():
    # Initialize once (loads weights into VRAM/RAM)
    pipeline = BirdIdentifierPipeline(
        yolo_model_name="yolov8n.pt",
        birder_model_name="mvit_v2_t_il-all",
        yolo_conf_threshold=0.30,
    )
    
    # Run inference on target image
    input_directory = Path("/mnt/c/Users/aydin/Desktop/photos/Lyon_052026/JPG")


    # Find all .jpg and .jpeg files (case-insensitive)
    valid_extensions = {".jpg", ".jpeg"}
    image_files = sorted(
        [p for p in input_directory.iterdir() if p.suffix.lower() in valid_extensions]
    ) if input_directory.exists() else []
    
    for file_path in image_files:
        tags = read_exif_data(file_path)
        if tags["XMP:HierarchicalSubject"]:
            print(f"{file_path} already tagged, skipping")
        write_species_tag(pipeline, file_path)


            
if __name__ == "__main__":
    main()