# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "google-genai>=0.1.1",
#     "pydantic>=2.0.0",
#     "pillow>=10.0.0",
# ]
# ///

import argparse
from functools import lru_cache
import json
import mimetypes
import os
from pathlib import Path
import sys

from google import genai
from google.genai import types
from pydantic import BaseModel, Field

API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL")

class SpeciesIdentification(BaseModel):
    is_organism: bool = Field(
        description="Whether a living organism or identifiable animal/plant is present."
    )
    common_name: str = Field(
        description="Standard common name in English (e.g., 'European Robin')."
    )
    scientific_name: str = Field(
        description="Binomial scientific name (e.g., 'Erithacus rubecula')."
    )
    family: str = Field(description="Taxonomic family (e.g., 'Muscicapidae').")
    confidence: int = Field(
        description="Confidence level of identification: High, Medium, or Low."
    )
    hierarchical_path: str = Field(
        description="Hierarchy pipe-separated (e.g., 'Animalia|Chordata|Aves|Passeriformes|Muscicapidae|Erithacus rubecula')."
    )
    description: str = Field(
        description="Concise 1-2 sentence description including plumage/distinguishing features visible."
    )


class LLMIdentifierPipeline:
    def __init__(self, gemini_api_key: str, gemini_model: str):
        self.client = genai.Client(api_key=gemini_api_key)
        self.model = gemini_model
        self.chat = self.client.chats.create(
                        model=self.model,
                        config=types.GenerateContentConfig(
                            response_mime_type="application/json",
                            response_schema=SpeciesIdentification,
                            temperature=0.1,
                        ),
                    )

        self.prompt = (
            "Identify the primary biological species present in this image. "
            "Provide accurate taxonomic information, common name, and distinguishing features."
        )

    def _generate_message_content(self, image_path: Path):
        mime_type, _ = mimetypes.guess_type(image_path)
        if not mime_type:
            mime_type = "image/jpeg"

        image_bytes = Path(image_path).read_bytes()
        return [
            types.Part.from_bytes(data=image_bytes, mime_type=mime_type),
            self.prompt,
        ]

    def identify_species(self, image_path: Path) -> SpeciesIdentification:
        contents = self._generate_message_content(image_path)
        # Send contents (image parts and prompt) via the chat session
        response = self.chat.send_message(message=contents)
        return SpeciesIdentification.model_validate_json(response.text)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract and process a file path passed via command-line flags."
    )
    parser.add_argument(
        "-f",
        "--file",
        dest="image_path",
        type=Path,
        required=True,
        help="Path to the target file.",
    )
    return parser.parse_args()


def main() -> None:
    # Run inference on target image
    args = parse_args()
    image_path: Path = args.image_path.resolve()

    if not image_path.is_file():
        print(f"Error: File not found at '{image_path}'", file=sys.stderr)
        sys.exit(1)

    llm_identifier_pipeline = LLMIdentifierPipeline(API_KEY, GEMINI_MODEL)
    identification = llm_identifier_pipeline.identify_species(image_path)
    print(identification)
    print(type(identification))

if __name__ == "__main__":
    main()
