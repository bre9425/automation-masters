#!/usr/bin/env python3
"""Run a differential diagnosis prompt over every image in a folder."""

import argparse
import base64
import csv
import json
import mimetypes
from pathlib import Path

from openai import OpenAI


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}

PROMPT = """
For this de-identified radiographic image, list the top 3 differential
diagnoses based on the visible imaging findings.

Return only the requested structured output. Do not include patient identifiers.
If the image quality is insufficient, still provide your best assessment and
explain the limitation.
"""

OUTPUT_SCHEMA = {
    "type": "json_schema",
    "name": "radiograph_differential",
    "strict": True,
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "image_quality": {"type": "string"},
            "overall_limitations": {"type": "string"},
            "differential": {
                "type": "array",
                "minItems": 3,
                "maxItems": 3,
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "rank": {"type": "integer"},
                        "diagnosis": {"type": "string"},
                        "confidence": {
                            "type": "number",
                            "minimum": 0,
                            "maximum": 1,
                        },
                        "supporting_findings": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                        "reasoning_summary": {"type": "string"},
                    },
                    "required": [
                        "rank",
                        "diagnosis",
                        "confidence",
                        "supporting_findings",
                        "reasoning_summary",
                    ],
                },
            },
        },
        "required": ["image_quality", "overall_limitations", "differential"],
    },
}


def image_to_data_url(path: Path) -> str:
    mime_type = mimetypes.guess_type(path.name)[0] or "image/jpeg"
    encoded = base64.b64encode(path.read_bytes()).decode("utf-8")
    return f"data:{mime_type};base64,{encoded}"


def find_images(image_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in image_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def request_differential(client: OpenAI, model: str, temperature: float, image_path: Path) -> dict:
    response = client.responses.create(
        model=model,
        temperature=temperature,
        input=[
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": PROMPT.strip()},
                    {
                        "type": "input_image",
                        "image_url": image_to_data_url(image_path),
                        "detail": "high",
                    },
                ],
            }
        ],
        text={"format": OUTPUT_SCHEMA},
    )
    return json.loads(response.output_text)


def write_csv(rows: list[dict], output_path: Path) -> None:
    fieldnames = [
        "image_filename",
        "model",
        "temperature",
        "image_quality",
        "overall_limitations",
        "rank",
        "diagnosis",
        "confidence",
        "supporting_findings",
        "reasoning_summary",
    ]

    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ask OpenAI for the top 3 differential diagnoses for each image."
    )
    parser.add_argument("image_dir", type=Path, help="Folder containing de-identified images")
    parser.add_argument("--output", type=Path, default=Path("image_differentials.csv"))
    parser.add_argument("--model", default="gpt-5.4-mini")
    parser.add_argument("--temperature", type=float, default=0.2)
    args = parser.parse_args()

    if not args.image_dir.exists() or not args.image_dir.is_dir():
        raise SystemExit(f"Image folder not found: {args.image_dir}")

    image_paths = find_images(args.image_dir)
    if not image_paths:
        raise SystemExit(f"No supported images found in: {args.image_dir}")

    client = OpenAI()
    rows = []

    for image_path in image_paths:
        print(f"Processing {image_path.name}...")
        result = request_differential(
            client=client,
            model=args.model,
            temperature=args.temperature,
            image_path=image_path,
        )

        for item in result["differential"]:
            rows.append(
                {
                    "image_filename": image_path.name,
                    "model": args.model,
                    "temperature": args.temperature,
                    "image_quality": result["image_quality"],
                    "overall_limitations": result["overall_limitations"],
                    "rank": item["rank"],
                    "diagnosis": item["diagnosis"],
                    "confidence": item["confidence"],
                    "supporting_findings": " | ".join(item["supporting_findings"]),
                    "reasoning_summary": item["reasoning_summary"],
                }
            )

    write_csv(rows, args.output)
    print(f"Saved {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()
