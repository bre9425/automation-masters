#!/usr/bin/env python3
"""Run approved-list differentials for panoramic radiograph cases."""

import argparse
import base64
import csv
import mimetypes
import re
from dataclasses import dataclass
from pathlib import Path

from openai import OpenAI


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}

APPROVED_DIAGNOSES = [
    "Adenomatoid Odontogenic Tumor",
    "Ameloblastic Fibroma",
    "Ameloblastoma (Conventional/Multicystic)",
    "Ameloblastoma (Unicystic)",
    "Amelogenesis Imperfecta",
    "Aneurysmal Bone Cyst",
    "Antrolith",
    "Brown Tumour/Hyperparathyroidism",
    "Buccal Bifurcation Cyst",
    "Calcified Lymph Nodes",
    "Calcifying Epithelial Odontogenic Tumor",
    "Calcifying Odontogenic Cyst",
    "Cementoblastoma",
    "Cemento-Ossifying Fibroma",
    "Central Giant Cell Granuloma",
    "Central Odontogenic Fibroma",
    "Cherubism",
    "Cleft Lip and Palate",
    "Cleidocranial Dysplasia",
    "Complex Odontoma",
    "Compound Odontoma",
    "Craniofacial Dysostosis (Crouzon Syndrome)",
    "Dentigerous Cyst",
    "Dentin Dysplasia",
    "Dentinogenesis Imperfecta",
    "Desmoplastic Fibroma",
    "Ectodermal Dysplasia",
    "Familial Adenomatous Polyposis (Gardner's Syndrome)",
    "Fibrous Dysplasia",
    "Glandular Odontogenic Cyst",
    "Hemangioma/Vascular Malformation",
    "Hemifacial Hyperplasia",
    "Hemifacial Microsomia",
    "Hyperparathyroidism-Jaw Tumour Syndrome",
    "Hypophosphatasia",
    "Lateral Periodontal Cyst",
    "Malignancy- Carcinoma",
    "Malignancy- Hematolymphoid",
    "Malignancy- Metastatic",
    "Malignancy- Sarcoma",
    "Mandibulofacial Dysostosis (Treacher Collins Syndrome)",
    "Mucopolysaccharidosis",
    "Mucous Retention Pseudocyst",
    "Myositis Ossificans",
    "Nasopalatine Duct Cyst",
    "Neuroma/Neurofibroma/Schwannoma",
    "Nevoid Basal Cell Carcinoma Syndrome (Gorlin-Goltz Syndrome)",
    "Odontogenic Keratocyst",
    "Odontogenic Myxoma",
    "Orthokeratinized Cyst",
    "Osteoblastoma/Osteoid Osteoma",
    "Osteoma",
    "Osteomyelitis/Medication-Related Osteonecrosis of the Jaw/Osteoradionecrosis",
    "Osteopetrosis",
    "Paget's Disease",
    "Periapical Cemento-Osseous Dysplasia",
    "Periapical Inflammatory Disease/Rarefying Osteitis (Including Radicular Cyst and Radicular Granuloma)",
    "Pericoronitis",
    "Periodontal Disease",
    "Phlebolith",
    "Pleomorphic Adenoma",
    "RASopathies (Such as Noonan Syndrome or Neurofibromatosis Type 1)",
    "Regional Odontodysplasia",
    "Residual Cyst",
    "Rhinolith",
    "Rickets/Osteomalacia",
    "Segmental Odontomaxillary Dysplasia",
    "Sialolith",
    "Sickle Cell Anemia/Thalassemia",
    "Simple Bone Cyst",
    "Stafne Bone Defect/Mandibular Lingual Bone Depression",
    "Surgical Ciliated Cyst",
    "Trauma/Fracture",
]

OUTPUT_COLUMNS = [
    "differential_1",
    "reasoning_1",
    "differential_2",
    "reasoning_2",
    "differential_3",
    "reasoning_3",
]

RANKED_LINE_PATTERN = re.compile(
    r"^\s*(?P<rank>[1-3])[\.\)]\s*(?P<diagnosis>.+?)\s+(?:\u2013|\u2014|-)\s+(?P<reasoning>.+?)\s*$"
)


@dataclass(frozen=True)
class CaseInput:
    case_id: str
    location_description: str
    image_path: Path


def approved_diagnosis_list_text() -> str:
    return "\n".join(
        f"{index}. {diagnosis}" for index, diagnosis in enumerate(APPROVED_DIAGNOSES, start=1)
    )


def build_prompt(location_description: str) -> str:
    return f"""You are asked to interpret a panoramic radiograph that contains an oral and maxillofacial pathologic lesion. A panoramic image may or may not be attached. You may be provided with a written radiographic description, a panoramic image, or both.

Using ONLY the information provided in this chat (written radiographic description, panoramic image, or both), provide a ranked differential diagnosis using ONLY diagnoses from the approved list below.

Instructions:
1. Provide between 1 and 3 diagnoses only.
2. Do NOT force 3 diagnoses if fewer are justified by the findings.
3. Rank diagnoses from most likely to least likely.
4. Use only diagnoses from the approved list exactly as written.
5. Do not invent diagnoses or use diagnoses not listed.
6. Base your answer only on the provided findings/image.
7. If uncertainty is high, provide the most plausible limited differential based on available information.
8. Keep the response concise.
9. For each diagnosis, provide brief reasoning based only on the radiographic findings.
10. Do not provide treatment recommendations or unrelated commentary.

Written Radiographic Description:
{location_description}

Approved Diagnosis List:
{approved_diagnosis_list_text()}

Return your answer in exactly this format:

1. [Most likely diagnosis] \u2013 [Brief radiographic reasoning]
2. [Second diagnosis if justified] \u2013 [Brief radiographic reasoning]
3. [Third diagnosis if justified] \u2013 [Brief radiographic reasoning]"""


def image_to_data_url(path: Path) -> str:
    mime_type = mimetypes.guess_type(path.name)[0] or "image/jpeg"
    encoded = base64.b64encode(path.read_bytes()).decode("utf-8")
    return f"data:{mime_type};base64,{encoded}"


def natural_sort_key(path: Path) -> list[object]:
    parts = re.split(r"(\d+)", path.name.lower())
    return [int(part) if part.isdigit() else part for part in parts]


def find_images(image_dir: Path) -> list[Path]:
    """Return supported image files in natural filename order."""
    return sorted(
        (
            path
            for path in image_dir.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        ),
        key=natural_sort_key,
    )


def image_case_number(path: Path) -> str | None:
    """Extract the leading case number from filenames like '1. 22-1391.jpg'."""
    match = re.match(r"^\s*(\d+)\.\s+", path.name)
    if not match:
        return None
    return str(int(match.group(1)))


def read_location_rows(locations_csv: Path) -> list[dict[str, str]]:
    with locations_csv.open(newline="", encoding="utf-8-sig") as csv_file:
        reader = csv.DictReader(csv_file)
        if not reader.fieldnames or len(reader.fieldnames) < 2:
            raise SystemExit("Locations CSV must contain at least two columns.")

        case_column = reader.fieldnames[0]
        location_column = reader.fieldnames[1]
        rows = []

        for row in reader:
            case_id = (row.get(case_column) or "").strip()
            location_description = (row.get(location_column) or "").strip()
            if not case_id and not location_description:
                continue
            if not case_id or not location_description:
                raise SystemExit(
                    f"Locations CSV row is missing a case ID or location description: {row}"
                )
            rows.append(
                {
                    "case_id": case_id,
                    "location_description": location_description,
                }
            )

    if not rows:
        raise SystemExit("No case rows found in locations CSV.")
    return rows


def pair_cases_with_images(case_rows: list[dict[str, str]], image_paths: list[Path]) -> list[CaseInput]:
    images_by_case_number = {}
    incorrectly_named_images = []

    for path in image_paths:
        case_number = image_case_number(path)
        if case_number is None:
            incorrectly_named_images.append(path.name)
            continue
        if case_number in images_by_case_number:
            raise SystemExit(
                f"Multiple images start with case number {case_number}: "
                f"{images_by_case_number[case_number].name}, {path.name}"
            )
        images_by_case_number[case_number] = path

    if incorrectly_named_images:
        raise SystemExit(
            "Image filenames must start with the case number followed by a period, "
            "for example '1. 22-1391.jpg'. These files did not match: "
            + ", ".join(incorrectly_named_images)
        )

    paired_cases = []
    missing_case_ids = []

    for row in case_rows:
        case_id = str(int(row["case_id"])) if row["case_id"].isdigit() else row["case_id"]
        image_path = images_by_case_number.get(case_id)

        if image_path is None:
            missing_case_ids.append(row["case_id"])
            continue

        paired_cases.append(
            CaseInput(
                case_id=row["case_id"],
                location_description=row["location_description"],
                image_path=image_path,
            )
        )

    if not missing_case_ids:
        return paired_cases

    raise SystemExit(
        "Could not match images for case IDs: "
        + ", ".join(missing_case_ids)
        + ". Image names should start with the matching case number, for example '1. 22-1391.jpg'."
    )


def request_differential(
    client: OpenAI,
    model: str,
    temperature: float,
    detail: str,
    case_input: CaseInput,
) -> str:
    response = client.responses.create(
        model=model,
        temperature=temperature,
        input=[
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": build_prompt(case_input.location_description)},
                    {
                        "type": "input_image",
                        "image_url": image_to_data_url(case_input.image_path),
                        "detail": detail,
                    },
                ],
            }
        ],
    )
    return response.output_text.strip()


def clean_diagnosis(value: str) -> str:
    return value.strip().strip("[]").strip("*").strip()


def clean_reasoning(value: str) -> str:
    return value.strip().strip("[]").strip("*").strip()


def parse_ranked_response(response_text: str) -> list[dict[str, str]]:
    diagnoses = []
    for line in response_text.splitlines():
        match = RANKED_LINE_PATTERN.match(line)
        if not match:
            continue
        diagnoses.append(
            {
                "diagnosis": clean_diagnosis(match.group("diagnosis")),
                "reasoning": clean_reasoning(match.group("reasoning")),
            }
        )

    if not diagnoses:
        raise ValueError(f"Could not parse ranked differential response: {response_text}")

    return diagnoses[:3]


def response_to_output_row(response_text: str) -> dict[str, str]:
    parsed_items = parse_ranked_response(response_text)
    row = {column: "" for column in OUTPUT_COLUMNS}

    for index, item in enumerate(parsed_items, start=1):
        row[f"differential_{index}"] = item["diagnosis"]
        row[f"reasoning_{index}"] = item["reasoning"]

    return row


def write_csv(rows: list[dict[str, str]], output_path: Path) -> None:
    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ask OpenAI for approved-list differential diagnoses for each case."
    )
    parser.add_argument("image_dir", type=Path, help="Folder containing de-identified images")
    parser.add_argument(
        "--locations-csv",
        type=Path,
        required=True,
        help="CSV where column 1 is case ID and column 2 is location description",
    )
    parser.add_argument("--output", type=Path, default=Path("image_differentials.csv"))
    parser.add_argument("--model", default="gpt-5.4-mini")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--detail", choices=["low", "high", "auto"], default="high")
    parser.add_argument("--runs-per-case", type=int, default=3)
    parser.add_argument(
        "--limit-cases",
        type=int,
        help="Limit the number of CSV cases processed, useful for low-cost pilot runs",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned case/image pairs without calling the API",
    )
    args = parser.parse_args()

    if not args.image_dir.exists() or not args.image_dir.is_dir():
        raise SystemExit(f"Image folder not found: {args.image_dir}")
    if not args.locations_csv.exists() or not args.locations_csv.is_file():
        raise SystemExit(f"Locations CSV not found: {args.locations_csv}")
    if args.runs_per_case < 1:
        raise SystemExit("--runs-per-case must be at least 1")

    case_rows = read_location_rows(args.locations_csv)
    if args.limit_cases is not None:
        if args.limit_cases < 1:
            raise SystemExit("--limit-cases must be at least 1")
        case_rows = case_rows[: args.limit_cases]

    image_paths = find_images(args.image_dir)
    if not image_paths:
        raise SystemExit(f"No supported images found in: {args.image_dir}")

    cases = pair_cases_with_images(case_rows, image_paths)

    print(f"Prepared {len(cases)} cases x {args.runs_per_case} run(s) per case.")
    for case_input in cases:
        print(f"Case {case_input.case_id}: {case_input.image_path.name}")

    if args.dry_run:
        print("Dry run complete. No API calls were made.")
        return

    client = OpenAI()
    output_rows = []

    for case_input in cases:
        for run_number in range(1, args.runs_per_case + 1):
            print(
                f"Processing case {case_input.case_id}, run {run_number}/{args.runs_per_case}..."
            )
            response_text = request_differential(
                client=client,
                model=args.model,
                temperature=args.temperature,
                detail=args.detail,
                case_input=case_input,
            )
            output_rows.append(response_to_output_row(response_text))

    write_csv(output_rows, args.output)
    print(f"Saved {len(output_rows)} rows to {args.output}")


if __name__ == "__main__":
    main()
