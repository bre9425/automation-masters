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

DIFFERENTIAL_OUTPUT_COLUMNS = [
    "description_source",
    "input_modality",
    "radiologist_id",
    "case_id",
    "image_filename",
    "run_number",
    "differential_1",
    "reasoning_1",
    "differential_2",
    "reasoning_2",
    "differential_3",
    "reasoning_3",
]

LLM_DIFFERENTIAL_OUTPUT_COLUMNS = [
    column for column in DIFFERENTIAL_OUTPUT_COLUMNS if column != "radiologist_id"
]

DESCRIPTION_OUTPUT_COLUMNS = [
    "description_source",
    "input_modality",
    "case_id",
    "image_filename",
    "run_number",
    "location_description",
    "generated_description",
]

RANKED_LINE_PATTERN = re.compile(
    r"^\s*(?P<rank>[1-3])[\.\)]\s*(?P<diagnosis>.+?)\s+(?:\u2013|\u2014|-)\s+(?P<reasoning>.+?)\s*$"
)


@dataclass(frozen=True)
class CaseInput:
    case_id: str
    image_path: Path
    written_description: str
    radiologist_id: str = ""
    source_run_number: str = ""


def approved_diagnosis_list_text() -> str:
    return "\n".join(
        f"{index}. {diagnosis}" for index, diagnosis in enumerate(APPROVED_DIAGNOSES, start=1)
    )


def build_prompt(written_description: str) -> str:
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
{written_description}

Approved Diagnosis List:
{approved_diagnosis_list_text()}

Return your answer in exactly this format:

1. [Most likely diagnosis] \u2013 [Brief radiographic reasoning]
2. [Second diagnosis if justified] \u2013 [Brief radiographic reasoning]
3. [Third diagnosis if justified] \u2013 [Brief radiographic reasoning]"""


def build_description_prompt(location_description: str) -> str:
    return f"""You are acting as an oral and maxillofacial radiologist reviewing a panoramic radiograph for descriptive purposes only.

Please provide a written radiographic description of the visible pathologic lesion. Do not provide a clinical impression, diagnosis, differential diagnosis, etiologic explanation, prognosis, or treatment recommendation.

Location of pathologic lesion:
{location_description}

Describe only what is visible on the image using objective radiographic terminology. Include, when applicable:

- Anatomic location of the lesion
- Shape and overall configuration
- Margins/borders, including whether well-defined, corticated, sclerotic, or ill-defined
- Internal radiographic appearance, such as radiolucent, radiopaque, or mixed density
- Unilocular or multilocular appearance, if applicable
- Relationship to adjacent teeth
- Effects on adjacent structures, including displacement, root resorption, cortical expansion, thinning, perforation, or involvement of the mandibular canal, maxillary sinus, nasal floor, or other landmarks

Use a professional radiology-style narrative format.

End the response after the descriptive findings only."""


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
            written_description = (row.get(location_column) or "").strip()
            if not case_id and not written_description:
                continue
            if not case_id or not written_description:
                raise SystemExit(
                    f"Locations CSV row is missing a case ID or location description: {row}"
                )
            rows.append(
                {
                    "case_id": case_id,
                    "written_description": written_description,
                }
            )

    if not rows:
        raise SystemExit("No case rows found in locations CSV.")
    return rows


def read_radiologist_description_rows(
    radiologist_csv: Path,
    limit_cases: int | None = None,
    limit_radiologists: int | None = None,
) -> list[dict[str, str]]:
    with radiologist_csv.open(newline="", encoding="utf-8-sig") as csv_file:
        rows = list(csv.reader(csv_file))

    if len(rows) < 2:
        raise SystemExit("Radiologist CSV must contain a header row and at least one data row.")

    headers = rows[0]
    description_indices = [
        index
        for index, header in enumerate(headers)
        if header.strip().lower() == "radiographic description"
    ]

    if not description_indices:
        raise SystemExit("No 'Radiographic Description' columns found in radiologist CSV.")
    if limit_cases is not None:
        description_indices = description_indices[:limit_cases]

    case_rows = []
    data_rows = rows[1:]
    if limit_radiologists is not None:
        data_rows = data_rows[:limit_radiologists]

    for radiologist_number, row in enumerate(data_rows, start=1):
        if not any(cell.strip() for cell in row):
            continue

        radiologist_id = row[0].strip() if row and row[0].strip() else str(radiologist_number)
        for case_number, description_index in enumerate(description_indices, start=1):
            written_description = row[description_index].strip() if description_index < len(row) else ""
            if not written_description:
                raise SystemExit(
                    f"Missing radiographic description for radiologist {radiologist_id}, "
                    f"case {case_number}."
                )
            case_rows.append(
                {
                    "case_id": str(case_number),
                    "written_description": written_description,
                    "radiologist_id": radiologist_id,
                }
            )

    if not case_rows:
        raise SystemExit("No radiologist description rows found in radiologist CSV.")
    return case_rows


def read_llm_description_rows(
    llm_description_csv: Path,
    limit_cases: int | None = None,
) -> list[dict[str, str]]:
    with llm_description_csv.open(newline="", encoding="utf-8-sig") as csv_file:
        reader = csv.DictReader(csv_file)
        required_columns = {"case_id", "generated_description"}
        if not reader.fieldnames or not required_columns.issubset(reader.fieldnames):
            raise SystemExit(
                "LLM description CSV must contain case_id and generated_description columns."
            )

        rows = []
        included_case_ids = []
        for row in reader:
            case_id = (row.get("case_id") or "").strip()
            written_description = (row.get("generated_description") or "").strip()
            if not case_id and not written_description:
                continue
            if not case_id or not written_description:
                raise SystemExit(
                    f"LLM description CSV row is missing a case ID or generated description: {row}"
                )

            normalized_case_id = str(int(case_id)) if case_id.isdigit() else case_id
            if limit_cases is not None:
                if normalized_case_id not in included_case_ids:
                    if len(included_case_ids) >= limit_cases:
                        continue
                    included_case_ids.append(normalized_case_id)
                elif normalized_case_id not in included_case_ids:
                    continue

            rows.append(
                {
                    "case_id": case_id,
                    "written_description": written_description,
                    "source_run_number": (row.get("run_number") or "").strip(),
                }
            )

    if not rows:
        raise SystemExit("No generated description rows found in LLM description CSV.")
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
                image_path=image_path,
                written_description=row["written_description"],
                radiologist_id=row.get("radiologist_id", ""),
                source_run_number=row.get("source_run_number", ""),
            )
        )

    if not missing_case_ids:
        return paired_cases

    raise SystemExit(
        "Could not match images for case IDs: "
        + ", ".join(missing_case_ids)
                        + ". Image names should start with the matching case number, for example '1. 22-1391.jpg'."
    )


def request_openai_response(
    client: OpenAI,
    model: str,
    temperature: float,
    prompt: str,
    image_path: Path | None,
) -> str:
    content = [{"type": "input_text", "text": prompt}]
    if image_path is not None:
        content.append(
            {
                "type": "input_image",
                "image_url": image_to_data_url(image_path),
            }
        )

    response = client.responses.create(
        model=model,
        temperature=temperature,
        input=[
            {
                "role": "user",
                "content": content,
            }
        ],
    )
    return response.output_text.strip()


def request_differential(
    client: OpenAI,
    model: str,
    temperature: float,
    input_modality: str,
    case_input: CaseInput,
) -> str:
    image_path = case_input.image_path if input_modality == "image-and-text" else None
    return request_openai_response(
        client=client,
        model=model,
        temperature=temperature,
        prompt=build_prompt(case_input.written_description),
        image_path=image_path,
    )


def request_description(
    client: OpenAI,
    model: str,
    temperature: float,
    case_input: CaseInput,
) -> str:
    return request_openai_response(
        client=client,
        model=model,
        temperature=temperature,
        prompt=build_description_prompt(case_input.written_description),
        image_path=case_input.image_path,
    )


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


def response_to_output_row(
    response_text: str,
    description_source: str,
    input_modality: str,
    case_input: CaseInput,
    run_number: int | str,
) -> dict[str, str]:
    parsed_items = parse_ranked_response(response_text)
    row = {column: "" for column in DIFFERENTIAL_OUTPUT_COLUMNS}
    row.update(
        {
            "description_source": description_source,
            "input_modality": input_modality,
            "radiologist_id": case_input.radiologist_id,
            "case_id": case_input.case_id,
            "image_filename": case_input.image_path.name,
            "run_number": str(run_number),
        }
    )

    for index, item in enumerate(parsed_items, start=1):
        row[f"differential_{index}"] = item["diagnosis"]
        row[f"reasoning_{index}"] = item["reasoning"]

    return row


def response_to_description_row(
    response_text: str,
    description_source: str,
    input_modality: str,
    case_input: CaseInput,
    run_number: int | str,
) -> dict[str, str]:
    return {
        "description_source": description_source,
        "input_modality": input_modality,
        "case_id": case_input.case_id,
        "image_filename": case_input.image_path.name,
        "run_number": str(run_number),
        "location_description": case_input.written_description,
        "generated_description": response_text.strip(),
    }


def write_csv(rows: list[dict[str, str]], output_path: Path, fieldnames: list[str]) -> None:
    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def output_input_modality(args: argparse.Namespace) -> str:
    if args.task == "description" and args.input_modality == "image-and-text":
        return "image-and-text2"
    return args.input_modality


def output_description_source(args: argparse.Namespace) -> str:
    if args.description_source == "llm":
        return "LLM"
    return args.description_source


def load_case_rows(args: argparse.Namespace) -> list[dict[str, str]]:
    if args.description_source == "locations":
        if args.locations_csv is None:
            raise SystemExit("--locations-csv is required when --description-source locations")
        if not args.locations_csv.exists() or not args.locations_csv.is_file():
            raise SystemExit(f"Locations CSV not found: {args.locations_csv}")

        case_rows = read_location_rows(args.locations_csv)
        if args.limit_cases is not None:
            case_rows = case_rows[: args.limit_cases]
        return case_rows

    if args.description_source == "llm":
        if args.llm_description_csv is None:
            raise SystemExit("--llm-description-csv is required when --description-source llm")
        if not args.llm_description_csv.exists() or not args.llm_description_csv.is_file():
            raise SystemExit(f"LLM description CSV not found: {args.llm_description_csv}")
        return read_llm_description_rows(
            args.llm_description_csv,
            limit_cases=args.limit_cases,
        )

    if args.radiologist_csv is None:
        raise SystemExit("--radiologist-csv is required when --description-source radiologist")
    if not args.radiologist_csv.exists() or not args.radiologist_csv.is_file():
        raise SystemExit(f"Radiologist CSV not found: {args.radiologist_csv}")

    return read_radiologist_description_rows(
        args.radiologist_csv,
        limit_cases=args.limit_cases,
        limit_radiologists=args.limit_radiologists,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ask OpenAI for differential diagnoses or radiographic descriptions for each case."
    )
    parser.add_argument("image_dir", type=Path, help="Folder containing de-identified images")
    parser.add_argument(
        "--task",
        choices=["differential", "description"],
        default="differential",
        help="Ask for approved-list differentials or descriptive radiographic findings",
    )
    parser.add_argument(
        "--description-source",
        choices=["locations", "radiologist", "llm"],
        default="locations",
        help="Use locations, radiologist descriptions, or generated LLM descriptions",
    )
    parser.add_argument(
        "--locations-csv",
        type=Path,
        help="CSV where column 1 is case ID and column 2 is location description",
    )
    parser.add_argument(
        "--radiologist-csv",
        type=Path,
        help="CSV containing repeated Radiographic Description columns for each image",
    )
    parser.add_argument(
        "--llm-description-csv",
        type=Path,
        help="CSV produced by --task description, containing generated_description rows",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--model", default="gpt-5.4-2026-03-05")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument(
        "--input-modality",
        choices=["image-and-text", "text-only"],
        default="image-and-text",
        help="Attach each image plus text, or send only the written description",
    )
    parser.add_argument("--runs-per-case", type=int)
    parser.add_argument(
        "--limit-cases",
        type=int,
        help="Limit the number of cases/images processed, useful for low-cost pilot runs",
    )
    parser.add_argument(
        "--limit-radiologists",
        type=int,
        help="Limit the number of radiologist rows processed in radiologist mode",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned case/image pairs without calling the API",
    )
    args = parser.parse_args()

    if not args.image_dir.exists() or not args.image_dir.is_dir():
        raise SystemExit(f"Image folder not found: {args.image_dir}")
    if args.runs_per_case is None:
        args.runs_per_case = 1 if args.description_source == "llm" else 3
    if args.runs_per_case < 1:
        raise SystemExit("--runs-per-case must be at least 1")
    if args.limit_cases is not None:
        if args.limit_cases < 1:
            raise SystemExit("--limit-cases must be at least 1")
    if args.limit_radiologists is not None and args.limit_radiologists < 1:
        raise SystemExit("--limit-radiologists must be at least 1")
    if args.output is None:
        if args.task == "description":
            args.output = Path("image_descriptions.csv")
        elif args.description_source == "llm":
            args.output = Path("llm_description_differentials.csv")
        else:
            args.output = Path("image_differentials.csv")
    if args.task == "description":
        if args.description_source != "locations":
            raise SystemExit("--task description requires --description-source locations")
        if args.input_modality != "image-and-text":
            raise SystemExit("--task description requires --input-modality image-and-text")
    if args.description_source == "llm":
        if args.task != "differential":
            raise SystemExit("--description-source llm requires --task differential")
        if args.input_modality != "text-only":
            raise SystemExit("--description-source llm requires --input-modality text-only")

    case_rows = load_case_rows(args)

    image_paths = find_images(args.image_dir)
    if not image_paths:
        raise SystemExit(f"No supported images found in: {args.image_dir}")

    cases = pair_cases_with_images(case_rows, image_paths)
    saved_input_modality = output_input_modality(args)
    saved_description_source = output_description_source(args)
    input_label = "generated descriptions" if args.description_source == "llm" else "cases"

    print(
        f"Prepared {len(cases)} {input_label} x {args.runs_per_case} run(s) per input "
        f"with {saved_input_modality} input."
    )
    for case_input in cases:
        if case_input.radiologist_id:
            print(
                f"Radiologist {case_input.radiologist_id}, case {case_input.case_id}: "
                f"{case_input.image_path.name}"
            )
        elif case_input.source_run_number:
            print(
                f"LLM description case {case_input.case_id}, "
                f"source run {case_input.source_run_number}: {case_input.image_path.name}"
            )
        else:
            print(f"Case {case_input.case_id}: {case_input.image_path.name}")

    if args.dry_run:
        print("Dry run complete. No API calls were made.")
        return

    client = OpenAI()
    output_rows = []

    for case_input in cases:
        for run_number in range(1, args.runs_per_case + 1):
            label = f"case {case_input.case_id}"
            if case_input.radiologist_id:
                label = f"radiologist {case_input.radiologist_id}, {label}"
            elif case_input.source_run_number:
                label = f"LLM description {label}, source run {case_input.source_run_number}"
            output_run_number = case_input.source_run_number or run_number
            print(f"Processing {label}, run {run_number}/{args.runs_per_case}...")
            if args.task == "description":
                response_text = request_description(
                    client=client,
                    model=args.model,
                    temperature=args.temperature,
                    case_input=case_input,
                )
                output_rows.append(
                    response_to_description_row(
                        response_text=response_text,
                        description_source=saved_description_source,
                        input_modality=saved_input_modality,
                        case_input=case_input,
                        run_number=output_run_number,
                    )
                )
            else:
                response_text = request_differential(
                    client=client,
                    model=args.model,
                    temperature=args.temperature,
                    input_modality=args.input_modality,
                    case_input=case_input,
                )
                output_rows.append(
                    response_to_output_row(
                        response_text=response_text,
                        description_source=saved_description_source,
                        input_modality=saved_input_modality,
                        case_input=case_input,
                        run_number=output_run_number,
                    )
                )

    if args.task == "description":
        output_columns = DESCRIPTION_OUTPUT_COLUMNS
    elif args.description_source == "llm":
        output_columns = LLM_DIFFERENTIAL_OUTPUT_COLUMNS
    else:
        output_columns = DIFFERENTIAL_OUTPUT_COLUMNS
    write_csv(output_rows, args.output, output_columns)
    print(f"Saved {len(output_rows)} rows to {args.output}")


if __name__ == "__main__":
    main()
