# Radiograph Differential Prototype

This prototype sends each de-identified panoramic radiograph case to the OpenAI
Responses API and saves the ranked differential diagnosis output to a CSV file.

Each case combines:

- one written radiographic description from a CSV
- optionally, one image from the image folder
- the approved diagnosis list prompt in `run_image_differentials.py`

It is intended for approved research workflows where identifiers have already
been removed from images, filenames, metadata, and text descriptions.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
export OPENAI_API_KEY="your_api_key_here"
```

## Run

### Location Description Mode

```bash
python3 run_image_differentials.py /path/to/images \
  --description-source locations \
  --locations-csv "/path/to/Locations.csv" \
  --output image_differentials.csv
```

Optional settings:

```bash
python3 run_image_differentials.py /path/to/images \
  --description-source locations \
  --locations-csv "/path/to/Locations.csv" \
  --model gpt-5.4-mini \
  --temperature 0.2 \
  --detail high \
  --input-modality image-and-text \
  --runs-per-case 3 \
  --output image_differentials.csv
```

The output CSV contains one row per model run and 12 columns:

1. `description_source`
2. `input_modality`
3. `radiologist_id`
4. `case_id`
5. `image_filename`
6. `run_number`
7. `differential_1`
8. `reasoning_1`
9. `differential_2`
10. `reasoning_2`
11. `differential_3`
12. `reasoning_3`

With 20 cases and `--runs-per-case 3`, the output should contain 60 data rows
plus a header row.

### Radiologist Description Mode

Use this mode for the CSV where each row is one radiologist and the repeated
`Radiographic Description` columns correspond to images 1-20 in order.

Radiologist descriptions plus images:

```bash
python3 run_image_differentials.py images \
  --description-source radiologist \
  --input-modality image-and-text \
  --radiologist-csv "Radiologist Interpretations 1.csv" \
  --runs-per-case 3 \
  --output radiologist_image_differentials.csv
```

Radiologist descriptions only, without images:

```bash
python3 run_image_differentials.py images \
  --description-source radiologist \
  --input-modality text-only \
  --radiologist-csv "Radiologist Interpretations 1.csv" \
  --runs-per-case 3 \
  --output radiologist_text_differentials.csv
```

With 5 radiologists, 20 cases, and `--runs-per-case 3`, the output should
contain 300 data rows plus a header row.

## No-cost dry run

Use `--dry-run` to verify case/image pairing without making API calls:

```bash
python3 run_image_differentials.py images \
  --description-source locations \
  --locations-csv "Locations 2(Sheet1).csv" \
  --limit-cases 2 \
  --runs-per-case 1 \
  --dry-run
```

Radiologist description dry run:

```bash
python3 run_image_differentials.py images \
  --description-source radiologist \
  --input-modality image-and-text \
  --radiologist-csv "Radiologist Interpretations 1.csv" \
  --limit-radiologists 1 \
  --limit-cases 2 \
  --runs-per-case 1 \
  --dry-run
```

## Low-cost pilot run

To make only one API call:

```bash
python3 run_image_differentials.py images \
  --description-source locations \
  --locations-csv "Locations 2(Sheet1).csv" \
  --limit-cases 1 \
  --runs-per-case 1 \
  --output pilot_differentials.csv
```

To make only one radiologist-description API call:

```bash
python3 run_image_differentials.py images \
  --description-source radiologist \
  --radiologist-csv "Radiologist Interpretations 1.csv" \
  --input-modality image-and-text \
  --limit-radiologists 1 \
  --limit-cases 1 \
  --runs-per-case 1 \
  --output pilot_radiologist_differentials.csv
```

To make only one text-only radiologist-description API call:

```bash
python3 run_image_differentials.py images \
  --description-source radiologist \
  --input-modality text-only \
  --radiologist-csv "Radiologist Interpretations 1.csv" \
  --limit-radiologists 1 \
  --limit-cases 1 \
  --runs-per-case 1 \
  --output pilot_radiologist_text_differentials.csv
```

## Notes

- Supported image formats: `.jpg`, `.jpeg`, `.png`, `.webp`.
- Images are matched to CSV cases by the leading number in filenames like
  `1. 22-1391.jpg`, `2. 21-6118.jpg`, and so on.
- The default model is `gpt-5.4-mini`.
- The default temperature is `0.2`.
- The default image detail setting is `high`.
- Generated CSV files and local image/data folders are ignored by git.
