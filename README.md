# OpenAI Prompt Automation

This project automates repeated OpenAI API prompts for a research study of
panoramic radiograph differential diagnosis. It reads case descriptions from
CSV files, optionally attaches the matching panoramic image, asks OpenAI to
return a ranked differential diagnosis from an approved diagnosis list, and
saves the parsed responses to CSV.

The script supports three study workflows:

1. **Location description + image**: 20 cases, each run 3 times.
2. **Radiologist description + image**: 5 radiologists x 20 cases, each run 3 times.
3. **Radiologist description only**: 5 radiologists x 20 cases, each run 3 times.

All OpenAI outputs are written with the same columns:

```text
description_source
input_modality
radiologist_id
case_id
image_filename
run_number
differential_1
reasoning_1
differential_2
reasoning_2
differential_3
reasoning_3
```

## Expected Files

Images should be in the `images/` folder and named with the case number first:

```text
1. 22-1391.jpg
2. 21-6118.jpg
...
20. 22-4214.jpg
```

The location CSV should have 20 rows with the first column as the case number
and the second column as the location description.

The radiologist CSV should have 5 rows, one per radiologist. Each repeated
`Radiographic Description` column is treated as the description for cases 1-20
in order.

## Setup

```bash
cd /Users/breanna/Documents/Codex/2026-05-07/openai-prompt-automation
source .venv/bin/activate
```

If dependencies are not installed yet:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
```

Store the OpenAI API key in a local `.env` file:

```text
OPENAI_API_KEY=your_api_key_here
```

Then load it before running API calls:

```bash
set -a
source .env
set +a
```

## One-Iteration Trials

These commands each make **one OpenAI API call**.

### Location Description + Image

Output file: `trial_location_image_differentials.csv`

```bash
.venv/bin/python run_image_differentials.py images \
  --description-source locations \
  --input-modality image-and-text \
  --locations-csv "Locations 2(Sheet1).csv" \
  --limit-cases 1 \
  --runs-per-case 1 \
  --output trial_location_image_differentials.csv
```

### Radiologist Description + Image

Output file: `trial_radiologist_image_differentials.csv`

```bash
.venv/bin/python run_image_differentials.py images \
  --description-source radiologist \
  --input-modality image-and-text \
  --radiologist-csv "Radiologist Interpretations 1.csv" \
  --limit-radiologists 1 \
  --limit-cases 1 \
  --runs-per-case 1 \
  --output trial_radiologist_image_differentials.csv
```

### Radiologist Description Only

Output file: `trial_radiologist_text_differentials.csv`

```bash
.venv/bin/python run_image_differentials.py images \
  --description-source radiologist \
  --input-modality text-only \
  --radiologist-csv "Radiologist Interpretations 1.csv" \
  --limit-radiologists 1 \
  --limit-cases 1 \
  --runs-per-case 1 \
  --output trial_radiologist_text_differentials.csv
```

## Full Study Runs

### Location Description + Image

Runs 20 prompts 3 times each for 60 output rows.

Output file: `location_image_differentials.csv`

```bash
.venv/bin/python run_image_differentials.py images \
  --description-source locations \
  --input-modality image-and-text \
  --locations-csv "Locations 2(Sheet1).csv" \
  --runs-per-case 3 \
  --output location_image_differentials.csv
```

### Radiologist Description + Image

Runs 5 radiologists x 20 cases x 3 repeats for 300 output rows.

Output file: `radiologist_image_differentials.csv`

```bash
.venv/bin/python run_image_differentials.py images \
  --description-source radiologist \
  --input-modality image-and-text \
  --radiologist-csv "Radiologist Interpretations 1.csv" \
  --runs-per-case 3 \
  --output radiologist_image_differentials.csv
```

### Radiologist Description Only

Runs 5 radiologists x 20 cases x 3 repeats for 300 output rows.

Output file: `radiologist_text_differentials.csv`

```bash
.venv/bin/python run_image_differentials.py images \
  --description-source radiologist \
  --input-modality text-only \
  --radiologist-csv "Radiologist Interpretations 1.csv" \
  --runs-per-case 3 \
  --output radiologist_text_differentials.csv
```

## No-Cost Dry Runs

Add `--dry-run` to any command to verify the case/image mapping without making
OpenAI API calls.

Example:

```bash
.venv/bin/python run_image_differentials.py images \
  --description-source radiologist \
  --input-modality text-only \
  --radiologist-csv "Radiologist Interpretations 1.csv" \
  --limit-radiologists 1 \
  --limit-cases 2 \
  --runs-per-case 1 \
  --dry-run
```

## Notes

- Supported image formats: `.jpg`, `.jpeg`, `.png`, `.webp`.
- Images are matched to CSV cases by the leading number in the filename.
- The default model is `gpt-5.4-mini`.
- The default temperature is `0.2`.
- The default image detail setting is `high`.
- Generated output CSVs, local input CSVs, images, `.env`, and virtual
  environments are ignored by git.
