# Radiograph Differential Prototype

This prototype sends each de-identified panoramic radiograph case to the OpenAI
Responses API and saves the ranked differential diagnosis output to a CSV file.

Each case combines:

- one image from the image folder
- one location description from the locations CSV
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

```bash
python3 run_image_differentials.py /path/to/images \
  --locations-csv "/path/to/Locations.csv" \
  --output image_differentials.csv
```

Optional settings:

```bash
python3 run_image_differentials.py /path/to/images \
  --locations-csv "/path/to/Locations.csv" \
  --model gpt-5.4-mini \
  --temperature 0.2 \
  --detail high \
  --runs-per-case 3 \
  --output image_differentials.csv
```

The output CSV contains one row per model run and six columns:

1. `differential_1`
2. `reasoning_1`
3. `differential_2`
4. `reasoning_2`
5. `differential_3`
6. `reasoning_3`

With 20 cases and `--runs-per-case 3`, the output should contain 60 data rows
plus a header row.

## No-cost dry run

Use `--dry-run` to verify case/image pairing without making API calls:

```bash
python3 run_image_differentials.py images \
  --locations-csv "/Users/breanna/Downloads/Locations 2(Sheet1).csv" \
  --limit-cases 2 \
  --runs-per-case 1 \
  --dry-run
```

## Low-cost pilot run

To make only one API call:

```bash
python3 run_image_differentials.py images \
  --locations-csv "/Users/breanna/Downloads/Locations 2(Sheet1).csv" \
  --limit-cases 1 \
  --runs-per-case 1 \
  --output pilot_differentials.csv
```

## Notes

- Supported image formats: `.jpg`, `.jpeg`, `.png`, `.webp`.
- Images are matched to CSV cases by the leading number in filenames like
  `1. 22-1391.jpg`, `2. 21-6118.jpg`, and so on.
- The default model is `gpt-5.4-mini`.
- The default temperature is `0.2`.
- The default image detail setting is `high`.
- Generated CSV files and local image/data folders are ignored by git.
