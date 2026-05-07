# Radiograph Differential Prototype

This small prototype sends each de-identified image in a folder to the OpenAI
Responses API and saves the top 3 differential diagnoses to a CSV file.

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
python3 run_image_differentials.py /path/to/your/3-images --output image_differentials.csv
```

Optional settings:

```bash
python3 run_image_differentials.py /path/to/your/3-images \
  --model gpt-5.4-mini \
  --temperature 0.2 \
  --output image_differentials.csv
```

The CSV contains one row per diagnosis, so three images should produce nine
rows.

## Notes

- Supported image formats: `.jpg`, `.jpeg`, `.png`, `.webp`.
- The default model is `gpt-5.4-mini`.
- The default temperature is `0.2`.
- Generated CSV files and local image/data folders are ignored by git.
