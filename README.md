# Riddim

This repository contains a simple Streamlit application for exploring audio features from a music file using **librosa**.

## Requirements

- Python 3.12+
- `librosa`
- `streamlit`
- `pandas`
- `magenta` (optional, for music generation)

Install the dependencies:

```bash
pip install streamlit librosa pandas magenta
```

## Usage

Place your audio file at `music_files/Metre_Fault_Line.mp3` or upload a file through the UI. Then run:

```bash
streamlit run app.py
```

The app uses a dark theme (configured in `.streamlit/config.toml`) and displays 20 audio features extracted from the chosen file.
It also provides a **Generate** button that uses Magenta to create a 30-minute track based on the extracted features.
