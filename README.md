# Riddim

This repository contains a simple Streamlit application for exploring audio features from a music file using **librosa**.

## Requirements

- Python 3.12+
- `librosa`
- `streamlit`
- `pandas`

Install the dependencies:

```bash
pip install streamlit librosa pandas
```

## Usage

Place your audio file at `music_files/Metre_Fault_Line.mp3` or upload a file through the UI. Then run:

```bash
streamlit run app.py
```

The app uses a dark theme (configured in `.streamlit/config.toml`) and displays 20 audio features extracted from the chosen file.


import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""   # keep TensorFlow on CPU

from magenta_rt import audio, system

mrt   = system.MagentaRT()
style = mrt.embed_style("ambient techno, 120 bpm")

state, chunks = None, []
for _ in range(round(10 / mrt.config.chunk_length)):     # ~10 s total
    state, chunk = mrt.generate_chunk(state=state, style=style)
    chunks.append(chunk)

track = audio.concatenate(chunks, crossfade_time=mrt.crossfade_length)
track.to_file("ambient_techno.wav")
print("✅ Generated", track.seconds, "sec")
