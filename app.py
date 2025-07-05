import os
import streamlit as st
import pandas as pd
import librosa

st.set_page_config(page_title="riddim.exe · ⋆.˚✮🎧✮˚.⋆ · ", page_icon="🎵")

# Hide Streamlit style elements for a cleaner look
hide_style = """
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .reportview-container .main .block-container{padding-top:2rem;}
    </style>
"""

st.markdown(hide_style, unsafe_allow_html=True)

st.title("🎵 riddim.exe · ⋆.˚✮🎧✮˚.⋆ · ")

st.write("Upload an audio file to analyze or use the default sample.")
file = st.file_uploader("Upload MP3/WAV", type=["mp3", "wav"])

if file is not None:
    path = file
    name = file.name
else:
    default_path = os.path.join("music_files", "Metre_Fault_Line.mp3")
    name = os.path.basename(default_path)
    if not os.path.isfile(default_path):
        st.error(f"Default file '{default_path}' not found.")
        st.stop()
    path = default_path
    st.info(f"Using default file: {name}")


@st.cache_data(show_spinner=True)
def extract_features(path):
    y, sr = librosa.load(path)
    duration = librosa.get_duration(y=y, sr=sr)
    tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
    beat_count = len(beats)

    zcr = librosa.feature.zero_crossing_rate(y)
    rms = librosa.feature.rms(y=y)
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    flatness = librosa.feature.spectral_flatness(y=y)
    contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=10)

    features = {
        "Duration (s)": duration,
        "Tempo (BPM)": tempo,
        "Beat count": beat_count,
        "Avg RMS": float(rms.mean()),
        "Avg Zero Crossing Rate": float(zcr.mean()),
        "Avg Spectral Centroid": float(centroid.mean()),
        "Avg Spectral Bandwidth": float(bandwidth.mean()),
        "Avg Spectral Rolloff": float(rolloff.mean()),
        "Avg Spectral Flatness": float(flatness.mean()),
        "Avg Spectral Contrast": float(contrast.mean()),
    }

    for i in range(10):
        features[f"MFCC {i+1}"] = float(mfcc[i].mean())
    return features


if st.button("Analyze"):
    with st.spinner("Extracting features..."):
        data = extract_features(path)
    df = pd.DataFrame(data.items(), columns=["Feature", "Value"])
    st.subheader(f"Analysis for: {name}")

    FEATURE_DESCRIPTIONS = {
        "Duration (s)": "Length of the track in seconds.",
        "Tempo (BPM)": "Speed of the music in beats per minute.",
        "Beat count": "Number of beats detected.",
        "Avg RMS": "Average loudness of the audio.",
        "Avg Zero Crossing Rate": "How often the signal changes sign; relates to noisiness.",
        "Avg Spectral Centroid": "Represents brightness; higher means more treble.",
        "Avg Spectral Bandwidth": "How spread out the frequencies are.",
        "Avg Spectral Rolloff": "Frequency below which most energy lies.",
        "Avg Spectral Flatness": "How noise-like versus tone-like the sound is.",
        "Avg Spectral Contrast": "Difference between peaks and valleys in the spectrum.",
    }
    for i in range(10):
        FEATURE_DESCRIPTIONS[f"MFCC {i+1}"] = "A coefficient describing the timbre."

    df["Value"] = df["Value"].apply(lambda v: f"{v:.2f}")
    df["Feature"] = df["Feature"].apply(
        lambda f: f'<span title="{FEATURE_DESCRIPTIONS.get(f, '')}">{f}</span>'
    )
    st.write(
        df.to_html(escape=False, index=False),
        unsafe_allow_html=True,
    )
