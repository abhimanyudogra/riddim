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


# Riddim – Installation Notes (WSL + Magenta-Realtime)

Below is every hurdle hit during setup and the exact fix that worked.  
Copy-paste as-is into your project’s `README.md`.

---

## 1  Trouble → Fix quick-reference

| # | Trouble / Error message | Fix that worked |
|---|------------------------|-----------------|
| 1 | **Old Magenta deps won’t compile on Windows** (`llvmlite`, `numba`, MSVC errors) | Move the whole build into **WSL 2 + Ubuntu 22.04**. |
| 2 | `conda : command not found` in PowerShell | Add `C:\ProgramData\miniconda3\;…\Scripts\;…\Library\bin\` to **User PATH** → `conda init powershell`. |
| 3 | `tensorflow-text-nightly` wheel missing on Windows | Run all Python installs inside WSL where Linux wheels exist. |
| 4 | `HCS_E_SERVICE_NOT_AVAILABLE` during `wsl --install` | Reboot → enable **VirtualMachinePlatform** & **WSL** features → rerun `wsl --install`. |
| 5 | `fasttext` C++ build fails (`uint64_t` not declared) | Pre-install **binary fasttext 0.9.2** from *conda-forge* before `pip`. |
| 6 | `t5x-nightly` missing for Python 3.11 | Create conda env with **Python 3.10** (`magenta_rt310`). |
| 7 | `ModuleNotFoundError: jax …` after `pip --no-deps` | Manually install `jax[cuda12_pip]`, `flax`, `t5x-nightly`, `seqio-nightly`, TF nightlies, `typing_extensions`. |
| 8 | `google.auth.exceptions.DefaultCredentialsError` | `sudo apt install google-cloud-cli` → `gcloud auth application-default login`. |
| 9 | `RuntimeError: Physical devices cannot be modified after being initialized` (TF) | Hide GPU from TF: `os.environ["CUDA_VISIBLE_DEVICES"] = ""` **before** importing `magenta_rt` (or set `TF_FORCE_GPU_ALLOW_GROWTH=true`). |
| 10 | PyCharm CE lacks WSL debugging | Switched to **VS Code + Remote-WSL** (`code .` inside Ubuntu). |
| 11 | Couldn’t find `\\wsl$\…` path | Just run `code .` from the desired folder in the WSL shell. |
| 12 | `AttributeError: system has no attribute embed_style` | Call **`mrt.embed_style()`** on the `MagentaRT` instance, not the module. |
| 13 | Large first-run GCS download | Accept once; checkpoint cached under `~/.cache/magenta_rt`. |

---

## 2  One-time WSL setup

```bash
# inside Ubuntu-22.04 on WSL 2
sudo apt update && sudo apt install -y curl apt-transport-https ca-certificates gnupg build-essential git wget

# Miniforge (Conda)
wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
bash Miniforge3-Linux-x86_64.sh -b
source ~/miniforge3/bin/activate

# Create env (Python 3.10) + binary fasttext
conda create -n magenta_rt310 python=3.10 -y
conda activate magenta_rt310
conda install -c conda-forge fasttext=0.9.2 -y

# JAX CUDA-12 wheel
pip install "jax[cuda12_pip]" \
  -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

# Magenta-Realtime with GPU support
pip install "git+https://github.com/magenta/magenta-realtime#egg=magenta_rt[gpu]"

# ADC token for public GCS checkpoints
curl -fsSL https://packages.cloud.google.com/apt/doc/apt-key.gpg | \
  sudo gpg --dearmor -o /usr/share/keyrings/cloud.google.gpg
echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] \
  http://packages.cloud.google.com/apt cloud-sdk main" | \
  sudo tee /etc/apt/sources.list.d/google-cloud-sdk.list
sudo apt update && sudo apt install -y google-cloud-cli
gcloud auth application-default login
