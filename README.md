# motiondata-lib

## Run the motion browser

Install Python dependencies with `uv`:

```bash
uv sync
```

On Ubuntu, install the Qt X11 runtime dependency before launching the GUI:

```bash
sudo apt install libxcb-cursor0
```

Then start the browser with a dataset directory:

```bash
uv run python main.py datasets/sonic_retargeted
```

The browser auto-detects these dataset layouts when a directory contains only one kind of motion data:

- standardized `retargeted_npz`
- Sonic CSV
- LAFAN1 CSV
- AMASS `.npy`

You can override detection or frame rate when needed:

```bash
uv run python main.py example_datasets/amass --format amass --fps-override 60
```

Checked clips are always exported as the standardized `.npz` layout used by this project.
