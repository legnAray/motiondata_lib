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
