# tinygpt

A small GPT language model in PyTorch, trained from scratch on any plain-text file.
Character-level, decoder-only. Runs on AMD (ROCm) or NVIDIA (CUDA).

## Setup

You need a **ROCm or CUDA build of PyTorch** — plain `pip install torch` gives you
the CPU build and won't use your GPU. Get the correct install command from
https://pytorch.org for your platform. Python 3.12 recommended.

## Use

1. Open `tinygpt.py` and fill in the placeholders:
   - `$YOUR_TEXT_FILE.txt` — your training text file
   - `$MARKER` — text after this string is trimmed off (e.g. license boilerplate); leave as-is if you don't need it
   - `$YOUR_PROMPT_HERE` — the seed text to generate from
2. Set `TRAIN = True` and run `python tinygpt.py`. It trains and saves `model.pt`.
3. Set `TRAIN = False` and run again to load those weights and generate instantly.

## Config

`n_embd`, `n_layer`, `block_size` change the model size and **require retraining**
(`TRAIN = True`). `prompt`, `temperature`, and `top_k` only affect generation and
can be changed instantly with `TRAIN = False`.

## License

MIT
