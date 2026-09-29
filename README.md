# textsnap

> **Snap any image, screenshot, or webpage into plaintext. No GPU. No cloud. One command.**

![textsnap demo](demo-textsnap.jpg)

![Python](https://img.shields.io/badge/python-3.9+-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Platforms](https://img.shields.io/badge/platforms-macOS%20%7C%20Linux%20%7C%20Windows-lightgrey)

```
textsnap screenshot.png
```

That's it. You get a `.txt` next to your shell, recognized on your CPU, from a screenshot, a photo, an image URL, or even a webpage.

---

## Why textsnap

- ⚡ **Runs on CPU.** A 0.9B PaddleOCR-VL-1.5 vision-language model, quantized to ONNX (8-bit vision encoder, 4-bit decoder), parses full pages on a plain laptop. No CUDA. No M-series-only tricks. Plain old cores, pinned to your physical-core count.
- 🖼 **Images, screenshots, URLs, webpages.** Point it at a local file, a direct image URL, or a full article URL — it isolates the main content and OCRs the most prominent image. Or OCR straight from your clipboard with no argument at all — and get the text put *back* on your clipboard, ready to paste.
- 📴 **Offline after first run.** ~1.1 GB of ONNX downloads once to your cache and stays there. No API keys. No quotas. Your images never leave your machine.
- 🎒 **Portable.** Drop the model files next to the script and the whole folder becomes a self-contained, copy-anywhere tool — no install, no download, no flags.
- 📚 **Batch mode.** Pass several images (or a glob) and the model loads once for all of them.
- 🪶 **One file.** The whole tool is a single Python module. Dependencies install themselves on first run if missing.
- 📝 **Markdown or plaintext.** Default output is the model's native markdown (tables, headings, structure preserved). Add `--plaintext` to flatten it.

---

## Quickstart

```
# Install
pip install textsnap

# Snap something
textsnap screenshot.png
textsnap https://example.com/article --plaintext
textsnap photo.jpg -o ~/notes/receipt.txt
textsnap scans/*.png -o ~/notes/scans/     # batch: model loads once
```

The first run downloads the model (~1.1 GB). Every run after is offline.

---

## What it handles

| Source            | Example                                  |
| ----------------- | ---------------------------------------- |
| Clipboard         | `textsnap` *(no argument)*               |
| Local image file  | `textsnap path/to/img.png`               |
| Direct image URL  | `textsnap https://example.com/x.png`     |
| Webpage URL       | `textsnap https://example.com/article`   |
| Several of the above | `textsnap a.png b.jpg https://example.com/x.png` |

Local files cover anything Pillow can decode: `.png`, `.jpg`, `.jpeg`, `.webp`, `.bmp`, `.gif`, `.tiff`, and friends. For webpage URLs, textsnap uses readability to isolate the main content, then picks the most prominent image on the page and OCRs that.

---

## Clipboard in, clipboard out

Run `textsnap` with **no argument** and it reads the image currently on your clipboard. The recognized text is then copied **straight back to the clipboard**, so a screenshot-to-text round trip is just: snap → `textsnap` → paste.

The `.txt` file is still written as well (and its path still printed to stdout), so nothing about scripting changes — the clipboard copy is a pure convenience layered on top.

Clipboard-out uses your platform's native tool — `pbcopy` (macOS), `clip` (Windows), or `wl-copy` / `xclip` / `xsel` (Linux) — so it needs no extra Python package. If none of those is installed, textsnap simply skips the clipboard copy; the `.txt` file is always there regardless. (Run with `-v` to see whether the copy succeeded.)

---

## Portable mode

By default textsnap downloads its model files to an OS cache directory (`~/.cache/textsnap/`). But if it finds the model files **sitting next to the script**, it uses those directly — no download, no `--model-dir` flag, no setup at all.

"Next to the script" means a layout like:

```
textsnap/
├── textsnap.py
├── onnx/
│   ├── vision_encoder_q8.onnx
│   ├── decoder_q4.onnx
│   ├── embedding.onnx
│   └── embedding.onnx.data
└── tokenizer.json
```

Drop those files in, and you can copy the entire `textsnap/` folder to any machine — a USB stick, an air-gapped box, a fresh laptop — and run it immediately, fully offline, with zero install steps.

Model-directory resolution order:

1. `--model-dir DIR` — if you pass it explicitly, it always wins.
2. **Portable** — model files found next to the script.
3. **OS cache** — `~/.cache/textsnap/`, downloading on first run if needed.

> Like `--model-dir`, portable-mode files are **not** SHA-256 verified — files you placed there yourself are trusted by definition. Integrity verification applies to files textsnap *downloads*. See [Security](#security).

---

## Install

```
pip install textsnap
```

Installs two equivalent commands on your `PATH`: **`textsnap`** (canonical) and **`ocr`** (alias, for when the name slips your mind).

To install from a local source checkout instead:

```
pip install .
```

For a reproducible install with exact pinned dependency versions:

```
pip install -r requirements-lock.txt
pip install .
```

> **Clipboard note.** Reading images *from* the clipboard relies on Pillow's `ImageGrab`; on Linux you may need `xclip` or `wl-clipboard` installed. Writing recognized text *back* to the clipboard uses `pbcopy` / `clip` / `wl-copy` / `xclip` / `xsel`. macOS and Windows work out of the box.

---

## Usage

```
# Clipboard (no argument) — text is also copied back to the clipboard
textsnap

# Local image file
textsnap path/to/screenshot.png

# Direct image URL
textsnap "https://example.com/diagram.png"

# Webpage — OCRs the most prominent image on the page
textsnap "https://example.com/article"

# Flatten the model's markdown to plain text
textsnap input.png --plaintext

# Custom output path
textsnap input.png -o ./out/extracted.txt

# Raise the token cap for very dense pages
textsnap dense-page.png --max-tokens 4096

# Trade accuracy for speed by shrinking the image budget
textsnap input.png --max-pixels 250000

# Use a local model directory instead of downloading
textsnap input.png --model-dir ~/models/paddleocr-vl

# Batch: many inputs, one model load. Globs work even where the shell
# doesn't expand them (Windows).
textsnap page1.png page2.png "scans/*.jpg" -o ./ocr-out/

# Use the smaller 4-bit vision encoder (see "Vision encoder variants")
textsnap input.png --vision q4
```

---

## Output

Plaintext, UTF-8. Default location is `./textsnaps/` (created if missing) under the current working directory; override with `-o`. The filename is derived from the image filename stem (`receipt_ocr.txt`), or from the webpage slug for URL inputs.

textsnap is quiet by default, Unix-style: the **only** thing printed to stdout is the path to the file it wrote, so it composes cleanly —

```
OUT=$(textsnap receipt.png)   # capture the path
textsnap receipt.png | xargs cat   # print the recognized text
```

### Several inputs

With more than one input, the model is loaded once and each image is processed in turn. stdout gets one path per line, printed as each result is written, so `textsnap *.png | while read f; do ...; done` starts working before the batch finishes.

- `-o` names a **directory** (created if missing); without it, results go to `./textsnaps/` as usual.
- Inputs that share a name (`a/page.png`, `b/page.png`) don't overwrite each other: the second becomes `page_ocr_2.txt`.
- An input that fails (missing file, bad URL, unreadable image) is reported on stderr as `[textsnap] FAILED <input>: <reason>` and skipped; the rest still run, and the exit status is 1 at the end. A model that fails to load stops the whole batch.
- Clipboard mode is unchanged: it's what you get with no arguments.

When the input is the clipboard, the recognized text is *also* placed on the clipboard — see [Clipboard in, clipboard out](#clipboard-in-clipboard-out).

Pass `-v` to send progress diagnostics (input type, image size, decode speed, token counts) to **stderr**; stdout stays just the path either way.

Default file output is the model's **native markdown** — it preserves tables, headings, and document structure:

```
# Quarterly Report

| Region | Revenue |
| ------ | ------- |
| EMEA   | $1.2M   |
| APAC   | $0.9M   |
```

With **`--plaintext`**, markdown is flattened to bare text:

```
Quarterly Report

Region Revenue
EMEA $1.2M
APAC $0.9M
```

---

## Flags

| Flag                  | Description                                                          |
| --------------------- | -------------------------------------------------------------------- |
| `-o`, `--output`      | Output `.txt` path, or a directory (required to be a directory with several inputs). Default: `./textsnaps/<name>_ocr.txt`. |
| `-v`, `--verbose`     | Print progress diagnostics to stderr. Off by default.                |
| `--plaintext`         | Flatten the model's native markdown to plain text.                   |
| `--model-dir`         | Use ONNX/config files from this directory. Overrides portable mode and the OS cache. |
| `--max-tokens`        | Cap generated tokens. Default `2048`. Raise it for very dense pages. |
| `--max-pixels`        | Image pixel budget fed to the vision encoder. Default is the model's maximum. Lower trades accuracy for speed; too low makes the model hallucinate. The image is only ever shrunk, never enlarged. |
| `--vision`            | Vision-encoder variant: `q8` (default) or `q4`. See [Vision encoder variants](#vision-encoder-variants). |
| `--no-verify`         | Skip SHA-256 verification of downloaded model files (not advised).   |
| `--generate-checksums`| Download the pinned model files, write a fresh manifest, and exit.   |

Environment variables:

| Variable                  | Effect |
| ------------------------- | ------ |
| `TEXTSNAP_DECODE_THREADS` | Decoder intra-op thread count, for tuning CPU decode on a specific machine. Default: a sensible value from your physical core count. |
| `TEXTSNAP_VISION`         | Default for `--vision`. |
| `TEXTSNAP_EMBED=ort`      | Run token embeddings through an ONNX Runtime session instead of the memory-mapped table (for debugging; output is identical). |
| `TEXTSNAP_LOGITS_SLICE=0` | Use the decoder graph exactly as shipped instead of the last-position patch (for debugging; output is identical). |

---

## Vision encoder variants

Two builds of the vision encoder are available, selectable with `--vision`:

| Variant | Download | Notes |
| ------- | -------- | ----- |
| `q8` *(default)* | 443 MB | 8-bit dynamic quantization. The encoder is compute-bound, and integer GEMMs make it faster: ~1.3x over q4 in testing, with fewer misreads. |
| `q4`    | 231 MB | 4-bit weight-only. Smallest download. |

Only the variant you use is downloaded, and every variant is SHA-256 pinned. The variants produce slightly different image features, so **switching changes output**. To compare them on images like yours:

```
python bench_vision.py samples/*.png                  # q4 vs q8
python bench_vision.py samples/*.png --truth truth/   # plus CER vs truth/<stem>.txt
```

It reports encoder time, total time, and character error rate against the q4 output, and against ground truth if given. Ground truth is the better yardstick, since neither variant is full precision.

---

## Security

textsnap auto-downloads ~1.1 GB of model weights from the Hugging Face Hub on first run, so it treats those files as untrusted until proven otherwise:

- **Pinned model revision.** Downloads are pinned to a specific repo revision, so a moved or retagged `main` can't silently swap the weights.
- **SHA-256 verification.** Every downloaded file — including external-data sidecars such as `embedding.onnx.data`, which holds the actual embedding weights — is hashed and checked against known-good digests before it's loaded. A mismatch aborts the run with a clear error rather than executing unverified weights. Digests live in [`model_checksums.sha256`](model_checksums.sha256) and are also embedded in the script; entries in the manifest override the embedded ones, and files the manifest doesn't mention still fall back to the embedded digest, so verification works whether you install from source or from a wheel.
- **Pinned dependencies.** [`requirements-lock.txt`](requirements-lock.txt) pins exact dependency versions for reproducible installs; the file documents how to add per-wheel `--hash` entries with `pip-compile --generate-hashes` for full supply-chain pinning.

Verification applies to files textsnap **downloads**. Model files you supply yourself — via `--model-dir` or [portable mode](#portable-mode) — are trusted as-is and not re-hashed; you are responsible for their provenance.

Regenerate the checksum manifest after a deliberate model-revision bump:

```
textsnap --generate-checksums
```

To bypass verification (for local experimentation with a modified model), pass `--no-verify`.

---

## How it works

1. **Load.** From the clipboard, a local file, a direct image URL, or — for a webpage URL — the most prominent image inside the page's main content (readability + a prominence heuristic).
2. **Preprocess.** The image is run through PaddleOCR-VL's Qwen2-VL-style smart-resize and patchify, producing the pixel-value tensor and grid the vision encoder expects. Smart-resize bounds the image to the model's pixel budget (tunable with `--max-pixels`) and snaps it to the patch grid — textsnap does not pre-shrink beyond that, since starving the encoder of resolution makes the model hallucinate rather than degrade gracefully.
3. **Recognize.** A vision encoder (q8 by default) and an autoregressive decoder (q4) run on CPU through ONNX Runtime; the decoder's KV cache is bound via IOBinding to avoid copying it each step. Token embeddings are read straight from the memory-mapped fp32 table rather than through a separate session, so only the rows actually used are paged in. The decoder graph is patched in memory so that prefill projects only the **last** prompt position onto the ~103k-token vocabulary, instead of all ~1,300 (the only row that is ever read); the file on disk is untouched. Greedy decode, guarded against runaway repetition by an n-gram block (it refuses to re-emit an n-gram it has already produced) plus a loop detector that trims any cycle that slips through.
4. **Format.** Native markdown by default; `--plaintext` reduces it to bare text.

No image is sent anywhere. No state is kept between runs except the cached model.

---

## Model & cache

The PaddleOCR-VL-1.5 ONNX components are downloaded on first run to `~/.cache/textsnap/`:

- `onnx/vision_encoder_q8.onnx` — vision encoder + spatial-merge projector (or `vision_encoder_q4.onnx` with `--vision q4`)
- `onnx/decoder_q4.onnx` — autoregressive decoder
- `onnx/embedding.onnx` + `onnx/embedding.onnx.data` — token-embedding graph and its fp32 table (no q4 variant exists)
- `tokenizer.json`, `config.json`

Together ~1.1 GB. To use your own copy, either point `--model-dir` at a directory containing the same `onnx/` files plus `tokenizer.json` and `config.json`, or place those files next to the script for [portable mode](#portable-mode).

---

## Notes & limits

- **First run is the slow one** — it downloads ~1.1 GB. After that, textsnap is fully offline.
- **Model loading is paid once per run**, so OCRing many images is much faster in one batch invocation than in a shell loop calling textsnap per file.
- **CPU decode is sequential.** Dense, full-page documents take longer than a short screenshot. textsnap pins thread counts to your physical cores and prints a live tokens/sec readout so a slow run is visibly alive, not hung.
- **`--max-tokens` caps the output.** Very dense pages can hit the default 2048-token cap and truncate; raise it if the tail of a page is missing.
- **`--max-pixels` is a speed/accuracy dial.** Lowering it speeds up the vision encoder but feeds the model a coarser image; set it too low and recognition quality drops sharply. The default (the model's full budget) is the safe choice.
- **Webpage inputs OCR one image** — the most prominent one in the main content, not the whole rendered page.
- **Greedy decoding** can occasionally loop on repetitive layouts; an n-gram block prevents most loops outright and a detector trims any that remain.

---

## License

MIT for this project — see [LICENSE](LICENSE).

The model is **PaddleOCR-VL-1.5**, distributed under Apache-2.0 by PaddlePaddle; textsnap downloads the ONNX export from [`kouhxp/PaddleOCR-VL-1.5-ONNX`](https://huggingface.co/kouhxp/PaddleOCR-VL-1.5-ONNX), an unmodified, pinned mirror of selected files from [`onnx-community/PaddleOCR-VL-1.5-ONNX`](https://huggingface.co/onnx-community/PaddleOCR-VL-1.5-ONNX). See the [original model card](https://huggingface.co/PaddlePaddle/PaddleOCR-VL-1.5) for model terms. Powered by [onnxruntime](https://onnxruntime.ai/) and [huggingface_hub](https://github.com/huggingface/huggingface_hub).
