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

- ⚡ **Fast by default, smart when it matters.** Every image first goes through **PP-OCRv6 medium** (34.5M parameters, ~140 MB of ONNX): text detection plus line recognition, a small fraction of the VLM's compute. If the result looks shaky, textsnap hands that page to **PaddleOCR-VL-1.5**, a 0.9B vision-language model, for a second read. See [How the fallback decides](#how-the-fallback-decides).
- 🧾 **Structure on request.** `--structured` goes straight to the VLM and keeps its native markdown: tables, headings, document structure.
- 🖥 **Runs on CPU.** Both engines run on ONNX Runtime with plain old cores, pinned to your physical-core count. No CUDA. No M-series-only tricks.
- 🖼 **Images, screenshots, URLs, webpages.** Point it at a local file, a direct image URL, or a full article URL — it isolates the main content and OCRs the most prominent image. Or OCR straight from your clipboard with no argument at all — and get the text put *back* on your clipboard, ready to paste.
- 📴 **Offline after first run.** ~140 MB downloads on first use; the ~1.1 GB VLM only the first time a page needs it (or you ask for `--structured`). Both stay cached. No API keys. No quotas. Your images never leave your machine.
- 🎒 **Portable.** Drop the model files next to the script and the whole folder becomes a self-contained, copy-anywhere tool — no install, no download, no flags.
- 📚 **Batch mode.** Pass several images (or a glob) and the model loads once for all of them.
- 🪶 **One file.** The whole tool is a single Python module. Dependencies install themselves on first run if missing.
- 📝 **Plaintext or markdown.** Default output is plain text in reading order. `--structured` gives markdown with tables and headings preserved.

---

## Quickstart

```
# Install
pip install textsnap

# Snap something
textsnap screenshot.png
textsnap https://example.com/article --plaintext
textsnap photo.jpg -o ~/notes/receipt.txt
textsnap scans/*.png -o ~/notes/scans/     # batch: models load once
textsnap invoice.png --structured          # markdown: tables, headings
```

The first run downloads PP-OCRv6 (~140 MB). The VLM (~1.1 GB) is fetched the first time a page falls back or you use `--structured`. Every run after is offline.

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
├── ppocr/                      # PP-OCRv6 (default engine)
│   ├── det/inference.onnx
│   ├── det/inference.yml
│   ├── rec/inference.onnx
│   └── rec/inference.yml
├── onnx/                       # PaddleOCR-VL (fallback, --structured)
│   ├── vision_encoder_q8.onnx
│   ├── decoder_q4.onnx
│   ├── embedding.onnx
│   └── embedding.onnx.data
└── tokenizer.json
```

The two engines are found independently: a folder with only `ppocr/` works fully offline for PP-OCRv6, and low-quality pages simply keep the PP-OCRv6 text (with a one-line warning) instead of falling back. Drop those files in, and you can copy the entire `textsnap/` folder to any machine — a USB stick, an air-gapped box, a fresh laptop — and run it immediately, fully offline, with zero install steps.

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

# Markdown with tables and headings (VLM only, skips PP-OCRv6)
textsnap input.png --structured
textsnap input.png --markdown              # same thing

# ...or the VLM's reading, flattened to plain text
textsnap input.png --structured --plaintext

# Never fall back to the VLM (PP-OCRv6 only, fastest, ~140 MB total)
textsnap input.png --min-quality 0

# French and English documents: judge words against both dictionaries
textsnap scan.png --lang fr,en

# Custom output path
textsnap input.png -o ./out/extracted.txt

# VLM: raise the token cap for very dense pages
textsnap dense-page.png --structured --max-tokens 4096

# VLM: trade accuracy for speed by shrinking the image budget
textsnap input.png --structured --max-pixels 250000

# Use a local model directory instead of downloading
textsnap input.png --model-dir ~/models/textsnap

# Batch: many inputs, one model load. Globs work even where the shell
# doesn't expand them (Windows).
textsnap page1.png page2.png "scans/*.jpg" -o ./ocr-out/

# VLM: the smaller 4-bit vision encoder (see "Vision encoder variants")
textsnap input.png --structured --vision q4
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

Pass `-v` to send progress diagnostics (input type, image size, line count, page-quality score, which engine answered, decode speed) to **stderr**; stdout stays just the path either way.

Default output is **plain text in reading order**: detected lines are grouped into rows, left to right, with a blank line where the vertical gap suggests a new paragraph.

```
Quarterly Report

Region Revenue
EMEA $1.2M
APAC $0.9M
```

When a page falls back to the VLM, its markdown is flattened to the same plain-text style, so the default output format never depends on which engine answered.

With **`--structured`** you get the VLM's **native markdown**, with tables, headings and document structure preserved:

```
# Quarterly Report

| Region | Revenue |
| ------ | ------- |
| EMEA   | $1.2M   |
| APAC   | $0.9M   |
```

> **Changed in 0.4.** Default output used to be the VLM's markdown. For the 0.3 behaviour, use `--structured`.

---

## Flags

| Flag                  | Description                                                          |
| --------------------- | -------------------------------------------------------------------- |
| `-o`, `--output`      | Output `.txt` path, or a directory (required to be a directory with several inputs). Default: `./textsnaps/<n>_ocr.txt`. |
| `-v`, `--verbose`     | Print progress diagnostics to stderr. Off by default.                |
| `--structured`, `--markdown` | Skip PP-OCRv6; run PaddleOCR-VL directly and keep its native markdown. |
| `--plaintext`         | With `--structured`: flatten the markdown to plain text. (Default output is already plain text.) |
| `--min-quality Q`     | Page-quality threshold, 0 to 1, below which the VLM re-reads the page. Default `0.8`. `0` never falls back. |
| `--lang L`            | Comma-separated [wordfreq](https://github.com/rspeer/wordfreq) language codes for the quality check's dictionary test. Default `en`. Example: `--lang fr,en`. |
| `--model-dir`         | Use model files from this directory (`ppocr/…` and `onnx/…` + `tokenizer.json`). Overrides portable mode and the OS cache. |
| `--max-tokens`        | VLM: cap generated tokens. Default `2048`. Raise it for very dense pages. |
| `--max-pixels`        | VLM: image pixel budget fed to the vision encoder. Default is the model's maximum. Lower trades accuracy for speed; too low makes the model hallucinate. The image is only ever shrunk, never enlarged. |
| `--vision`            | VLM vision-encoder variant: `q8` (default) or `q4`. See [Vision encoder variants](#vision-encoder-variants). |
| `--no-verify`         | Skip SHA-256 verification of downloaded model files (not advised).   |
| `--generate-checksums`| Download the pinned model files, write a fresh manifest, and exit.   |
| `--version`           | Print the version and exit.                                          |

Environment variables:

| Variable                  | Effect |
| ------------------------- | ------ |
| `TEXTSNAP_DECODE_THREADS` | Decoder intra-op thread count, for tuning CPU decode on a specific machine. Default: a sensible value from your physical core count. |
| `TEXTSNAP_VISION`         | Default for `--vision`. |
| `TEXTSNAP_MIN_QUALITY`    | Default for `--min-quality`. |
| `TEXTSNAP_LANG`           | Default for `--lang`. |
| `TEXTSNAP_EMBED=ort`      | Run token embeddings through an ONNX Runtime session instead of the memory-mapped table (for debugging; output is identical). |
| `TEXTSNAP_LOGITS_SLICE=0` | Use the decoder graph exactly as shipped instead of the last-position patch (for debugging; output is identical). |

---

## How the fallback decides

PP-OCRv6 returns a confidence score with every recognized line. That score is roughly the average of the per-character top probabilities, so on its own it can be fooled: the model can be confident about garbage like a stamp, a table border read as `|||`, or a signature read as `llIl1`, and it is least reliable on very short lines. textsnap therefore treats it as one signal among several cheap checks, applied to each line:

- the line score is at least 0.80, and the line is at least 2 characters long;
- no more than 35% of the characters are neither letters, digits nor spaces;
- there are no runs like `aaaaa` or `-----`, and the line is not made only of `l`, `I`, `1` and `|`;
- fewer than half of its tokens mix letters, digits and odd symbols inside one word (such as `a1#b`);
- if it has 3 or more words, at least half of them are real words in one of the `--lang` languages (wordfreq Zipf frequency above 1.5). Words in CJK scripts are skipped, since wordfreq needs a segmenter for those.

The page score is the share of characters that sit in passing lines. Weighting by character count keeps one bad header from sinking an otherwise clean page. Empty detections are ignored, and a page with no text at all scores 0. If the page score is below `--min-quality` (default 0.8), PaddleOCR-VL reads the image again and its answer is used.

Some practical notes:

- **Other languages.** The dictionary test only helps if it knows the language. For non-English documents, pass `--lang` (for example `--lang de` or `--lang fr,en`). Otherwise clean French text looks like gibberish to an English dictionary and falls back needlessly.
- **Tuning.** Run with `-v` to see each page's score. Raise `--min-quality` to send more pages to the VLM; set it to `0` to never fall back.
- **The first fallback downloads the VLM** (~1.1 GB). textsnap prints a one-line notice on stderr when that happens. If the VLM can't be loaded (for example, offline with only `ppocr/` present), the PP-OCRv6 text is kept and a warning is printed once.

---

## Vision encoder variants

These apply to the VLM (PaddleOCR-VL), used for `--structured` and for fallback pages. Two builds of its vision encoder are available, selectable with `--vision`:

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

textsnap auto-downloads model weights from the Hugging Face Hub (~140 MB for PP-OCRv6, ~1.1 GB more for the VLM), so it treats those files as untrusted until proven otherwise:

- **Pinned model revisions.** Both repos are textsnap-owned mirrors, and downloads are pinned to a specific revision of each, so a moved or retagged `main` can't silently swap the weights.
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
2. **Detect (PP-OCRv6).** The image is resized so its short side is at least 64 px and its long side at most 4000 px (snapped to multiples of 32), as in PaddleOCR's general OCR pipeline. The detector's probability map is then turned into text-line quadrilaterals with DB post-processing, using the thresholds from the model's own `inference.yml`.
3. **Recognize (PP-OCRv6).** Each line is perspective-cropped upright, resized to height 48, and batched by aspect ratio through the recognizer. Greedy CTC decoding against the model's 50-language dictionary yields the text and a confidence score per line. This is a small re-implementation of the relevant parts of PaddleOCR on ONNX Runtime and OpenCV, so neither `paddleocr` nor `paddlepaddle` is needed.
4. **Judge.** The lines are scored as described in [How the fallback decides](#how-the-fallback-decides). A good page is assembled into reading-order plain text and written. That is the whole run for most images.
5. **Re-read (PaddleOCR-VL), when needed or with `--structured`.** The image goes through PaddleOCR-VL's Qwen2-VL-style smart-resize and patchify. A vision encoder (q8 by default) and an autoregressive decoder (q4) then run on CPU through ONNX Runtime, with the KV cache bound via IOBinding and token embeddings read straight from the memory-mapped fp32 table. The decoder graph is patched in memory so that prefill projects only the last prompt position onto the ~103k-token vocabulary. Decoding is greedy, guarded against runaway repetition by an n-gram block plus a loop detector.
6. **Format.** Plain text by default. Native markdown with `--structured`, or flattened with `--structured --plaintext`.

No image is sent anywhere. No state is kept between runs except the cached models.

---

## Model & cache

Models are downloaded on first use to `~/.cache/textsnap/`:

**PP-OCRv6 medium** (always, ~140 MB), from [`kouhxp/PP-OCRv6_medium-ONNX`](https://huggingface.co/kouhxp/PP-OCRv6_medium-ONNX):

- `ppocr/det/inference.onnx` + `inference.yml` — text detector (15.5M parameters) and its post-processing settings
- `ppocr/rec/inference.onnx` + `inference.yml` — line recognizer (19M parameters) and its character dictionary

**PaddleOCR-VL-1.5** (~1.1 GB, only when a page falls back or with `--structured`), from [`kouhxp/PaddleOCR-VL-1.5-ONNX`](https://huggingface.co/kouhxp/PaddleOCR-VL-1.5-ONNX):

- `onnx/vision_encoder_q8.onnx` — vision encoder + spatial-merge projector (or `vision_encoder_q4.onnx` with `--vision q4`)
- `onnx/decoder_q4.onnx` — autoregressive decoder
- `onnx/embedding.onnx` + `onnx/embedding.onnx.data` — token-embedding graph and its fp32 table
- `tokenizer.json`, `config.json`

To use your own copy, point `--model-dir` at a directory with the same layout (`ppocr/…`, `onnx/…`, `tokenizer.json`, `config.json`), or place those files next to the script for [portable mode](#portable-mode).

---

## Notes & limits

- **First run is the slow one** — it downloads ~140 MB, plus ~1.1 GB the first time the VLM is needed. After that, textsnap is fully offline.
- **Model loading is paid once per run**, so OCRing many images is much faster in one batch invocation than in a shell loop calling textsnap per file. Each engine loads only if some input needs it.
- **Plain-text mode doesn't understand layout.** PP-OCRv6 output is ordered row by row, so multi-column pages interleave their columns and tables lose their grid. Use `--structured` for those.
- **No rotation handling in PP-OCRv6 mode.** Text lines are read as detected; pages photographed sideways or upside down are better served by `--structured`.
- **VLM decode is sequential.** Dense, full-page documents take longer than a short screenshot on the VLM path. textsnap pins thread counts to your physical cores and prints a live tokens/sec readout (with `-v`) so a slow run is visibly alive, not hung.
- **`--max-tokens` caps the VLM output.** Very dense pages can hit the default 2048-token cap and truncate; raise it if the tail of a page is missing.
- **`--max-pixels` is a speed/accuracy dial.** Lowering it speeds up the vision encoder but feeds the model a coarser image; set it too low and recognition quality drops sharply. The default (the model's full budget) is the safe choice.
- **Webpage inputs OCR one image** — the most prominent one in the main content, not the whole rendered page.
- **Greedy decoding** can occasionally loop on repetitive layouts; an n-gram block prevents most loops outright and a detector trims any that remain.

---

## License

MIT for this project — see [LICENSE](LICENSE).

The default engine is **PP-OCRv6 medium**, distributed under Apache-2.0 by PaddlePaddle; textsnap downloads it from [`kouhxp/PP-OCRv6_medium-ONNX`](https://huggingface.co/kouhxp/PP-OCRv6_medium-ONNX), an unmodified, pinned mirror of [`PaddlePaddle/PP-OCRv6_medium_det_onnx`](https://huggingface.co/PaddlePaddle/PP-OCRv6_medium_det_onnx) and [`PaddlePaddle/PP-OCRv6_medium_rec_onnx`](https://huggingface.co/PaddlePaddle/PP-OCRv6_medium_rec_onnx) (paper: [arXiv:2606.13108](https://arxiv.org/abs/2606.13108)).

The fallback and `--structured` engine is **PaddleOCR-VL-1.5**, distributed under Apache-2.0 by PaddlePaddle; textsnap downloads the ONNX export from [`kouhxp/PaddleOCR-VL-1.5-ONNX`](https://huggingface.co/kouhxp/PaddleOCR-VL-1.5-ONNX), an unmodified, pinned mirror of selected files from [`onnx-community/PaddleOCR-VL-1.5-ONNX`](https://huggingface.co/onnx-community/PaddleOCR-VL-1.5-ONNX). See the [original model card](https://huggingface.co/PaddlePaddle/PaddleOCR-VL-1.5) for model terms. Powered by [onnxruntime](https://onnxruntime.ai/), [OpenCV](https://opencv.org/), [wordfreq](https://github.com/rspeer/wordfreq) and [huggingface_hub](https://github.com/huggingface/huggingface_hub).
