#!/usr/bin/env python3
"""
textsnap.py - Lean CPU OCR: PP-OCRv6 first, PaddleOCR-VL-1.5 when it matters.

Snap any image, screenshot, or webpage into plaintext. No GPU. No cloud.
One command.

Usage:
    textsnap                          # OCR image from clipboard
    textsnap path/to/img.jpg          # OCR a local image file
    textsnap https://.../x.png        # OCR a direct image URL
    textsnap https://example.com/page # OCR the biggest image on a webpage
    textsnap a.png b.png scans/*.jpg  # batch: models are loaded once
    textsnap page.png --structured    # markdown (tables, headings) via the VLM

Engines:
    By default every image goes through PP-OCRv6 medium (text detection +
    line recognition, 34.5M parameters, ~140 MB). Its per-line confidences
    and a few cheap linguistic checks give a page-quality score; if that
    falls below --min-quality (default 0.8) the image is re-read by
    PaddleOCR-VL-1.5 (0.9B VLM, ~1.1 GB, downloaded on first need).
    --structured (alias --markdown) skips PP-OCRv6 and asks the VLM for its
    native markdown directly.

Options:
    --structured    VLM only, markdown output (tables, headings preserved).
    --plaintext     With --structured: flatten the markdown to plain text.
    --min-quality Q Fallback threshold, 0..1 (default 0.8; 0 = never fall back).
    --lang L        Word-check language(s) for the quality score, e.g. en,fr.
    -o, --output    Output .txt path. Default: ./textsnaps/<name>_ocr.txt.
    --model-dir DIR Use model files from DIR instead of downloading.
    --max-tokens N  VLM: cap generated tokens (default 2048).
    --max-pixels N  VLM: image pixel budget for the vision encoder.
    --vision V      VLM: vision-encoder variant, q8 (default) or q4.

Output:
    Plaintext, UTF-8 (markdown with --structured). Default location is
    ./textsnaps/ (created if missing) under the current working directory;
    override with -o (with several inputs, -o names a directory; one output
    path per line). The filename is "<name>_ocr.txt", where <name> is the
    image filename stem (for image inputs) or the webpage slug (for HTML
    inputs).

    When the input image comes from the clipboard (textsnap run with no
    arguments), the OCR text is ALSO copied back to the clipboard so it can
    be pasted immediately -- the .txt file is still written as well.

Model files:
    Downloaded on first use and cached in ~/.cache/textsnap:
      ppocr/det/*, ppocr/rec/*   PP-OCRv6 medium ONNX (~140 MB), always.
      onnx/*, tokenizer.json     PaddleOCR-VL-1.5 ONNX (~1.1 GB), only the
                                 first time a page falls back or
                                 --structured is used.

    Portable mode: if the model files are found next to this script
    (./ppocr/... and/or ./onnx/* + ./tokenizer.json), they are used
    directly -- no download, no --model-dir flag, no setup.
"""

import sys
import os
import io
import re
import hashlib
import subprocess
import argparse
import mmap
from pathlib import Path
from urllib.parse import urlparse, unquote

# --------------------------------------------------------------------------
# Logging: all diagnostics go to stderr and are silent unless -v is passed.
# stdout is reserved for the one thing a Unix pipe wants -- the output path.
# --------------------------------------------------------------------------
__version__ = "0.4.0"
VERBOSE = False


def log(*args, **kwargs):
    """Print a [textsnap] diagnostic to stderr, but only when verbose."""
    if VERBOSE:
        kwargs.setdefault("file", sys.stderr)
        print(*args, **kwargs)

# --------------------------------------------------------------------------
# Thread env vars: must be set BEFORE numpy / onnxruntime import their native
# math backends (OpenMP / MKL / OpenBLAS), otherwise these are ignored.
# We pin them to physical-core count so the BLAS pool does not fight ORT's
# own intra-op thread pool (double-booking cores = cache thrash, slower).
# --------------------------------------------------------------------------
def _early_core_estimate():
    try:
        phys = set()
        cur = {}
        with open("/proc/cpuinfo") as f:
            for line in f:
                if ":" in line:
                    k, v = line.split(":", 1)
                    cur[k.strip()] = v.strip()
                elif line.strip() == "":
                    if "physical id" in cur and "core id" in cur:
                        phys.add((cur["physical id"], cur["core id"]))
                    cur = {}
        if phys:
            return len(phys)
    except Exception:
        pass
    return max(1, (os.cpu_count() or 4) // 2)


_NTHREADS = str(_early_core_estimate())
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, _NTHREADS)
os.environ.setdefault("OMP_WAIT_POLICY", "ACTIVE")   # keep threads hot

# --------------------------------------------------------------------------
# 0. Dependency bootstrap -- install everything inline so the script "just runs"
# --------------------------------------------------------------------------
REQUIRED = {
    # import name : pip spec
    "numpy": "numpy",
    "PIL": "pillow",
    "onnxruntime": "onnxruntime",
    "huggingface_hub": "huggingface_hub",
    "requests": "requests",
    "tokenizers": "tokenizers",
    "psutil": "psutil",
    "bs4": "beautifulsoup4",
    "readability": "readability-lxml",
    "lxml": "lxml",
    "cv2": "opencv-python-headless",
    "wordfreq": "wordfreq",
}


def _ensure_deps():
    missing = []
    for mod, pkg in REQUIRED.items():
        try:
            __import__(mod)
        except ImportError:
            missing.append(pkg)
    if missing:
        # This runs before argparse, so it can't honor -v. Send it to stderr
        # (never stdout) -- it only appears on a first run with missing deps.
        print(f"[textsnap] Installing missing packages: {', '.join(missing)}",
              file=sys.stderr)
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "--quiet", *missing]
        )
    # Clipboard support is optional/platform dependent; install lazily later.


_ensure_deps()

import numpy as np  # noqa: E402
from PIL import Image, ImageGrab  # noqa: E402
import onnxruntime as ort  # noqa: E402
import requests  # noqa: E402
from tokenizers import Tokenizer  # noqa: E402

# --------------------------------------------------------------------------
# 1. Constants from PaddleOCR-VL-1.5 config (verified against HF repo)
# --------------------------------------------------------------------------
# textsnap's own mirror of selected files from
# onnx-community/PaddleOCR-VL-1.5-ONNX @ 371b52d142968ff09e9cb5275a75eae55aa27a96
# (byte-identical; see the mirror's README). Mirroring means an upstream
# change can never alter or break what textsnap downloads.
HF_REPO = "kouhxp/PaddleOCR-VL-1.5-ONNX"
# Pin the model revision so a moved/retagged 'main' can't silently swap weights
# out from under the checksums in model_checksums.sha256.
HF_REVISION = "1605cb733ff4d1136bdb0f43f35bc717f36f58a9"
CACHE_DIR = Path(os.path.expanduser("~/.cache/textsnap"))

# SHA-256 manifest of known-good model files. Shipped alongside the script;
# also located next to the module after install. Empty/absent -> verification
# is skipped with a warning (never blocks a run).
CHECKSUM_MANIFEST = "model_checksums.sha256"

# Embedded fallback digests for the files at HF_REPO @ HF_REVISION.
# Used when the external model_checksums.sha256 is not found (e.g. a wheel
# install that didn't carry the data file). The external file, if present,
# takes precedence -- it's the source of truth and is easy to regenerate.
EMBEDDED_CHECKSUMS = {
    "onnx/vision_encoder_q4.onnx":
        "d737d600be1bd90ec1e3b537ffe1645a6d780de688904ca4301353df6086f46e",
    "onnx/decoder_q4.onnx":
        "87858a011c3f5ae8b373ec7298fba781dfe3ceb49828a803a197becdee26853c",
    "onnx/embedding.onnx":
        "91b1babbe9dbc44f2b59f8462cbf27dd1520a88b1b85695e342b05e5b4a50004",
    # embedding.onnx is a 1.8 kB graph; the 424 MB table lives in this
    # sidecar, so the sidecar must be pinned too.
    "onnx/embedding.onnx.data":
        "a2299447a5449d9bc68e4d1d1ab32b214a0e8c06f98244fbc7487342187ae6f3",
    # Default vision encoder (q4 above is the --vision q4 alternative).
    "onnx/vision_encoder_q8.onnx":
        "caf7ea82a44c9e3b5a4093129f3b9e467c4a831c9f0068aa911518cee1eadfb1",
    "tokenizer.json":
        "c8a215a59183d0d0781adc33bacd3ce6162716f7fd568fb30234a74d69803a7d",
    "config.json":
        "164809b94c8dd5b352cb9a0b9964572844398faeab27f9a6e1dd7d1a984410c8",
}

PATCH_SIZE = 14
MERGE_SIZE = 2
FACTOR = PATCH_SIZE * MERGE_SIZE          # 28
TEMPORAL_PATCH = 1
IMAGE_MEAN = np.array([0.5, 0.5, 0.5], dtype=np.float32)
IMAGE_STD = np.array([0.5, 0.5, 0.5], dtype=np.float32)
MIN_PIXELS = 112896
MAX_PIXELS = 1003520

IMAGE_TOKEN_ID = 100295
VISION_START_ID = 101305
VISION_END_ID = 101306
EOS_TOKEN_ID = 2
PAD_TOKEN_ID = 0

# Decoder architecture (for KV-cache tensor shapes)
NUM_LAYERS = 18
NUM_KV_HEADS = 2
HEAD_DIM = 128
HIDDEN_SIZE = 1024
MROPE_SECTION = [16, 24, 24]

# The OCR prompt. PaddleOCR-VL is trained for document parsing; this is the
# generic full-page parse instruction used by the reference pipeline.
OCR_PROMPT = "OCR:"

# Model files, by component. Every file listed here is downloaded, pinned in
# the checksum manifest and verified -- including external-data sidecars.
# The vision encoder comes in two variants (see --vision):
#   q8   -- 8-bit dynamic quantization (443 MB). The default. Uses integer
#           GEMMs, which suit this compute-bound ViT: in testing it ran the
#           encoder ~1.3x faster than q4 and misread less.
#   q4   -- NNCF 4-bit weight-only (231 MB). Smallest download.
# Variants produce slightly different image features, so switching changes
# outputs; compare them on your own images (bench_vision.py) before adopting.
VISION_VARIANTS = {
    "q8": ["onnx/vision_encoder_q8.onnx"],
    "q4": ["onnx/vision_encoder_q4.onnx"],
}
DEFAULT_VISION = "q8"
DECODER_FILES = ["onnx/decoder_q4.onnx"]
EMBEDDING_FILES = ["onnx/embedding.onnx", "onnx/embedding.onnx.data"]
AUX_FILES = ["tokenizer.json", "config.json"]
TOKENIZER_FILE = "tokenizer.json"


def model_files(vision=DEFAULT_VISION):
    """Repo-relative paths of every file needed to run with `vision`."""
    if vision not in VISION_VARIANTS:
        raise SystemExit(f"[textsnap] unknown vision variant '{vision}' "
                         f"(choose from {', '.join(VISION_VARIANTS)}).")
    return (VISION_VARIANTS[vision] + DECODER_FILES + EMBEDDING_FILES
            + AUX_FILES)


# --------------------------------------------------------------------------
# 1b. PP-OCRv6 medium (the default engine)
# --------------------------------------------------------------------------
# textsnap's own mirror of PaddlePaddle's official ONNX exports:
#   PaddlePaddle/PP-OCRv6_medium_det_onnx @ 61323801669c338b7891481ec7bac61ce31b576a
#   PaddlePaddle/PP-OCRv6_medium_rec_onnx @ 50c7eacafc52fa7bcf4194e8cd08e46f8558504b
# (byte-identical, repacked as det/ and rec/ in one repo). Downloads are
# pinned to this revision and verified against the digests below.
PPOCR_HF_REPO = "kouhxp/PP-OCRv6_medium-ONNX"
PPOCR_HF_REVISION = "75776f6b3864e7b3d2144c29e0278243b60e4e8f"
PPOCR_SUBDIR = "ppocr"          # under the cache / portable / --model-dir root
PPOCR_FILES = ["det/inference.onnx", "det/inference.yml",
               "rec/inference.onnx", "rec/inference.yml"]
PPOCR_CHECKSUMS = {
    "det/inference.onnx": "eb13b44b25bb36f89528b68720af8a61d9cf381176107f465db1757b65d086e1",
    "det/inference.yml": "7298d5ead546584af2504d03355f881ac7a7bc0eb1e282d3e159277c1d0af871",
    "rec/inference.onnx": "9c09abf0957f7968c7586464b7397b84ad2387a0497a351af40e9acc71b673ba",
    "rec/inference.yml": "991b700facf5b50a7de193468207d5f4255b538dde0d312ae3b7c7a9b6873129",
}
EMBEDDED_CHECKSUMS.update(PPOCR_CHECKSUMS)

# Detection preprocessing, as in PaddleOCR 3.x's general OCR pipeline:
# upscale so the SHORT side is at least 64 px, cap the long side at 4000,
# snap both to multiples of 32. The DB post-processing thresholds come from
# the model's own inference.yml; these are fallbacks.
DET_LIMIT_SIDE = 64
DET_MAX_SIDE = 4000
DET_DEFAULTS = {"thresh": 0.3, "box_thresh": 0.6, "unclip_ratio": 1.5,
                "max_candidates": 1000}
DET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
DET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
DET_MIN_BOX = 3
# Recognition: text lines are resized to height 48, width by aspect ratio
# (at least 320, at most 3200), and batched by similar aspect ratio.
REC_HEIGHT = 48
REC_MIN_WIDTH = 320
REC_MAX_WIDTH = 3200
REC_BATCH = 6

# Quality gate: share of characters that sit in lines passing line_ok().
DEFAULT_MIN_QUALITY = 0.8
DEFAULT_LANGS = "en"


# --------------------------------------------------------------------------
# 2. Input detection: figure out what the positional arg is
# --------------------------------------------------------------------------
def detect_input(arg):
    """
    Returns (kind, value) where kind is one of:
        'clipboard'  -> value is None
        'file'       -> value is a Path
        'image_url'  -> value is a URL string
        'html_url'   -> value is a URL string
    """
    if arg is None:
        return "clipboard", None

    p = Path(arg)
    if p.exists() and p.is_file():
        return "file", p

    parsed = urlparse(arg)
    if parsed.scheme in ("http", "https"):
        # Probe the URL: Content-Type header is the source of truth.
        ctype = ""
        try:
            head = requests.head(arg, allow_redirects=True, timeout=15)
            ctype = head.headers.get("Content-Type", "").lower()
            # Some servers don't answer HEAD usefully; fall back to a ranged GET.
            if not ctype or head.status_code >= 400:
                g = requests.get(arg, stream=True, timeout=15,
                                 headers={"Range": "bytes=0-0"})
                ctype = g.headers.get("Content-Type", "").lower()
                g.close()
        except requests.RequestException:
            pass

        if ctype.startswith("image/"):
            return "image_url", arg
        if "html" in ctype or "xml" in ctype:
            return "html_url", arg
        # Ambiguous / no content-type: fall back to extension heuristic.
        ext = Path(parsed.path).suffix.lower()
        if ext in (".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif", ".tiff"):
            return "image_url", arg
        return "html_url", arg

    # Not a URL, not an existing file.
    raise SystemExit(f"[textsnap] '{arg}' is neither an existing file nor a "
                     f"http(s) URL.")


# --------------------------------------------------------------------------
# 3. Load image from each input kind
# --------------------------------------------------------------------------
def _download_bytes(url):
    r = requests.get(url, timeout=60, headers={"User-Agent": f"textsnap/{__version__}"})
    r.raise_for_status()
    return r.content


def load_from_clipboard():
    try:
        img = ImageGrab.grabclipboard()
    except Exception as e:
        raise SystemExit(f"[textsnap] Could not read clipboard: {e}\n"
                         "On Linux you may need 'xclip' or 'wl-clipboard' "
                         "installed.")
    if img is None:
        raise SystemExit("[textsnap] No image found in clipboard.")
    if isinstance(img, list):
        # Clipboard held file path(s) rather than raw image data.
        paths = [Path(x) for x in img if Path(x).is_file()]
        if not paths:
            raise SystemExit("[textsnap] Clipboard holds no usable image.")
        img = Image.open(paths[0])
    return img.convert("RGB"), "clipboard"


def load_from_file(path):
    return Image.open(path).convert("RGB"), path.stem


def copy_text_to_clipboard(text):
    """Best-effort: put `text` on the system clipboard. Returns True on
    success, False otherwise. Never raises -- clipboard-out is a convenience,
    not a contract, so a failure here must not fail the run.

    Tries platform-native tools so it works without extra Python deps:
      macOS   -> pbcopy
      Windows -> clip
      Linux   -> wl-copy (Wayland) or xclip / xsel (X11)
    """
    data = text.encode("utf-8")
    if sys.platform == "darwin":
        cmds = [["pbcopy"]]
    elif sys.platform.startswith("win"):
        cmds = [["clip"]]
    else:
        cmds = [["wl-copy"], ["xclip", "-selection", "clipboard"],
                ["xsel", "--clipboard", "--input"]]
    for cmd in cmds:
        try:
            p = subprocess.run(cmd, input=data,
                               stdout=subprocess.DEVNULL,
                               stderr=subprocess.DEVNULL)
            if p.returncode == 0:
                return True
        except FileNotFoundError:
            continue   # tool not installed -- try the next one
        except Exception:
            continue
    return False


def load_from_image_url(url):
    data = _download_bytes(url)
    img = Image.open(io.BytesIO(data)).convert("RGB")
    stem = Path(unquote(urlparse(url).path)).stem or "image"
    return img, stem


def load_from_html_url(url):
    """Use readability-lxml to isolate main content, then pick the most
    prominent <img> from that cleaned region."""
    from readability import Document
    from bs4 import BeautifulSoup
    from urllib.parse import urljoin

    html = requests.get(url, timeout=60,
                         headers={"User-Agent": f"textsnap/{__version__}"}).text
    doc = Document(html)
    main_html = doc.summary()           # de-fluffed main content
    title = doc.short_title() or urlparse(url).netloc

    soup = BeautifulSoup(main_html, "lxml")
    candidates = soup.find_all("img")

    # If readability stripped all images, fall back to the full page.
    if not candidates:
        soup = BeautifulSoup(html, "lxml")
        candidates = soup.find_all("img")

    def score(tag):
        """Prominence heuristic: prefer explicit large dimensions, then
        document order (earlier = more prominent)."""
        w = h = 0
        for attr in ("width", "height"):
            v = tag.get(attr, "")
            m = re.search(r"\d+", str(v))
            if m and attr == "width":
                w = int(m.group())
            if m and attr == "height":
                h = int(m.group())
        return w * h

    # Build ordered list of (score, order_index, src)
    scored = []
    for i, tag in enumerate(candidates):
        src = (tag.get("src") or tag.get("data-src")
               or tag.get("data-original") or "")
        if not src:
            continue
        # Skip obvious non-content images.
        if src.startswith("data:"):
            continue
        if re.search(r"(sprite|icon|logo|avatar|pixel|tracking|spacer|"
                     r"blank|1x1)", src, re.I):
            continue
        scored.append((score(tag), -i, urljoin(url, src)))

    if not scored:
        raise SystemExit("[textsnap] No usable image found on the page.")

    # Highest declared area wins; ties broken by earliest appearance.
    scored.sort(reverse=True)

    # If no element declared a size (all score 0), download top few and
    # measure real pixels to pick the biggest.
    if scored[0][0] == 0:
        best_img, best_px, best_src = None, -1, None
        for _, _, src in scored[:8]:
            try:
                data = _download_bytes(src)
                im = Image.open(io.BytesIO(data))
                px = im.width * im.height
                if px > best_px:
                    best_img, best_px, best_src = im, px, src
            except Exception:
                continue
        if best_img is None:
            raise SystemExit("[textsnap] Could not download any page image.")
        img = best_img.convert("RGB")
    else:
        src = scored[0][2]
        img = Image.open(io.BytesIO(_download_bytes(src))).convert("RGB")

    slug = re.sub(r"[^\w\-]+", "_", title).strip("_").lower()[:60] or "webpage"
    return img, slug


# --------------------------------------------------------------------------
# 4. (Resizing is handled entirely by smart_resize() in section 5, which
#    bounds the image to MAX_PIXELS / MIN_PIXELS and snaps to the patch grid.
#    There is deliberately no separate "cap the longest side" step -- pre-
#    shrinking on top of smart_resize only destroys resolution the vision
#    encoder could have used.)
# --------------------------------------------------------------------------


# --------------------------------------------------------------------------
# 5. PaddleOCR-VL image preprocessing (Qwen2-VL style smart_resize + patchify)
#    Mirrors image_processing_paddleocr_vl.py from the HF repo.
# --------------------------------------------------------------------------
def smart_resize(height, width, factor=FACTOR,
                 min_pixels=MIN_PIXELS, max_pixels=MAX_PIXELS):
    import math
    if height < factor:
        width = round((width * factor) / height)
        height = factor
    if width < factor:
        height = round((height * factor) / width)
        width = factor
    if max(height, width) / min(height, width) > 200:
        raise ValueError("aspect ratio too extreme")
    h_bar = round(height / factor) * factor
    w_bar = round(width / factor) * factor
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = math.floor(height / beta / factor) * factor
        w_bar = math.floor(width / beta / factor) * factor
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor
    return h_bar, w_bar


def preprocess_image(img, max_pixels=MAX_PIXELS):
    """Returns (pixel_values, grid_thw).

    pixel_values is produced in the canonical rank-5 layout
        (num_patches, channel, temporal_patch, patch, patch)
    which matches the PaddleOCR-VL / Qwen2-VL ONNX vision encoder export.
    run_ocr() adapts this to rank 2 or rank 4 if the loaded graph declares
    a different rank.
    """
    w, h = img.size
    rh, rw = smart_resize(h, w, max_pixels=max_pixels)
    img = img.resize((rw, rh), Image.BICUBIC)

    arr = np.asarray(img, dtype=np.float32) / 255.0       # rescale
    arr = (arr - IMAGE_MEAN) / IMAGE_STD                  # normalize
    arr = arr.transpose(2, 0, 1)                          # HWC -> CHW
    arr = arr[np.newaxis, ...]                            # (1, 3, H, W)

    # temporal tiling (temporal_patch_size == 1 here, so tile to 1)
    patches = np.tile(arr, (TEMPORAL_PATCH, 1, 1, 1))

    channel = patches.shape[1]
    grid_t = patches.shape[0] // TEMPORAL_PATCH
    grid_h = rh // PATCH_SIZE
    grid_w = rw // PATCH_SIZE

    patches = patches.reshape(
        grid_t, TEMPORAL_PATCH, channel,
        grid_h, PATCH_SIZE, grid_w, PATCH_SIZE,
    )
    patches = patches.transpose(0, 3, 5, 2, 1, 4, 6)
    # rank-5: (num_patches, channel, temporal_patch, patch, patch)
    pixel_values = patches.reshape(
        grid_t * grid_h * grid_w, channel, TEMPORAL_PATCH,
        PATCH_SIZE, PATCH_SIZE,
    ).astype(np.float32)
    return pixel_values, (grid_t, grid_h, grid_w)


def fit_pixel_values(pixel_values, declared_shape):
    """Reshape pixel_values (produced as (N, C, T, P, P)) to match the
    vision encoder's declared input shape.

    declared_shape is session_input.shape, a list mixing ints and strings
    (symbolic dims). The PaddleOCR-VL-1.5 ONNX export declares:
        [1, 'num_patches', 3, 14, 14]
    i.e. a leading batch axis, patches on axis 1, channel on axis 2, no
    temporal axis. We collapse the temporal axis (size 1) and re-place the
    batch axis as needed.
    """
    n = pixel_values.shape[0]
    c = pixel_values.shape[1]
    # Drop the temporal axis -> (N, C, P, P)
    pv = pixel_values.reshape(n, c, PATCH_SIZE, PATCH_SIZE)

    if not declared_shape:
        return pv

    rank = len(declared_shape)

    if rank == 5:
        # [batch, num_patches, C, P, P]  -> add leading batch axis
        return pv[np.newaxis, ...]                       # (1, N, C, P, P)
    if rank == 4:
        # [num_patches, C, P, P]
        return pv                                        # (N, C, P, P)
    if rank == 3:
        # [num_patches, C, P*P]  (rare)
        return pv.reshape(n, c, PATCH_SIZE * PATCH_SIZE)
    if rank == 2:
        # [num_patches, C*P*P]  (Qwen2-VL flattened style)
        return pv.reshape(n, -1)
    return pv


# --------------------------------------------------------------------------
# 6. Model download + integrity verification
# --------------------------------------------------------------------------
_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_HEX40 = re.compile(r"^[0-9a-f]{40}$")


def _sha256_file(path, chunk=1 << 20):
    """Stream a file through SHA-256 without loading it all into memory."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def _find_checksum_manifest():
    """Locate model_checksums.sha256: next to this module, in CWD, or in the
    cache dir. Returns a Path or None."""
    here = Path(__file__).resolve().parent
    for cand in (here / CHECKSUM_MANIFEST,
                 Path.cwd() / CHECKSUM_MANIFEST,
                 CACHE_DIR / CHECKSUM_MANIFEST):
        if cand.is_file():
            return cand
    return None


def _load_checksums(manifest_path):
    """Parse a `sha256sum`-style manifest into {repo_path: sha256}."""
    sums = {}
    for line in manifest_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split(None, 1)
        if len(parts) != 2:
            continue
        digest, name = parts[0].lower(), parts[1].strip().lstrip("*")
        sums[name] = digest
    return sums


def verify_files(file_map, checksums):
    """Verify downloaded files against the manifest.

    file_map: {repo_relative_name: local_path}
    checksums: {repo_relative_name: expected_sha256}

    Hard-fails (SystemExit) on any mismatch. Files with no manifest entry are
    reported but not fatal -- the manifest is the source of truth for *what*
    is pinned, and a partial manifest is still useful.
    """
    verified = 0
    for name, path in file_map.items():
        expected = checksums.get(name)
        if not expected:
            log(f"[textsnap]   (no pinned checksum for {name} -- skipped)")
            continue
        if not _HEX64.match(expected):
            raise SystemExit(
                f"[textsnap] No pinned checksum for {name} yet (placeholder "
                f"'{expected}'). Pin its SHA-256 in textsnap.py, or "
                f"pass --no-verify for local experiments.")
        actual = _sha256_file(path)
        if actual != expected:
            raise SystemExit(
                f"[textsnap] CHECKSUM MISMATCH for {name}\n"
                f"             expected {expected}\n"
                f"             got      {actual}\n"
                f"[textsnap] Refusing to use a model file that does not match "
                f"the pinned digest. Delete {path} and re-run, or update "
                f"{CHECKSUM_MANIFEST} if you intend to use a new revision."
            )
        verified += 1
        log(f"[textsnap]   verified {name}")
    return verified


def _looks_like_model_dir(d, vision=DEFAULT_VISION):
    """True if `d` contains every file needed to run with `vision`.

    Used for 'portable mode': if the model files sit next to the script, we
    use them directly -- no download, no --model-dir, no setup. This lets a
    user copy the whole textsnap folder (script + model files) to any machine
    and run it offline immediately.
    """
    d = Path(d)
    needed = [f for f in model_files(vision) if f != "config.json"]
    return all((d / f).is_file() for f in needed)


def _portable_model_dir(vision=DEFAULT_VISION):
    """Return the directory next to the textsnap script if it holds a model
    set, else None. Tries the module dir and (for frozen/symlinked installs)
    the resolved executable dir."""
    candidates = []
    try:
        candidates.append(Path(__file__).resolve().parent)
    except NameError:
        pass
    # Frozen build (PyInstaller etc.): sys.executable is the binary.
    if getattr(sys, "frozen", False):
        candidates.append(Path(sys.executable).resolve().parent)
    for c in candidates:
        if _looks_like_model_dir(c, vision):
            return c
    return None


def _pinned_checksums():
    """Known-good digests: the embedded table, overridden by the external
    manifest where it has an entry. Merging (rather than letting the manifest
    replace the table) keeps files the manifest predates -- e.g. new
    sidecars -- verified instead of silently skipped."""
    checksums = dict(EMBEDDED_CHECKSUMS)
    manifest = _find_checksum_manifest()
    if manifest is not None:
        checksums.update(_load_checksums(manifest))
        return checksums, f"{manifest.name} + embedded digests"
    log(f"[textsnap] {CHECKSUM_MANIFEST} not found -- using embedded "
        f"digests.")
    return checksums, "embedded digests"


def get_model_dir(override=None, verify=True, vision=DEFAULT_VISION):
    files = model_files(vision)   # validates `vision` early
    if override:
        d = Path(override)
        if not d.exists():
            raise SystemExit(f"[textsnap] --model-dir {d} does not exist.")
        return d

    # Portable mode: model files sitting next to the script take precedence
    # over the OS cache. No flag needed -- copy the folder, run it anywhere.
    portable = _portable_model_dir(vision)
    if portable is not None:
        log(f"[textsnap] Portable mode: using model files next to the "
            f"script ({portable}). Integrity check skipped -- locally "
            f"placed files are trusted, same as --model-dir.")
        return portable

    from huggingface_hub import hf_hub_download

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    log(f"[textsnap] Ensuring model files are cached (vision encoder: "
        f"{vision}; ~1.1 GB on first run with the default q8)...")

    # Every file is required, sidecars included: the exact set per component
    # is known (see model_files()), so there is no need to probe for
    # optional *_data files and swallow 404s on every run.
    downloaded = {}   # repo-relative name -> local path, for verification
    for fname in files:
        local = hf_hub_download(repo_id=HF_REPO, filename=fname,
                                revision=HF_REVISION,
                                local_dir=str(CACHE_DIR))
        downloaded[fname] = local

    # ---- integrity check -------------------------------------------------
    if not verify:
        log("[textsnap] WARNING: --no-verify set -- skipping model integrity "
            "check.")
        return CACHE_DIR

    checksums, source = _pinned_checksums()
    log(f"[textsnap] Verifying model files against {source}...")
    n = verify_files(downloaded, checksums)
    log(f"[textsnap] Integrity OK ({n} files verified).")

    return CACHE_DIR


def _script_dirs():
    """Directories that count as 'next to the script' for portable mode."""
    out = []
    try:
        out.append(Path(__file__).resolve().parent)
    except NameError:
        pass
    if getattr(sys, "frozen", False):
        out.append(Path(sys.executable).resolve().parent)
    return out


def _looks_like_ppocr_dir(d):
    d = Path(d)
    return all((d / f).is_file() for f in PPOCR_FILES)


def vlm_available(override=None, vision=DEFAULT_VISION):
    """True if the VLM can load without a download (used to warn before a
    fallback triggers the ~1.1 GB first-time fetch)."""
    if override:
        return True
    if _portable_model_dir(vision) is not None:
        return True
    return all((CACHE_DIR / f).is_file() for f in model_files(vision))


def get_ppocr_dir(override=None, verify=True):
    """Directory holding det/ and rec/ for PP-OCRv6 medium. Same resolution
    order as the VLM: --model-dir, then portable, then the OS cache."""
    if override:
        d = Path(override) / PPOCR_SUBDIR
        if not _looks_like_ppocr_dir(d):
            raise SystemExit(
                f"[textsnap] --model-dir {override}: PP-OCRv6 files not found "
                f"under {d} (expected {', '.join(PPOCR_FILES)}).")
        return d
    for c in _script_dirs():
        if _looks_like_ppocr_dir(c / PPOCR_SUBDIR):
            log(f"[textsnap] Portable mode: PP-OCRv6 files next to the "
                f"script ({c / PPOCR_SUBDIR}); integrity check skipped.")
            return c / PPOCR_SUBDIR

    if not _HEX40.match(PPOCR_HF_REVISION):
        raise SystemExit(
            "[textsnap] The PP-OCRv6 mirror revision is not pinned yet "
            "(PPOCR_HF_REVISION). Pin it in textsnap.py, or point "
            "--model-dir at a directory containing ppocr/det and ppocr/rec.")

    from huggingface_hub import hf_hub_download

    dest = CACHE_DIR / PPOCR_SUBDIR
    dest.mkdir(parents=True, exist_ok=True)
    log("[textsnap] Ensuring PP-OCRv6 files are cached (~140 MB on first "
        "run)...")
    downloaded = {}
    for fname in PPOCR_FILES:
        downloaded[fname] = hf_hub_download(
            repo_id=PPOCR_HF_REPO, filename=fname,
            revision=PPOCR_HF_REVISION, local_dir=str(dest))
    if not verify:
        log("[textsnap] WARNING: --no-verify set -- skipping PP-OCRv6 "
            "integrity check.")
        return dest
    checksums, source = _pinned_checksums()
    n = verify_files(downloaded, checksums)
    log(f"[textsnap] PP-OCRv6 integrity OK ({n} files verified against "
        f"{source}).")
    return dest


# --------------------------------------------------------------------------
# 7. ONNX session helpers -- introspect graph I/O so we bind by pattern,
#    not by hard-coded names (robust to export naming differences).
# --------------------------------------------------------------------------
def _physical_cores():
    """Best-effort physical (not logical/hyperthread) core count."""
    try:
        import psutil
        n = psutil.cpu_count(logical=False)
        if n:
            return n
    except Exception:
        pass
    # Linux: parse /proc/cpuinfo for distinct core ids.
    try:
        phys = set()
        cur = {}
        with open("/proc/cpuinfo") as f:
            for line in f:
                if ":" in line:
                    k, v = line.split(":", 1)
                    cur[k.strip()] = v.strip()
                elif line.strip() == "":
                    if "physical id" in cur and "core id" in cur:
                        phys.add((cur["physical id"], cur["core id"]))
                    cur = {}
        if phys:
            return len(phys)
    except Exception:
        pass
    # Fallback: assume hyperthreading, halve logical count.
    log = os.cpu_count() or 4
    return max(1, log // 2)


_PHYS_CORES = _physical_cores()


def make_session(path, role="generic"):
    """Build an ORT session tuned for the model's access pattern.

    role:
        'vision'  -> big single parallel forward pass; use all phys cores.
        'decoder' -> autoregressive, latency-bound; oversubscription hurts,
                     so cap intra-op threads (<=4, env-overridable) and
                     disable mem_pattern (seq length grows every step).
        'embed'   -> trivial lookup; minimal threads.
    """
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    so.enable_cpu_mem_arena = True
    # mem_pattern pre-plans buffers assuming STATIC shapes. The decoder's
    # sequence axis grows by one every token, so the plan is invalidated each
    # step and ORT just pays the planning overhead for nothing. Enable it only
    # where shapes are actually static (vision encoder, embedding lookup).
    so.enable_mem_pattern = True

    if role == "vision":
        so.intra_op_num_threads = _PHYS_CORES
        so.inter_op_num_threads = 1
    elif role == "decoder":
        # Per-token decode is dominated by many SMALL GEMMs (batch=1, one
        # token). Beyond a handful of threads, cross-core synchronization
        # costs more than the parallel work saved -- classic oversubscription.
        # Default to <=4; let a deployment override via TEXTSNAP_DECODE_THREADS
        # since the sweet spot is CPU-dependent.
        _env = os.environ.get("TEXTSNAP_DECODE_THREADS")
        if _env and _env.isdigit() and int(_env) > 0:
            so.intra_op_num_threads = int(_env)
        else:
            so.intra_op_num_threads = max(1, min(_PHYS_CORES, 4))
        so.inter_op_num_threads = 1
        # Dynamic (growing) seq length defeats mem_pattern -- turn it off here.
        so.enable_mem_pattern = False
    elif role == "embed":
        so.intra_op_num_threads = min(_PHYS_CORES, 2)
        so.inter_op_num_threads = 1
    else:
        so.intra_op_num_threads = _PHYS_CORES
        so.inter_op_num_threads = 1

    # `path` may also be the serialized model itself (bytes), used for the
    # decoder graph patched in memory by _last_position_logits_model().
    model = path if isinstance(path, (bytes, bytearray)) else str(path)
    return ort.InferenceSession(model, sess_options=so,
                                providers=["CPUExecutionProvider"])


def _find(names, *keywords):
    """Return the first name containing all keywords (case-insensitive)."""
    for n in names:
        low = n.lower()
        if all(k in low for k in keywords):
            return n
    return None


# --------------------------------------------------------------------------
# 7b. A minimal ONNX protobuf reader/writer.
#
# Two load-time optimizations need to look inside an .onnx file: the token
# embedding (to find the table's bytes so they can be memory-mapped) and the
# decoder's final vocab projection (to restrict it to the last position).
# The `onnx` package would do this, but it is a large dependency for a
# handful of fields, so this is a small wire-format reader for exactly the
# fields used. Field numbers are from onnx/onnx.proto3.
# --------------------------------------------------------------------------
_PB_VARINT, _PB_I64, _PB_LEN, _PB_I32 = 0, 1, 2, 5

# ModelProto / OperatorSetIdProto
_M_GRAPH, _M_OPSET = 7, 8
# GraphProto
_G_NODE, _G_INIT, _G_INPUT, _G_OUTPUT, _G_SPARSE_INIT = 1, 5, 11, 12, 15
# NodeProto
_N_INPUT, _N_OUTPUT, _N_NAME, _N_OP, _N_ATTR, _N_DOMAIN = 1, 2, 3, 4, 5, 7
# TensorProto
_T_DIMS, _T_DTYPE, _T_NAME, _T_RAW, _T_EXT, _T_LOC = 1, 2, 8, 9, 13, 14
_T_TYPED_DATA = {4, 5, 6, 7, 10, 11}   # float/int32/string/int64/double/uint64
_ONNX_FLOAT, _ONNX_INT64, _ONNX_FLOAT16 = 1, 7, 10
_ONNX_DOMAINS = ("", "ai.onnx")


def _pb_varint(buf, pos):
    result = shift = 0
    while True:
        b = buf[pos]
        pos += 1
        result |= (b & 0x7F) << shift
        if not b & 0x80:
            return result, pos
        shift += 7
        if shift >= 70:
            raise ValueError("malformed varint")


def _pb_signed(v):
    return v - (1 << 64) if v >= (1 << 63) else v


def _pb_fields(buf, start, end):
    """Yield (field, wire_type, value, record_start, record_end) for each
    record in buf[start:end]. `value` is an int for scalar wire types and a
    (start, end) span into `buf` for length-delimited ones, so large payloads
    (weights) are skipped without being copied."""
    pos = start
    while pos < end:
        rec_start = pos
        key, pos = _pb_varint(buf, pos)
        field, wt = key >> 3, key & 7
        if wt == _PB_VARINT:
            val, pos = _pb_varint(buf, pos)
        elif wt == _PB_I64:
            val = int.from_bytes(buf[pos:pos + 8], "little")
            pos += 8
        elif wt == _PB_LEN:
            n, pos = _pb_varint(buf, pos)
            val = (pos, pos + n)
            pos += n
        elif wt == _PB_I32:
            val = int.from_bytes(buf[pos:pos + 4], "little")
            pos += 4
        else:
            raise ValueError(f"unsupported protobuf wire type {wt}")
        if pos > end:
            raise ValueError("truncated protobuf")
        yield field, wt, val, rec_start, pos


def _pb_text(buf, span):
    return bytes(buf[span[0]:span[1]]).decode("utf-8")


def _pb_enc_varint(n):
    n &= (1 << 64) - 1
    out = bytearray()
    while True:
        b = n & 0x7F
        n >>= 7
        if n:
            out.append(b | 0x80)
        else:
            out.append(b)
            return bytes(out)


def _pb_enc_len(field, payload):
    return (_pb_enc_varint((field << 3) | _PB_LEN)
            + _pb_enc_varint(len(payload)) + payload)


def _pb_enc_int(field, value):
    return _pb_enc_varint((field << 3) | _PB_VARINT) + _pb_enc_varint(value)


def _pb_parse_node(buf, span):
    node = {"input": [], "output": [], "name": "", "op_type": "",
            "domain": "", "attrs": {}}
    for f, wt, v, _, _ in _pb_fields(buf, *span):
        if wt != _PB_LEN:
            continue
        if f == _N_INPUT:
            node["input"].append(_pb_text(buf, v))
        elif f == _N_OUTPUT:
            node["output"].append(_pb_text(buf, v))
        elif f == _N_NAME:
            node["name"] = _pb_text(buf, v)
        elif f == _N_OP:
            node["op_type"] = _pb_text(buf, v)
        elif f == _N_DOMAIN:
            node["domain"] = _pb_text(buf, v)
        elif f == _N_ATTR:
            # Only int-valued attributes are needed (e.g. Gather's axis).
            name, ival = None, None
            for af, awt, av, _, _ in _pb_fields(buf, *v):
                if af == 1 and awt == _PB_LEN:
                    name = _pb_text(buf, av)
                elif af == 3 and awt == _PB_VARINT:
                    ival = _pb_signed(av)
            if name is not None:
                node["attrs"][name] = ival
    return node


def _pb_parse_tensor(buf, span):
    t = {"name": "", "dims": [], "dtype": 0, "raw": None, "external": {},
         "location": 0, "typed_data": False}
    for f, wt, v, _, _ in _pb_fields(buf, *span):
        if f == _T_DIMS:
            if wt == _PB_VARINT:
                t["dims"].append(_pb_signed(v))
            elif wt == _PB_LEN:                       # packed
                pos = v[0]
                while pos < v[1]:
                    d, pos = _pb_varint(buf, pos)
                    t["dims"].append(_pb_signed(d))
        elif f == _T_DTYPE and wt == _PB_VARINT:
            t["dtype"] = v
        elif f == _T_NAME and wt == _PB_LEN:
            t["name"] = _pb_text(buf, v)
        elif f == _T_RAW and wt == _PB_LEN:
            t["raw"] = v
        elif f == _T_EXT and wt == _PB_LEN:
            key = val = None
            for ef, ewt, ev, _, _ in _pb_fields(buf, *v):
                if ewt == _PB_LEN and ef == 1:
                    key = _pb_text(buf, ev)
                elif ewt == _PB_LEN and ef == 2:
                    val = _pb_text(buf, ev)
            if key is not None:
                t["external"][key] = val
        elif f == _T_LOC and wt == _PB_VARINT:
            t["location"] = v
        elif f in _T_TYPED_DATA:
            t["typed_data"] = True
    return t


def _pb_value_info_name(buf, span):
    for f, wt, v, _, _ in _pb_fields(buf, *span):
        if f == 1 and wt == _PB_LEN:
            return _pb_text(buf, v)
    return ""


def _pb_parse_model(buf):
    """Return (graph_payload_span, graph_record_span, {domain: opset})."""
    graphs, opsets = [], {}
    for f, wt, v, rs, re_ in _pb_fields(buf, 0, len(buf)):
        if f == _M_GRAPH and wt == _PB_LEN:
            graphs.append((v, (rs, re_)))
        elif f == _M_OPSET and wt == _PB_LEN:
            domain, version = "", 0
            for of, owt, ov, _, _ in _pb_fields(buf, *v):
                if of == 1 and owt == _PB_LEN:
                    domain = _pb_text(buf, ov)
                elif of == 2 and owt == _PB_VARINT:
                    version = ov
            opsets[domain] = version
    if len(graphs) != 1:
        raise ValueError(f"expected one graph, found {len(graphs)}")
    return graphs[0][0], graphs[0][1], opsets


def _pb_parse_graph(buf, span):
    """Nodes (with their record/payload spans), initializers by name, and
    graph input/output names. Initializer payloads are not copied."""
    g = {"nodes": [], "inits": {}, "inputs": [], "outputs": [],
         "sparse": False}
    for f, wt, v, rs, re_ in _pb_fields(buf, *span):
        if wt != _PB_LEN:
            continue
        if f == _G_NODE:
            g["nodes"].append((_pb_parse_node(buf, v), (rs, re_), v))
        elif f == _G_INIT:
            t = _pb_parse_tensor(buf, v)
            g["inits"][t["name"]] = t
        elif f == _G_INPUT:
            g["inputs"].append(_pb_value_info_name(buf, v))
        elif f == _G_OUTPUT:
            g["outputs"].append(_pb_value_info_name(buf, v))
        elif f == _G_SPARSE_INIT:
            g["sparse"] = True
    return g


def _map_file(path):
    """Read-only memory map of a whole file (b"" for an empty file)."""
    with open(path, "rb") as f:
        if os.fstat(f.fileno()).st_size == 0:
            return b""
        return mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)


# --------------------------------------------------------------------------
# 7c. Token embeddings as a memory-mapped table.
#
# embedding.onnx is a single Gather over a ~103k x 1024 fp32 table (424 MB in
# embedding.onnx.data). Running it through an ORT session loads the whole
# table into session memory and pays a full session.run() for every
# generated token. Instead we locate the table's bytes, memory-map them and
# index rows with numpy: the OS pages in only the rows actually used (a few
# MB per image), and a per-token lookup is a plain array index.
#
# The graph is checked to be exactly Gather(table, ids) -- any extra op
# (scaling, casting the output, ...) falls back to the ORT session, as does
# TEXTSNAP_EMBED=ort.
# --------------------------------------------------------------------------
class EmbeddingTable:
    _NP_DTYPES = {_ONNX_FLOAT: "<f4", _ONNX_FLOAT16: "<f2"}

    def __init__(self, table, mapping):
        self.table = table          # (vocab, hidden), read-only, file-backed
        self._mapping = mapping     # keeps the mmap alive
        self.vocab, self.hidden = table.shape
        self.dtype_name = "fp32" if table.dtype.itemsize == 4 else "fp16"

    def lookup(self, ids):
        """int ids (any shape) -> float32 embeddings, shape ids.shape + (H,).
        Fancy indexing copies the rows, so the result is writable."""
        ids = np.asarray(ids, dtype=np.int64)
        if ids.size and (int(ids.min()) < 0 or int(ids.max()) >= self.vocab):
            raise ValueError("token id outside the embedding table")
        return np.ascontiguousarray(self.table[ids], dtype=np.float32)

    @classmethod
    def from_onnx(cls, path):
        """Return (EmbeddingTable, None), or (None, reason) if the graph is
        not a plain lookup this class can serve exactly."""
        path = Path(path)
        mm = _map_file(path)
        buf = memoryview(mm)
        try:
            spec = cls._inspect(buf)
        finally:
            buf.release()
        if isinstance(spec, str):
            _close_quietly(mm)
            return None, spec

        np_dtype = np.dtype(cls._NP_DTYPES[spec["dtype"]])
        vocab, hidden = spec["dims"]
        count = vocab * hidden
        nbytes = count * np_dtype.itemsize

        if spec["location"] == 1:                     # external data file
            _close_quietly(mm)
            ext = spec["external"]
            loc = ext.get("location")
            if not loc:
                return None, "external tensor has no location"
            base = path.parent.resolve()
            data_path = (base / loc).resolve()
            if base not in data_path.parents:
                return None, "external data path leaves the model directory"
            if not data_path.is_file():
                return None, f"missing external data file {loc}"
            offset = int(ext.get("offset") or 0)
            length = int(ext.get("length") or nbytes)
            if length != nbytes:
                return None, (f"external length {length} != expected "
                              f"{nbytes} bytes")
            dm = _map_file(data_path)
            if offset + nbytes > len(dm):
                _close_quietly(dm)
                return None, "external data file is too short"
            table = np.frombuffer(dm, dtype=np_dtype, count=count,
                                  offset=offset).reshape(vocab, hidden)
            return cls(table, dm), None

        if spec["raw"] is None:
            _close_quietly(mm)
            return None, "table is not stored as raw bytes"
        start, end = spec["raw"]
        if end - start != nbytes:
            _close_quietly(mm)
            return None, "raw table size does not match its shape"
        table = np.frombuffer(mm, dtype=np_dtype, count=count,
                              offset=start).reshape(vocab, hidden)
        return cls(table, mm), None

    @classmethod
    def _inspect(cls, buf):
        """The table's TensorProto fields, or a reason string."""
        gspan, _, _ = _pb_parse_model(buf)
        g = _pb_parse_graph(buf, gspan)
        if g["sparse"]:
            return "graph has sparse initializers"
        if len(g["outputs"]) != 1:
            return f"graph has {len(g['outputs'])} outputs, expected 1"
        producer = {}
        for node, _, _ in g["nodes"]:
            for o in node["output"]:
                producer[o] = node
        used = []

        # Output <- [Identity]* <- Gather
        n = producer.get(g["outputs"][0])
        while (n is not None and n["op_type"] == "Identity"
               and n["domain"] in _ONNX_DOMAINS):
            used.append(n)
            n = producer.get(n["input"][0])
        if (n is None or n["op_type"] != "Gather"
                or n["domain"] not in _ONNX_DOMAINS or len(n["input"]) != 2):
            return "output is not produced by a plain Gather"
        if (n["attrs"].get("axis") or 0) != 0:
            return "Gather axis is not 0"
        used.append(n)
        table = g["inits"].get(n["input"][0])
        if table is None:
            return "Gather data is not an initializer"

        # Gather indices <- [Cast | Identity]* <- graph input
        ids = n["input"][1]
        m = producer.get(ids)
        while (m is not None and m["op_type"] in ("Cast", "Identity")
               and m["domain"] in _ONNX_DOMAINS):
            used.append(m)
            ids = m["input"][0]
            m = producer.get(ids)
        if ids not in g["inputs"] or ids in g["inits"]:
            return "Gather indices are not a graph input"
        if len(used) != len(g["nodes"]):
            return "graph contains ops besides the lookup"

        if table["dtype"] not in cls._NP_DTYPES:
            return f"unsupported table dtype {table['dtype']}"
        if len(table["dims"]) != 2 or min(table["dims"]) <= 0:
            return f"table shape {table['dims']} is not 2-D"
        if table["typed_data"]:
            return "table is stored in typed fields, not raw bytes"
        return table


def _close_quietly(m):
    try:
        if hasattr(m, "close"):
            m.close()
    except (BufferError, ValueError):
        pass


# --------------------------------------------------------------------------
# 7d. Decoder: compute the vocab projection for the last position only.
#
# Prefill runs the whole prompt (~1,300 positions at the default pixel
# budget) through the decoder, and the exported graph projects EVERY
# position onto the ~103k vocab -- a 1,300 x 1,024 x 103k matmul plus a
# ~530 MB fp32 logits tensor, of which only the last row is ever read.
#
# The patch inserts Slice(hidden, starts=[-1], axes=[-2]) in front of the
# final MatMul / MatMulNBits, so the projection sees one row. The result is
# identical: each logits row depends only on its own hidden-state row.
# During per-token decode the sequence length is already 1, so nothing
# changes there. The patch is applied in memory (the file on disk -- and its
# checksum -- are untouched) and only when the graph has the expected shape;
# otherwise, or with TEXTSNAP_LOGITS_SLICE=0, the original graph is used.
# --------------------------------------------------------------------------
_SLICE_PASSTHROUGH_OPS = ("Identity", "Cast")


def _last_position_logits_model(path):
    """Return (patched_model_bytes, description) or (None, reason)."""
    mm = _map_file(path)
    if not mm:
        return None, "empty model file"
    buf = memoryview(mm)
    try:
        return _patch_last_position_logits(buf)
    finally:
        buf.release()
        _close_quietly(mm)


def _patch_last_position_logits(buf):
    gspan, grec, opsets = _pb_parse_model(buf)
    onnx_opset = max(opsets.get("", 0), opsets.get("ai.onnx", 0))
    if onnx_opset < 11:
        return None, f"ONNX opset {onnx_opset} predates negative Slice axes"
    g = _pb_parse_graph(buf, gspan)
    if any("logits_to_keep" in i.lower() for i in g["inputs"]):
        return None, "graph already takes a logits_to_keep input"
    if g["sparse"]:
        return None, "graph has sparse initializers"
    if any(t["location"] == 1 for t in g["inits"].values()):
        return None, "graph stores weights in external data"
    logits = next((o for o in g["outputs"] if "logits" in o.lower()), None)
    if logits is None:
        return None, "no logits output"

    nodes = g["nodes"]
    producer, consumers = {}, {}
    for idx, (node, _, _) in enumerate(nodes):
        for o in node["output"]:
            producer[o] = idx
        for i in node["input"]:
            if i:
                consumers.setdefault(i, []).append(idx)

    # Walk back from the logits output through dtype/identity passthroughs
    # to the vocab projection.
    chain = []
    idx = producer.get(logits)
    while idx is not None:
        node = nodes[idx][0]
        if (node["op_type"] in _SLICE_PASSTHROUGH_OPS
                and node["domain"] in _ONNX_DOMAINS):
            chain.append(idx)
            idx = producer.get(node["input"][0])
            continue
        break
    if idx is None:
        return None, "logits are not produced by a node"
    lm, lm_rec, lm_span = nodes[idx]
    is_proj = ((lm["op_type"] == "MatMul" and lm["domain"] in _ONNX_DOMAINS)
               or (lm["op_type"] == "MatMulNBits"
                   and lm["domain"] == "com.microsoft"))
    if not is_proj:
        return None, (f"final projection is {lm['domain'] or 'ai.onnx'}:"
                      f"{lm['op_type']}, not MatMul/MatMulNBits")

    # Every tensor from the projection to the logits output must feed only
    # the next node of that chain, so slicing cannot affect anything else.
    path = [idx] + chain[::-1]
    graph_outputs = set(g["outputs"])
    for k, nidx in enumerate(path):
        node = nodes[nidx][0]
        if len(node["output"]) != 1:
            return None, f"{node['op_type']} on the logits path has " \
                         f"several outputs"
        out = node["output"][0]
        users = consumers.get(out, [])
        if k + 1 < len(path):
            if users != [path[k + 1]] or out in graph_outputs:
                return None, "projection output is used elsewhere"
        elif out != logits or users:
            return None, "logits tensor is used inside the graph"

    hidden = lm["input"][0] if lm["input"] else ""
    if not hidden or hidden in g["inits"]:
        return None, "projection input is not an activation"

    taken = set(g["inits"]) | set(g["inputs"]) | set(producer)
    node_names = {n[0]["name"] for n in nodes}

    def fresh(base, used):
        name, k = base, 1
        while name in used:
            name, k = f"{base}_{k}", k + 1
        used.add(name)
        return name

    sliced = fresh("textsnap/last_position_hidden", taken)
    starts = fresh("textsnap/slice_starts", taken)
    ends = fresh("textsnap/slice_ends", taken)
    axes = fresh("textsnap/slice_axes", taken)
    slice_name = fresh("textsnap/last_position_slice", node_names)

    def int64_initializer(name, values):
        raw = np.asarray(values, dtype="<i8").tobytes()
        return _pb_enc_len(_G_INIT, b"".join([
            _pb_enc_int(_T_DIMS, len(values)),
            _pb_enc_int(_T_DTYPE, _ONNX_INT64),
            _pb_enc_len(_T_NAME, name.encode()),
            _pb_enc_len(_T_RAW, raw),
        ]))

    # Slice(hidden, [-1], [INT64_MAX], [-2]) -> last position only. Axis -2
    # is the sequence axis for both [B, S, H] and flattened [B*S, H] inputs
    # (batch is always 1 here).
    slice_node = _pb_enc_len(_G_NODE, b"".join(
        [_pb_enc_len(_N_INPUT, t.encode())
         for t in (hidden, starts, ends, axes)]
        + [_pb_enc_len(_N_OUTPUT, sliced.encode()),
           _pb_enc_len(_N_NAME, slice_name.encode()),
           _pb_enc_len(_N_OP, b"Slice")]))

    # The projection node, with its first input renamed to the sliced tensor.
    parts, replaced = [], False
    for f, wt, _, rs, re_ in _pb_fields(buf, *lm_span):
        if f == _N_INPUT and wt == _PB_LEN and not replaced:
            parts.append(_pb_enc_len(_N_INPUT, sliced.encode()))
            replaced = True
        else:
            parts.append(bytes(buf[rs:re_]))
    new_lm = _pb_enc_len(_G_NODE, b"".join(parts))

    extra_inits = (int64_initializer(starts, [-1])
                   + int64_initializer(ends, [(1 << 63) - 1])
                   + int64_initializer(axes, [-2]))

    # Reassemble. The Slice goes immediately before the projection so the
    # node list stays topologically sorted; the weights are copied once.
    gs, ge = gspan
    (lm_rs, lm_re), (g_rs, g_re) = lm_rec, grec
    graph_len = ((lm_rs - gs) + len(slice_node) + len(new_lm)
                 + (ge - lm_re) + len(extra_inits))
    model = b"".join([
        buf[:g_rs],
        _pb_enc_varint((_M_GRAPH << 3) | _PB_LEN), _pb_enc_varint(graph_len),
        buf[gs:lm_rs], slice_node, new_lm, buf[lm_re:ge], extra_inits,
        buf[g_re:],
    ])
    return model, (f"vocab projection {lm['op_type']} '{lm['name']}' "
                   f"restricted to the last position")


# --------------------------------------------------------------------------
# 8. (Position IDs are handled internally by the decoder graph -- the ONNX
#    export has no position_ids input, so no mRoPE construction is needed
#    here. The decoder derives positions from attention_mask + cache length.)
# --------------------------------------------------------------------------


# --------------------------------------------------------------------------
# 9. The OCR inference pipeline
#
# OCREngine loads the model once and can then recognize any number of
# images: batch mode pays the session-construction cost a single time
# instead of once per image.
# --------------------------------------------------------------------------
def _looping(seq, min_run=40, max_period=60):
    """Detect a greedy-decoding repetition loop.

    Three cheap checks:
      * a single token id repeated >= min_run times in a row;
      * a short cycle (period 2..12) repeating for >= min_run tokens;
      * a long block (period up to max_period) repeating >= 3 times --
        this catches sentence-level loops (e.g. the same line of a page
        emitted over and over) that the short-period scan cannot see.
    Greedy argmax on dense or low-quality input frequently falls into
    these; without this guard it runs to max_tokens every time.
    """
    if len(seq) < min_run:
        return False
    tail = seq[-min_run:]
    if len(set(tail)) == 1:
        return True
    for period in range(2, 13):
        if len(seq) < period * 6:
            continue
        window = seq[-period * 6:]
        if all(window[i] == window[i % period]
               for i in range(len(window))):
            return True
    # Long-period: does the last `period` tokens repeat >=3x back-to-back?
    for period in range(13, max_period + 1):
        if len(seq) < period * 3:
            continue
        window = seq[-period * 3:]
        if all(window[i] == window[i % period]
               for i in range(len(window))):
            return True
    return False


# --- no-repeat-ngram banning ----------------------------------------------
# Greedy decoding has no randomness, so once the model starts repeating
# an n-gram it will repeat it forever -- the run only ends at max_tokens
# (expensive) or when _looping() trips (after the loop already wasted
# many tokens). Banning is preventive: before committing a token, if it
# would complete an n-gram that already occurred, we forbid it and take
# the next-best token instead. This is the standard no_repeat_ngram_size
# from HF generate(). It stops a runaway at the FIRST repeat, not the
# 50th, which is the single biggest wall-time win on hard inputs.
NO_REPEAT_NGRAM = 4


def _banned_next_tokens(seq):
    """Token ids that would complete a previously-seen NO_REPEAT_NGRAM-gram
    if appended to `seq`. Returns a set (usually empty or tiny)."""
    n = NO_REPEAT_NGRAM
    if len(seq) < n - 1:
        return ()
    prefix = tuple(seq[-(n - 1):])
    banned = set()
    # Scan all prior n-grams; small n + a few thousand tokens is cheap.
    for i in range(len(seq) - n + 1):
        if tuple(seq[i:i + n - 1]) == prefix:
            banned.add(seq[i + n - 1])
    return banned


def _pick_next(logit_row, seq):
    """argmax over the vocab, with previously-seen n-grams masked out."""
    banned = _banned_next_tokens(seq)
    if not banned:
        return int(np.argmax(logit_row))
    # Copy only when we actually need to mutate -- keeps the common
    # (no-ban) path allocation-free.
    row = logit_row.copy()
    for t in banned:
        if 0 <= t < row.shape[0]:
            row[t] = -np.inf
    return int(np.argmax(row))


class OCREngine:
    """The three ONNX components plus tokenizer, loaded once.

    engine = OCREngine(model_dir)
    for img in images:
        text = engine.recognize(img)
    """

    def __init__(self, model_dir, vision=DEFAULT_VISION):
        import time
        t0 = time.time()
        model_dir = Path(model_dir)
        model_files(vision)                      # validates the variant
        self.vision_variant = vision
        self.tok = Tokenizer.from_file(str(model_dir / TOKENIZER_FILE))
        self.last_stats = {}

        vis_path = model_dir / VISION_VARIANTS[vision][0]
        if not vis_path.is_file():
            raise SystemExit(f"[textsnap] vision encoder '{vision}' not found "
                             f"at {vis_path}.")
        log(f"[textsnap] Loading ONNX sessions (vision encoder: {vision})...")
        self.vis = make_session(vis_path, role="vision")
        self._init_vision_io()
        self._init_embedding(model_dir / "onnx" / "embedding.onnx")
        self._init_decoder(model_dir / "onnx" / "decoder_q4.onnx")
        self._init_prompt()
        self.load_seconds = time.time() - t0
        log(f"[textsnap] Model loaded in {self.load_seconds:.1f}s.")

    # ---- setup -----------------------------------------------------------
    def _init_vision_io(self):
        vin = [i.name for i in self.vis.get_inputs()]
        self._pv_name = _find(vin, "pixel") or vin[0]
        self._grid_name = _find(vin, "grid")
        self._pv_shape = next(i for i in self.vis.get_inputs()
                              if i.name == self._pv_name).shape

    def _init_embedding(self, emb_path):
        self.embed_table = None
        self.emb_session = None
        if os.environ.get("TEXTSNAP_EMBED", "").lower() == "ort":
            why = "TEXTSNAP_EMBED=ort"
        else:
            try:
                self.embed_table, why = EmbeddingTable.from_onnx(emb_path)
            except (OSError, ValueError, IndexError, KeyError) as e:
                why = f"could not read graph ({type(e).__name__}: {e})"
            t = self.embed_table
            if t is not None and t.hidden != HIDDEN_SIZE:
                why = f"table width {t.hidden} != hidden size {HIDDEN_SIZE}"
                self.embed_table = None
        if self.embed_table is not None:
            t = self.embed_table
            log(f"[textsnap] Embeddings: memory-mapped {t.vocab}x{t.hidden} "
                f"{t.dtype_name} table (no ORT session).")
            return
        log(f"[textsnap] Embeddings: ORT session ({why}).")
        self.emb_session = make_session(emb_path, role="embed")
        names = [i.name for i in self.emb_session.get_inputs()]
        self._emb_ids_name = _find(names, "input") or names[0]

    def _init_decoder(self, dec_path):
        dec = None
        why = "disabled by TEXTSNAP_LOGITS_SLICE=0"
        self.logits_sliced = False
        if os.environ.get("TEXTSNAP_LOGITS_SLICE", "1") != "0":
            try:
                patched, why = _last_position_logits_model(dec_path)
            except (OSError, ValueError, IndexError, KeyError) as e:
                patched, why = None, (f"could not read graph "
                                      f"({type(e).__name__}: {e})")
            if patched is not None:
                try:
                    dec = make_session(patched, role="decoder")
                    self.logits_sliced = True
                except Exception as e:  # ORT rejected the patched graph
                    why = f"patched graph rejected by ONNX Runtime ({e})"
                del patched
        if dec is None:
            dec = make_session(dec_path, role="decoder")
        self.dec = dec
        log(f"[textsnap] Prefill logits: "
            f"{'last position only' if self.logits_sliced else 'full'} "
            f"({why}).")

        dec_inputs = [i.name for i in dec.get_inputs()]
        dec_outputs = [o.name for o in dec.get_outputs()]
        self._dec_outputs = dec_outputs
        self._emb_name = (_find(dec_inputs, "inputs_embeds")
                          or _find(dec_inputs, "embed"))
        self._mask_name = _find(dec_inputs, "attention", "mask")
        self._logits_name = _find(dec_outputs, "logits") or dec_outputs[0]

        # If the export has a logits_to_keep input, it does natively what
        # the graph patch does: ask for one position.
        self._keep_name = _find(dec_inputs, "logits_to_keep")
        self._keep_value = None
        if self._keep_name:
            meta = next(i for i in dec.get_inputs()
                        if i.name == self._keep_name)
            self._keep_value = (np.array(1, dtype=np.int64) if not meta.shape
                                else np.array([1], dtype=np.int64))
            log(f"[textsnap] Using decoder input '{self._keep_name}' = 1.")

        # --- KV cache wiring -----------------------------------------------
        # The decoder declares 18 layers of past_key_values.{i}.{key,value}
        # as inputs and present.{i}.{key,value} as outputs. They MUST be
        # paired by numeric layer index. A plain lexical sort would order
        # them 0,1,10,11,...,17,2,3,... and feed layer 2's cache into layer
        # 10 -- a silent correctness bug that also wrecks performance. We
        # sort by the integer index parsed from the name.
        def _layer_idx(name):
            m = re.search(r"\.(\d+)\.", name)
            return int(m.group(1)) if m else -1

        self._past_names = sorted(
            [n for n in dec_inputs
             if "past" in n.lower() or "cache" in n.lower()],
            key=lambda n: (_layer_idx(n), "value" in n.lower()),
        )
        self._present_names = sorted(
            [n for n in dec_outputs if n.lower().startswith("present")
             or "present" in n.lower()],
            key=lambda n: (_layer_idx(n), "value" in n.lower()),
        )
        if len(self._past_names) != len(self._present_names):
            raise SystemExit("[textsnap] KV cache input/output count mismatch "
                             f"({len(self._past_names)} vs "
                             f"{len(self._present_names)})")
        # Map each present output -> the past input it feeds next step.
        self._present_to_past = dict(zip(self._present_names,
                                         self._past_names))

        # KV cache dtype from the declared input type.
        self._kv_dtype = np.float32
        for inp in dec.get_inputs():
            if inp.name in self._past_names and "float16" in inp.type:
                self._kv_dtype = np.float16
                break

    def _init_prompt(self):
        # Chat template:
        #   "<bos>User: <IMAGE_START><img placeholders><IMAGE_END>{prompt}\n
        #    Assistant:\n"
        enc = lambda t: self.tok.encode(t, add_special_tokens=False).ids
        self._pre_ids = enc("<|begin_of_sentence|>User: ")
        self._img_open = enc("<|IMAGE_START|>")
        self._img_close = enc("<|IMAGE_END|>")
        self._suf_ids = enc(f"{OCR_PROMPT}\nAssistant:\n")

    # ---- building blocks ---------------------------------------------------
    def embed(self, ids):
        """int64 ids (1, S) -> float32 embeddings (1, S, HIDDEN_SIZE)."""
        if self.embed_table is not None:
            return self.embed_table.lookup(ids)
        out = self.emb_session.run(None, {self._emb_ids_name: ids})[0]
        return out if out.dtype == np.float32 else out.astype(np.float32)

    def _empty_cache(self):
        """Zero-length past KV tensors for the prefill pass."""
        return {
            name: np.zeros((1, NUM_KV_HEADS, 0, HEAD_DIM),
                           dtype=self._kv_dtype)
            for name in self._past_names
        }

    # --- IOBinding: bind I/O once, avoid marshaling the whole KV cache
    # through Python dicts on every single token. -------------------------
    # Without binding, each dec.run() copies ~2*NUM_LAYERS cache tensors IN
    # and the same number OUT, per token. IOBinding lets ORT keep the
    # present.* outputs on its own allocator and we hand those OrtValues
    # straight back as the next step's past.* inputs: zero-copy feedback.
    def _decoder_step(self, embeds, attn_mask, past):
        """past: dict name -> OrtValue (or ndarray, for the prefill seed).
        Returns (logits ndarray, new_past dict name->OrtValue).
        embeds must already be float32; attn_mask must already be int64.
        """
        io = self.dec.io_binding()
        io.bind_cpu_input(self._emb_name, embeds)
        io.bind_cpu_input(self._mask_name, attn_mask)
        if self._keep_name:
            io.bind_cpu_input(self._keep_name, self._keep_value)
        for name, val in past.items():
            if isinstance(val, np.ndarray):
                io.bind_cpu_input(name, val)
            else:
                # An OrtValue from the previous step -- bind it directly,
                # no copy back into Python.
                io.bind_ortvalue_input(name, val)
        # Let ORT allocate every output on its own CPU allocator so the
        # present.* tensors can be reused as next-step inputs without a copy.
        for oname in self._dec_outputs:
            io.bind_output(oname, "cpu")
        self.dec.run_with_iobinding(io)
        named = dict(zip(self._dec_outputs, io.get_outputs()))
        # logits is small (one position) and we need it on the host for
        # argmax -- materialize just this one.
        logits = named[self._logits_name].numpy()
        # present.* stay as OrtValues; re-key them to the past.* names they
        # feed next step. No numpy() call -> no copy.
        new_past = {self._present_to_past[p]: named[p]
                    for p in self._present_names}
        return logits, new_past

    # ---- the pipeline ------------------------------------------------------
    def recognize(self, img, max_tokens=2048, max_pixels=MAX_PIXELS):
        """OCR one PIL image. Returns the model's text (markdown).
        Per-run timings are left in self.last_stats."""
        import time
        stats = {"vision": self.vision_variant}

        # ---- preprocess image ----
        pixel_values, grid_thw = preprocess_image(img, max_pixels=max_pixels)
        grid_t, grid_h, grid_w = grid_thw

        # ---- vision encoder (run FIRST, before building the prompt) ------
        # The encoder includes the spatial-merge projector, so its output
        # row count IS the number of image tokens the decoder expects. We
        # must build the prompt with exactly that many IMAGE_TOKEN_ID
        # placeholders -- never predict the count and truncate, which
        # silently corrupts the visual features.
        pixel_values = fit_pixel_values(pixel_values, self._pv_shape)
        feed = {self._pv_name: pixel_values}
        if self._grid_name:
            feed[self._grid_name] = np.array([[grid_t, grid_h, grid_w]],
                                             dtype=np.int64)
        log(f"[textsnap] Running vision encoder ({self.vision_variant})... "
            f"(pixel_values shape {pixel_values.shape})")
        t_vis = time.time()
        image_embeds = self.vis.run(None, feed)[0]
        stats["vision_s"] = time.time() - t_vis
        image_embeds = image_embeds.reshape(-1, HIDDEN_SIZE).astype(
            np.float32)

        # The actual, authoritative image-token count.
        n_image_tokens = image_embeds.shape[0]
        predicted = (grid_t * grid_h * grid_w) // (MERGE_SIZE * MERGE_SIZE)
        log(f"[textsnap] Image tokens: {n_image_tokens} "
            f"(grid {grid_t}x{grid_h}x{grid_w}, predicted {predicted}; "
            f"encoder {stats['vision_s']:.1f}s)")
        stats["image_tokens"] = n_image_tokens

        # ---- build the prompt token sequence ------------------------------
        input_ids = np.array(
            [self._pre_ids + self._img_open
             + [IMAGE_TOKEN_ID] * n_image_tokens
             + self._img_close + self._suf_ids],
            dtype=np.int64,
        )
        seq_len = input_ids.shape[1]

        # ---- token embeddings ----
        inputs_embeds = self.embed(input_ids)            # (1, seq, hidden)

        # ---- splice image embeddings into the placeholder positions -------
        # By construction the counts match exactly -- assert, do not patch.
        mask = (input_ids[0] == IMAGE_TOKEN_ID)
        if int(mask.sum()) != image_embeds.shape[0]:
            raise SystemExit(
                f"[textsnap] internal error: placeholder count "
                f"{int(mask.sum())} != image embed rows "
                f"{image_embeds.shape[0]}"
            )
        inputs_embeds[0, mask, :] = image_embeds

        log(f"[textsnap] Decoding on {_PHYS_CORES} cores "
            f"(KV-cache enabled, IOBinding on, "
            f"{len(self._past_names)//2} layers, max {max_tokens} tokens)...")
        t0 = time.time()

        # The mask is all-ones and only grows; allocate it full-size once
        # and feed a contiguous slice each step.
        attn_buf = np.ones((1, seq_len + max_tokens), dtype=np.int64)

        # --- Prefill: process the full prompt once, populate the cache -----
        logits, past = self._decoder_step(inputs_embeds, attn_buf[:, :seq_len],
                                          self._empty_cache())
        next_id = int(np.argmax(logits[0, -1]))
        t_prefill = time.time() - t0
        stats["prefill_s"] = t_prefill
        log(f"[textsnap] Prefill done in {t_prefill:.1f}s "
            f"({seq_len} positions); generating...")

        # --- Decode: one token at a time, feeding back the growing cache ---
        generated = []
        total = seq_len
        stop_reason = "max_tokens"
        last_print = time.time()

        for step in range(max_tokens):
            if next_id == EOS_TOKEN_ID:
                stop_reason = "EOS"
                break
            generated.append(next_id)
            total += 1

            # Repetition guard -- backstop for any loop the n-gram ban
            # doesn't prevent. _looping() is O(n) in the generated length;
            # running it every token makes the decode O(n^2), so check
            # every 12th.
            if len(generated) % 12 == 0 and _looping(generated):
                stop_reason = "repetition loop"
                # Trim the looped tail so it doesn't pollute the output.
                while len(generated) > 1 and _looping(generated):
                    generated.pop()
                break

            # Embed only the single new token (a table row lookup).
            tok_embed = self.embed(np.array([[next_id]], dtype=np.int64))

            # attention_mask spans the full context so far (cached + new).
            logits, past = self._decoder_step(
                tok_embed, attn_buf[:, :total], past)
            # n-gram-aware token selection -- prevents greedy runaway loops.
            next_id = _pick_next(logits[0, -1], generated)

            # Live progress -- so a slow run is visibly alive, not hung.
            now = time.time()
            if now - last_print > 2.0:
                el = now - t0
                r = len(generated) / el if el > 0 else 0
                log(f"[textsnap]   ... {len(generated)} tokens "
                    f"({r:.1f} tok/s)", flush=True)
                last_print = now

        dt = time.time() - t0
        n = len(generated)
        rate = n / dt if dt > 0 else 0
        stats.update(decode_s=dt - t_prefill, tokens=n, stop=stop_reason)
        self.last_stats = stats
        log(f"[textsnap] Stopped: {stop_reason}. "
            f"{n} tokens in {dt:.1f}s ({rate:.1f} tok/s)")
        if stop_reason == "max_tokens":
            log("[textsnap] NOTE: hit the token cap -- output may be "
                "truncated. Raise --max-tokens if the page is very dense.")

        text = self.tok.decode(generated, skip_special_tokens=True)
        return text.strip()


def run_ocr(img, model_dir, max_tokens=2048, max_pixels=MAX_PIXELS,
            vision=DEFAULT_VISION):
    """One-shot convenience wrapper (loads the model for a single image).
    For several images, create one OCREngine and call recognize() on it."""
    return OCREngine(model_dir, vision=vision).recognize(
        img, max_tokens=max_tokens, max_pixels=max_pixels)


# --------------------------------------------------------------------------
# 9b. PP-OCRv6: text detection (DB) + line recognition (CTC) on ONNX Runtime
#
# A re-implementation of the slice of PaddleOCR's general OCR pipeline that
# textsnap needs (no doc unwarping, no orientation classifiers), so the
# `paddleocr` / `paddlepaddle` packages are not required. Pre/post-processing
# follows PaddleOCR 3.x: DetResizeForTest + NormalizeImage, DBPostProcess in
# "quad" mode, get_rotate_crop_image, OCRReisizeNormImg, CTCLabelDecode.
# Both models expect BGR input, as they were trained on cv2-decoded images.
# --------------------------------------------------------------------------
def _yaml_scalar(v):
    """Unquote one YAML plain/quoted scalar (the subset inference.yml uses)."""
    # Strip ASCII blanks only: str.strip() would also eat U+3000 (the
    # ideographic space), which is a real entry in PP-OCR dictionaries.
    v = v.strip(" \t\r\n")
    if len(v) >= 2 and v[0] == v[-1] == "'":
        return v[1:-1].replace("''", "'")
    if len(v) >= 2 and v[0] == v[-1] == '"':
        import json
        try:
            return json.loads(v)
        except ValueError:
            return v[1:-1].encode("utf-8").decode("unicode_escape")
    return v


def parse_inference_yml(text):
    """Pull what textsnap needs out of a PaddleX inference.yml without a YAML
    dependency: the PostProcess scalars and its character_dict list.
    Returns {"PostProcess": {key: str | list}}."""
    post, in_post, list_key = {}, False, None
    for raw in text.splitlines():
        stripped = raw.strip(" \t\r")
        if not stripped or stripped.startswith("#"):
            continue
        indent = len(raw) - len(raw.lstrip(" "))
        if indent == 0:
            in_post = stripped == "PostProcess:"
            list_key = None
            continue
        if not in_post:
            continue
        if list_key and (stripped == "-" or stripped.startswith("- ")):
            post[list_key].append(_yaml_scalar(stripped[2:]) if
                                  stripped != "-" else "")
            continue
        m = re.match(r"^([A-Za-z_][\w]*):(?:\s+(.*))?$", stripped)
        if m and indent <= 2:
            key, val = m.group(1), m.group(2)
            if val is None or val == "":
                post[key], list_key = [], key
            else:
                post[key], list_key = _yaml_scalar(val), None
    return {"PostProcess": post}


def _order_quad(pts):
    """cv2.boxPoints -> [top-left, top-right, bottom-right, bottom-left],
    exactly as PaddleOCR's get_mini_boxes orders them."""
    p = sorted(list(pts), key=lambda x: x[0])
    i1, i4 = (0, 1) if p[1][1] > p[0][1] else (1, 0)
    i2, i3 = (2, 3) if p[3][1] > p[2][1] else (3, 2)
    return np.array([p[i1], p[i2], p[i3], p[i4]], dtype=np.float32)


def _mini_box(points):
    import cv2
    rect = cv2.minAreaRect(np.asarray(points, dtype=np.float32))
    return _order_quad(cv2.boxPoints(rect)), min(rect[1])


def _box_score(pred, quad):
    """Mean probability inside the quad (PaddleOCR's box_score_fast)."""
    import cv2
    h, w = pred.shape
    xmin = int(np.clip(np.floor(quad[:, 0].min()), 0, w - 1))
    xmax = int(np.clip(np.ceil(quad[:, 0].max()), 0, w - 1))
    ymin = int(np.clip(np.floor(quad[:, 1].min()), 0, h - 1))
    ymax = int(np.clip(np.ceil(quad[:, 1].max()), 0, h - 1))
    mask = np.zeros((ymax - ymin + 1, xmax - xmin + 1), dtype=np.uint8)
    q = quad.copy()
    q[:, 0] -= xmin
    q[:, 1] -= ymin
    cv2.fillPoly(mask, q.reshape(1, -1, 2).astype(np.int32), 1)
    return cv2.mean(pred[ymin:ymax + 1, xmin:xmax + 1], mask)[0]


def _unclip_rect(quad, ratio):
    """Grow a rectangle by the DB 'unclip' distance d = area*ratio/perimeter.
    PaddleOCR offsets the polygon with pyclipper and then takes the minimum
    area rectangle of the result; for a rectangle input that is exactly the
    rectangle grown by d on every side, so no clipper library is needed."""
    u = quad[1] - quad[0]
    v = quad[3] - quad[0]
    w, h = float(np.linalg.norm(u)), float(np.linalg.norm(v))
    if w < 1e-6 or h < 1e-6:
        return quad
    d = (w * h) * ratio / (2 * (w + h))
    u, v = u / w, v / h
    c = quad.mean(axis=0)
    a, b = (w / 2 + d) * u, (h / 2 + d) * v
    return np.array([c - a - b, c + a - b, c + a + b, c - a + b],
                    dtype=np.float32)


def db_boxes(pred, orig_w, orig_h, thresh, box_thresh, unclip_ratio,
             max_candidates=1000, min_size=DET_MIN_BOX):
    """DBPostProcess ('quad' box type): probability map -> text quads in
    original-image pixels, each with its detection score."""
    import cv2
    h, w = pred.shape
    bitmap = (pred > thresh).astype(np.uint8) * 255
    contours = cv2.findContours(bitmap, cv2.RETR_LIST,
                                cv2.CHAIN_APPROX_SIMPLE)[-2]
    boxes, scores = [], []
    for contour in contours[:max_candidates]:
        quad, sside = _mini_box(contour.reshape(-1, 2))
        if sside < min_size:
            continue
        score = _box_score(pred, quad)
        if score < box_thresh:
            continue
        quad, sside = _mini_box(_unclip_rect(quad, unclip_ratio))
        if sside < min_size + 2:
            continue
        quad[:, 0] = np.clip(np.round(quad[:, 0] / w * orig_w), 0, orig_w)
        quad[:, 1] = np.clip(np.round(quad[:, 1] / h * orig_h), 0, orig_h)
        if (np.linalg.norm(quad[0] - quad[1]) <= 3
                or np.linalg.norm(quad[0] - quad[3]) <= 3):
            continue
        boxes.append(quad)
        scores.append(float(score))
    return boxes, scores


def det_resize(h, w, limit=DET_LIMIT_SIDE, max_side=DET_MAX_SIDE):
    """DetResizeForTest(limit_type='min') + max_side_limit, snapped to 32."""
    ratio = limit / min(h, w) if min(h, w) < limit else 1.0
    if max(h, w) * ratio > max_side:
        ratio = max_side / max(h, w)
    rh = max(int(round(h * ratio / 32) * 32), 32)
    rw = max(int(round(w * ratio / 32) * 32), 32)
    return rh, rw


def crop_quad(img, quad):
    """Perspective-crop one text quad to an upright strip
    (PaddleOCR's get_rotate_crop_image)."""
    import cv2
    cw = int(max(np.linalg.norm(quad[0] - quad[1]),
                 np.linalg.norm(quad[2] - quad[3])))
    ch = int(max(np.linalg.norm(quad[0] - quad[3]),
                 np.linalg.norm(quad[1] - quad[2])))
    cw, ch = max(cw, 1), max(ch, 1)
    dst = np.float32([[0, 0], [cw, 0], [cw, ch], [0, ch]])
    m = cv2.getPerspectiveTransform(quad.astype(np.float32), dst)
    out = cv2.warpPerspective(img, m, (cw, ch),
                              borderMode=cv2.BORDER_REPLICATE,
                              flags=cv2.INTER_CUBIC)
    if out.shape[0] / out.shape[1] >= 1.5:          # vertical line
        out = np.rot90(out)
    return out


def ctc_decode(probs, charset):
    """Greedy CTC: collapse repeats, drop blanks (index 0). Returns
    [(text, score)], score = mean max-probability of the kept characters
    (0.0 for an empty line), as in PaddleOCR's CTCLabelDecode."""
    idx = probs.argmax(axis=-1)
    prob = probs.max(axis=-1)
    out = []
    for row_i, row_p in zip(idx, prob):
        keep = np.ones(len(row_i), dtype=bool)
        keep[1:] = row_i[1:] != row_i[:-1]
        keep &= row_i != 0
        chars = [charset[k] for k in row_i[keep] if k < len(charset)]
        out.append(("".join(chars),
                    float(row_p[keep].mean()) if keep.any() else 0.0))
    return out


def assemble_lines(boxes, texts):
    """Reading-order plaintext from recognized quads: group quads into rows
    by vertical overlap, left-to-right inside a row, and a blank line where
    the vertical gap suggests a new paragraph."""
    items = [(b, t) for b, t in zip(boxes, texts) if t.strip()]
    if not items:
        return ""
    items.sort(key=lambda it: (it[0][:, 1].mean(), it[0][:, 0].min()))
    rows = []                     # [top, bottom, [(xmin, text)]]
    for b, t in items:
        top, bot = b[:, 1].min(), b[:, 1].max()
        hgt = max(bot - top, 1.0)
        for r in rows[-3:]:       # rows are appended top-down; look back a bit
            ov = min(bot, r[1]) - max(top, r[0])
            if ov > 0.5 * min(hgt, r[1] - r[0]):
                r[0], r[1] = min(r[0], top), max(r[1], bot)
                r[2].append((b[:, 0].min(), t.strip()))
                break
        else:
            rows.append([top, bot, [(b[:, 0].min(), t.strip())]])
    rows.sort(key=lambda r: r[0])
    heights = sorted(r[1] - r[0] for r in rows)
    med = heights[len(heights) // 2] or 1.0
    lines, prev_bot = [], None
    for top, bot, cells in rows:
        if prev_bot is not None and top - prev_bot > 0.9 * med:
            lines.append("")
        lines.append(" ".join(t for _, t in sorted(cells)))
        prev_bot = bot
    return "\n".join(lines)


class PPOCREngine:
    """PP-OCRv6 medium det + rec, loaded once.

    engine = PPOCREngine(ppocr_dir)
    res = engine.recognize(img)   # {'rec_texts', 'rec_scores', 'rec_polys',
                                  #  'text'}
    """

    def __init__(self, ppocr_dir):
        import time
        t0 = time.time()
        d = Path(ppocr_dir)
        self.det = make_session(d / "det" / "inference.onnx", role="vision")
        self.rec = make_session(d / "rec" / "inference.onnx", role="vision")
        det_post = parse_inference_yml(
            (d / "det" / "inference.yml").read_text(encoding="utf-8")
        )["PostProcess"]
        self.det_params = {}
        for k, dflt in DET_DEFAULTS.items():
            try:
                self.det_params[k] = type(dflt)(det_post.get(k, dflt))
            except (TypeError, ValueError):
                self.det_params[k] = dflt
        rec_post = parse_inference_yml(
            (d / "rec" / "inference.yml").read_text(encoding="utf-8")
        )["PostProcess"]
        chars = rec_post.get("character_dict")
        if not isinstance(chars, list) or not chars:
            raise SystemExit("[textsnap] PP-OCRv6 rec/inference.yml has no "
                             "character_dict.")
        self._chars = chars
        self._charset = None          # fixed on first run (needs out dim)
        self._det_in = self.det.get_inputs()[0].name
        self._rec_in = self.rec.get_inputs()[0].name
        out_dim = self.rec.get_outputs()[0].shape[-1]
        if isinstance(out_dim, int):
            self._set_charset(out_dim)
        self.last_stats = {}
        self.load_seconds = time.time() - t0
        log(f"[textsnap] PP-OCRv6 loaded in {self.load_seconds:.1f}s "
            f"(det {self.det_params}, {len(chars)} characters).")

    def _set_charset(self, n_classes):
        # CTCLabelDecode: index 0 is the blank, then the dictionary, then a
        # space when use_space_char is on (it is for PP-OCR rec models).
        cs = ["<blank>"] + list(self._chars)
        if n_classes == len(cs) + 1:
            cs.append(" ")
        if n_classes != len(cs):
            raise SystemExit(
                f"[textsnap] PP-OCRv6 rec model has {n_classes} classes but "
                f"its dictionary implies {len(cs)} -- mismatched files?")
        self._charset = cs

    # ---- detection ---------------------------------------------------------
    def detect(self, bgr):
        import cv2
        h, w = bgr.shape[:2]
        rh, rw = det_resize(h, w)
        x = cv2.resize(bgr, (rw, rh)).astype(np.float32) / 255.0
        x = (x - DET_MEAN) / DET_STD
        x = x.transpose(2, 0, 1)[np.newaxis].astype(np.float32)
        pred = self.det.run(None, {self._det_in: x})[0]
        pred = pred.reshape(pred.shape[-2], pred.shape[-1])
        p = self.det_params
        return db_boxes(pred, w, h, p["thresh"], p["box_thresh"],
                        p["unclip_ratio"], int(p["max_candidates"]))

    # ---- recognition -------------------------------------------------------
    def _rec_batch(self, crops):
        import cv2
        max_ratio = max([REC_MIN_WIDTH / REC_HEIGHT]
                        + [c.shape[1] / max(c.shape[0], 1) for c in crops])
        width = min(int(REC_HEIGHT * max_ratio), REC_MAX_WIDTH)
        batch = np.zeros((len(crops), 3, REC_HEIGHT, width), np.float32)
        for i, c in enumerate(crops):
            rw = min(width, int(np.ceil(REC_HEIGHT * c.shape[1]
                                        / max(c.shape[0], 1))))
            rw = max(rw, 1)
            r = cv2.resize(c, (rw, REC_HEIGHT)).astype(np.float32)
            batch[i, :, :, :rw] = (r.transpose(2, 0, 1) / 255.0 - 0.5) / 0.5
        out = self.rec.run(None, {self._rec_in: batch})[0]
        if self._charset is None:
            self._set_charset(out.shape[-1])
        # The export ends in softmax; guard against one that doesn't.
        if out.min() < 0 or not np.allclose(out.sum(-1), 1.0, atol=1e-2):
            out = np.exp(out - out.max(-1, keepdims=True))
            out /= out.sum(-1, keepdims=True)
        return ctc_decode(out, self._charset)

    def recognize_crops(self, crops):
        """[(text, score)] for BGR line crops, in input order."""
        order = sorted(range(len(crops)),
                       key=lambda i: crops[i].shape[1] / max(crops[i].shape[0], 1))
        res = [None] * len(crops)
        for s in range(0, len(order), REC_BATCH):
            ids = order[s:s + REC_BATCH]
            for i, r in zip(ids, self._rec_batch([crops[i] for i in ids])):
                res[i] = r
        return res

    def recognize(self, img):
        """OCR one PIL image. Returns a dict in the shape of a PaddleOCR
        page result (rec_texts / rec_scores / rec_polys, aligned) plus
        'text', the assembled plaintext. Timings go to self.last_stats."""
        import time
        bgr = np.ascontiguousarray(np.asarray(img.convert("RGB"))[:, :, ::-1])
        t0 = time.time()
        boxes, _ = self.detect(bgr)
        t1 = time.time()
        rec = self.recognize_crops([crop_quad(bgr, b) for b in boxes])
        t2 = time.time()
        texts = [t for t, _ in rec]
        scores = [s for _, s in rec]
        self.last_stats = {"det_s": t1 - t0, "rec_s": t2 - t1,
                           "lines": len(boxes)}
        log(f"[textsnap] PP-OCRv6: {len(boxes)} lines "
            f"(det {t1 - t0:.2f}s, rec {t2 - t1:.2f}s).")
        return {"rec_texts": texts, "rec_scores": scores, "rec_polys": boxes,
                "text": assemble_lines(boxes, texts)}


# --------------------------------------------------------------------------
# 9c. Quality gate: is the PP-OCRv6 page good enough, or ask the VLM?
#
# The recognizer's line score is roughly the mean of per-character max
# probabilities, so it can be confident about garbage (a stamp, a table
# border read as '|||', a signature read as 'llIl1') and is least reliable
# on very short lines. It is therefore one signal among several cheap
# linguistic ones, combined per line and then weighted by character count
# per page, so one bad header can't sink an otherwise clean page.
# --------------------------------------------------------------------------
_CJK = re.compile(r"[\u3040-\u30ff\u3400-\u9fff\uac00-\ud7af\uf900-\ufaff]")
_WORD = re.compile(r"[^\W\d_]{2,}")
_REPEAT = re.compile(r"(.)\1{4,}")
_STROKES = re.compile(r"^[lI1|!ǀ\s]+$")
_ALLOWED_INNER = set("-./:,'’_%&+@#")


def _zipf_lookup():
    """wordfreq.zipf_frequency, memoized; None if wordfreq is unavailable
    (the dictionary check is then skipped, the others still run)."""
    try:
        from wordfreq import zipf_frequency
    except Exception:            # ImportError, or its data failing to load
        log("[textsnap] wordfreq unavailable -- dictionary check skipped.")
        return None
    from functools import lru_cache
    return lru_cache(maxsize=65536)(zipf_frequency)


_ZIPF = False                    # sentinel: not looked up yet


def _is_word(w, langs):
    global _ZIPF
    if _ZIPF is False:
        _ZIPF = _zipf_lookup()
    if _ZIPF is None:
        return True
    lw = w.lower()
    return any(_ZIPF(lw, lang) > 1.5 for lang in langs)


def _jumbled(tok):
    """Letters, digits and odd symbols mixed inside one token ('a1#b')."""
    t = tok.strip(".,;:!?()[]{}\"'«»“”‘’")
    if len(t) < 3:
        return False
    has_l = any(c.isalpha() for c in t)
    has_d = any(c.isdigit() for c in t)
    odd = any(not c.isalnum() and c not in _ALLOWED_INNER for c in t)
    return has_l and has_d and odd


def line_ok(text, score, langs=("en",), min_score=0.80):
    t = text.strip()
    if len(t) < 2 or score < min_score:
        return False
    non_alnum = sum(not (c.isalnum() or c.isspace()) for c in t) / len(t)
    if non_alnum > 0.35 or _REPEAT.search(t) or _STROKES.match(t):
        return False
    toks = t.split()
    if len(toks) >= 2 and sum(map(_jumbled, toks)) * 2 >= len(toks):
        return False
    # Dictionary hit rate, on words wordfreq can judge without a segmenter.
    words = [w for w in _WORD.findall(t) if not _CJK.search(w)]
    if len(words) >= 3:
        hits = sum(_is_word(w, langs) for w in words) / len(words)
        if hits < 0.5:
            return False
    return True


def page_quality(res, langs=("en",)):
    """Share of characters (0..1) in lines that pass line_ok(). Empty
    detections (text '' and score 0.0) are ignored; a page with no text at
    all scores 0, so it goes to the VLM for a second look."""
    pairs = [(t, s) for t, s in zip(res["rec_texts"], res["rec_scores"])
             if t.strip()]
    total = sum(len(t) for t, _ in pairs) or 1
    good = sum(len(t) for t, s in pairs if line_ok(t, s, langs))
    return good / total


# --------------------------------------------------------------------------
# 10. Output formatting
# --------------------------------------------------------------------------
def to_plaintext(md):
    """Lightweight markdown -> plain text reduction."""
    t = md
    t = re.sub(r"!\[[^\]]*\]\([^)]*\)", "", t)          # images
    t = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", t)      # links -> text
    t = re.sub(r"`{1,3}([^`]*)`{1,3}", r"\1", t)        # code spans
    t = re.sub(r"^#{1,6}\s*", "", t, flags=re.M)        # headings
    t = re.sub(r"(\*\*|__|\*|_)", "", t)                # bold/italic
    t = re.sub(r"^\s*[-*+]\s+", "", t, flags=re.M)      # bullet markers
    t = re.sub(r"^\s*>\s?", "", t, flags=re.M)          # blockquotes
    t = re.sub(r"^\s*\|", "", t, flags=re.M)            # table pipes (leading)
    t = re.sub(r"\|", " ", t)                           # remaining pipes
    t = re.sub(r"^\s*[-:|\s]+\s*$", "", t, flags=re.M)  # table rule rows
    t = re.sub(r"\n{3,}", "\n\n", t)
    return t.strip()


# --------------------------------------------------------------------------
# 11. main
# --------------------------------------------------------------------------
def generate_checksums(dest=None):
    """Download every pinned model file (both repos, all vision variants)
    and write a fresh model_checksums.sha256.

    Used to regenerate the manifest after deliberately moving to a new model
    revision. Writes next to this module by default. Note: this fetches the
    whole VLM (~1.3 GB incl. both vision encoders) as well as PP-OCRv6.
    """
    from huggingface_hub import hf_hub_download

    if not _HEX40.match(PPOCR_HF_REVISION):
        raise SystemExit("[textsnap] PPOCR_HF_REVISION is not pinned yet; "
                         "pin it in textsnap.py first.")
    vl_files = []
    for f in ([f for v in VISION_VARIANTS.values() for f in v]
              + DECODER_FILES + EMBEDDING_FILES + AUX_FILES):
        if f not in vl_files:
            vl_files.append(f)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    lines = [
        "# textsnap model checksums -- SHA-256 of every downloaded model file.",
        "# Regenerate with: textsnap --generate-checksums",
    ]
    for repo, rev, files, local in (
            (PPOCR_HF_REPO, PPOCR_HF_REVISION, PPOCR_FILES,
             CACHE_DIR / PPOCR_SUBDIR),
            (HF_REPO, HF_REVISION, vl_files, CACHE_DIR)):
        lines.append(f"# {repo} @ {rev}")
        for fname in files:
            path = hf_hub_download(repo_id=repo, filename=fname, revision=rev,
                                   local_dir=str(local))
            digest = _sha256_file(path)
            lines.append(f"{digest}  {fname}")
            print(f"{digest}  {fname}", file=sys.stderr)

    if dest is None:
        dest = Path(__file__).resolve().parent / CHECKSUM_MANIFEST
    dest = Path(dest)
    dest.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(dest)


def load_input(arg):
    """Resolve one positional argument (None = clipboard) to
    (kind, PIL image, output-name stem)."""
    kind, value = detect_input(arg)
    log(f"[textsnap] Input type: {kind}")
    if kind == "clipboard":
        img, stem = load_from_clipboard()
    elif kind == "file":
        img, stem = load_from_file(value)
    elif kind == "image_url":
        img, stem = load_from_image_url(value)
    elif kind == "html_url":
        img, stem = load_from_html_url(value)
    else:
        raise SystemExit("[textsnap] unreachable input kind")
    log(f"[textsnap] Source image: {img.size[0]}x{img.size[1]}")
    # NOTE: do NOT pre-shrink here. preprocess_image() runs smart_resize(),
    # which bounds the image to MAX_PIXELS and snaps to the patch grid. An
    # extra cap on top of that just discards resolution the model could have
    # used -- and for text-dense screenshots, too-low resolution makes this
    # VLM hallucinate confident garbage rather than degrade gracefully.
    return kind, img, stem


def _expand_inputs(args):
    """Expand glob patterns the shell left alone (Windows cmd/PowerShell
    don't expand `*.png`). Anything that isn't a pattern, is a URL, or
    matches nothing is passed through so detect_input() reports it."""
    import glob
    out = []
    for a in args:
        is_pattern = any(c in a for c in "*?[")
        if (not is_pattern or urlparse(a).scheme in ("http", "https")
                or Path(a).exists()):
            out.append(a)
            continue
        matches = sorted(m for m in glob.glob(a, recursive=True)
                         if Path(m).is_file())
        out.extend(matches if matches else [a])
    return out


def _output_path(stem, output, single, used):
    """Where to write one result. `used` holds paths already written in this
    run, so two inputs with the same stem don't overwrite each other."""
    if output and single and not Path(output).is_dir():
        path = Path(output)                       # exact file, as before
        path.parent.mkdir(parents=True, exist_ok=True)
    else:
        out_dir = Path(output) if output else Path.cwd() / "textsnaps"
        out_dir.mkdir(parents=True, exist_ok=True)
        path, k = out_dir / f"{stem}_ocr.txt", 2
        while path in used:
            path, k = out_dir / f"{stem}_ocr_{k}.txt", k + 1
    used.add(path)
    return path


def _report_failure(label, e, failures):
    """Batch mode: one line on stderr per failed input (shown even without
    -v, since it changes the result), then carry on."""
    msg = (str(e) or type(e).__name__).removeprefix("[textsnap] ")
    if not isinstance(e, SystemExit):
        msg = f"{type(e).__name__}: {msg}"
    print(f"[textsnap] FAILED {label}: {msg}", file=sys.stderr)
    failures.append(label)


class _Engines:
    """Loads each engine lazily, at most once per run, so a batch pays for
    a model only if some input actually needs it."""

    def __init__(self, args):
        self.args = args
        self._ppocr = None
        self._vlm = None
        self.vlm_failed = None       # reason, once a fallback load failed

    def ppocr(self):
        if self._ppocr is None:
            a = self.args
            self._ppocr = PPOCREngine(get_ppocr_dir(a.model_dir,
                                                    verify=not a.no_verify))
        return self._ppocr

    def vlm(self, announce=False):
        if self._vlm is None:
            a = self.args
            if announce and not vlm_available(a.model_dir, a.vision):
                # A silent 1.1 GB download mid-batch would look like a hang.
                print("[textsnap] Low-confidence page: fetching "
                      "PaddleOCR-VL (~1.1 GB, one time) for a second read. "
                      "--min-quality 0 disables this.", file=sys.stderr)
            model_dir = get_model_dir(a.model_dir, verify=not a.no_verify,
                                      vision=a.vision)
            self._vlm = OCREngine(model_dir, vision=a.vision)
        return self._vlm

    def run_vlm(self, img):
        a = self.args
        return self._vlm.recognize(img, max_tokens=a.max_tokens,
                                   max_pixels=a.max_pixels)


def recognize_one(img, eng, args, langs):
    """The per-image policy. Returns (text, engine_name).

    --structured: VLM markdown (flattened only with --plaintext).
    default:      PP-OCRv6; if its page quality < --min-quality, the VLM
                  re-reads the image and its output is flattened to plain
                  text, so the default mode's format doesn't depend on
                  which engine answered.
    """
    if args.structured:
        eng.vlm()
        text = eng.run_vlm(img)
        return (to_plaintext(text) if args.plaintext else text), "vlm"

    res = eng.ppocr().recognize(img)
    q = page_quality(res, langs)
    log(f"[textsnap] Page quality {q:.2f} (threshold {args.min_quality:.2f}).")
    if q >= args.min_quality or args.min_quality <= 0:
        return res["text"], "ppocr"
    if eng.vlm_failed:
        return res["text"], "ppocr"
    try:
        eng.vlm(announce=True)
    except (Exception, SystemExit) as e:
        # Keep the PP-OCRv6 text rather than failing the input: the VLM is a
        # second opinion, not a requirement. Say so once, then stop trying.
        eng.vlm_failed = str(e) or type(e).__name__
        print(f"[textsnap] PaddleOCR-VL unavailable ({eng.vlm_failed}); "
              f"keeping PP-OCRv6 output for low-quality pages.",
              file=sys.stderr)
        return res["text"], "ppocr"
    log("[textsnap] Falling back to PaddleOCR-VL.")
    return to_plaintext(eng.run_vlm(img)), "vlm"


def _quality_arg(v):
    try:
        q = float(v)
    except ValueError:
        raise argparse.ArgumentTypeError(f"not a number: {v!r}")
    if not 0.0 <= q <= 1.0:
        raise argparse.ArgumentTypeError("must be between 0 and 1")
    return q


def main():
    ap = argparse.ArgumentParser(
        prog="textsnap",
        description="Lean CPU OCR: PP-OCRv6 medium, with PaddleOCR-VL-1.5 "
                    "for low-confidence pages and --structured output.",
    )
    ap.add_argument("inputs", nargs="*", metavar="input",
                    help="image file(s), image URL(s), or webpage URL(s). "
                         "Several inputs are processed in one run with the "
                         "models loaded once. Omit to read from the "
                         "clipboard.")
    ap.add_argument("-o", "--output", default=None,
                    help="output .txt path (single input) or directory. "
                         "Default: ./textsnaps/<name>_ocr.txt.")
    ap.add_argument("-v", "--verbose", action="store_true",
                    help="print progress diagnostics to stderr. "
                         "By default only the output path is printed.")
    ap.add_argument("--version", action="version",
                    version=f"textsnap {__version__}")
    ap.add_argument("--structured", "--markdown", action="store_true",
                    dest="structured",
                    help="skip PP-OCRv6 and run PaddleOCR-VL directly, "
                         "keeping its native markdown (tables, headings).")
    ap.add_argument("--plaintext", action="store_true",
                    help="with --structured: flatten the markdown to plain "
                         "text. (Default output is already plain text.)")
    ap.add_argument("--min-quality", type=_quality_arg,
                    default=_quality_arg(os.environ.get(
                        "TEXTSNAP_MIN_QUALITY", DEFAULT_MIN_QUALITY)),
                    help=f"page-quality threshold (0..1) below which "
                         f"PaddleOCR-VL re-reads the image (default "
                         f"{DEFAULT_MIN_QUALITY}; env TEXTSNAP_MIN_QUALITY). "
                         f"0 never falls back.")
    ap.add_argument("--lang", default=os.environ.get("TEXTSNAP_LANG",
                                                     DEFAULT_LANGS),
                    help="comma-separated wordfreq language codes used by "
                         "the quality check's dictionary test (default "
                         f"'{DEFAULT_LANGS}'; env TEXTSNAP_LANG), e.g. en,fr.")
    ap.add_argument("--model-dir", default=None,
                    help="use model files from this directory (ppocr/det, "
                         "ppocr/rec, and onnx/* + tokenizer.json for the "
                         "VLM). If omitted: files next to the script "
                         "(portable mode), else the OS cache.")
    ap.add_argument("--max-tokens", type=int, default=2048,
                    help="VLM: max generated tokens (default 2048).")
    ap.add_argument("--max-pixels", type=int, default=MAX_PIXELS,
                    help=f"VLM: image pixel budget fed to the vision encoder "
                         f"(default {MAX_PIXELS}). Lower = faster but less "
                         f"accurate; too low makes the model hallucinate. "
                         f"The image is only ever shrunk, never enlarged.")
    ap.add_argument("--vision", choices=list(VISION_VARIANTS),
                    default=os.environ.get("TEXTSNAP_VISION", DEFAULT_VISION),
                    help=f"VLM vision-encoder variant (default "
                         f"{DEFAULT_VISION}; env TEXTSNAP_VISION). q4 is a "
                         f"smaller download (231 vs 443 MB) but slower here.")
    ap.add_argument("--no-verify", action="store_true",
                    help="skip SHA-256 verification of downloaded model files "
                         "(not recommended).")
    ap.add_argument("--generate-checksums", action="store_true",
                    help="download the pinned model files, write a fresh "
                         "model_checksums.sha256, and exit.")
    args = ap.parse_args()

    global VERBOSE
    VERBOSE = args.verbose

    if args.generate_checksums:
        VERBOSE = True
        generate_checksums()
        return

    if args.vision not in VISION_VARIANTS:     # bad TEXTSNAP_VISION value
        ap.error(f"invalid vision variant '{args.vision}' "
                 f"(choose from {', '.join(VISION_VARIANTS)})")
    langs = tuple(x.strip() for x in args.lang.split(",") if x.strip()) \
        or (DEFAULT_LANGS,)

    inputs = _expand_inputs(args.inputs) if args.inputs else [None]
    single = len(inputs) == 1
    if (not single and args.output and Path(args.output).suffix
            and not Path(args.output).is_dir()):
        ap.error("with several inputs, -o must name a directory")

    eng = _Engines(args)
    used, failures = set(), []
    for i, arg in enumerate(inputs, 1):
        label = arg if arg is not None else "clipboard"
        if not single:
            log(f"[textsnap] ({i}/{len(inputs)}) {label}")
        # One input: errors end the run. Several: a bad input is reported
        # and skipped, and the exit status is 1 at the end.
        # (KeyboardInterrupt is not an Exception, so Ctrl-C still stops.)
        try:
            kind, img, stem = load_input(arg)
        except (Exception, SystemExit) as e:
            if single:
                raise
            _report_failure(label, e, failures)
            continue

        # Models load lazily, once, after the first input loads -- a typo in
        # the only argument shouldn't cost a model load. A primary engine
        # that fails to load is fatal for the whole batch.
        if args.structured:
            eng.vlm()
        else:
            eng.ppocr()

        try:
            result, used_engine = recognize_one(img, eng, args, langs)
        except (Exception, SystemExit) as e:
            if single:
                raise
            _report_failure(label, e, failures)
            continue

        out_path = _output_path(stem, args.output, single, used)
        out_path.write_text(result, encoding="utf-8")
        log(f"[textsnap] Wrote {out_path}  ({len(result)} chars, "
            f"{used_engine})")

        # Clipboard-in -> clipboard-out: if the image came from the
        # clipboard, put the OCR text straight back so the user can paste it
        # immediately. The .txt file is still written (the stdout-path
        # contract is unchanged). Best-effort -- a failure never aborts.
        if kind == "clipboard":
            if copy_text_to_clipboard(result):
                log("[textsnap] OCR text copied back to the clipboard.")
            else:
                log("[textsnap] Could not copy to clipboard "
                    "(no pbcopy/clip/wl-copy/xclip/xsel found); "
                    "text saved to the file above.")

        # The one thing stdout is for: output paths, bare, one per line, as
        # each finishes -- so `OUT=$(textsnap x.png)`, `| xargs cat` and
        # `| while read f; ...` all work.
        print(out_path, flush=True)

    if failures:
        print(f"[textsnap] {len(failures)} of {len(inputs)} inputs failed.",
              file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
