#!/usr/bin/env python3
"""
textsnap.py - Lean CPU OCR using PaddleOCR-VL-1.5 ONNX (quantized).

Snap any image, screenshot, or webpage into plaintext. No GPU. No cloud.
One command.

Usage:
    textsnap                          # OCR image from clipboard
    textsnap path/to/img.jpg          # OCR a local image file
    textsnap https://.../x.png        # OCR a direct image URL
    textsnap https://example.com/page # OCR the biggest image on a webpage
    textsnap a.png b.png scans/*.jpg  # batch: models are loaded once

Options:
    --plaintext     Strip markdown -> plain text (default output is the model's
                    native markdown).
    -o, --output    Output .txt path. Default: ./textsnaps/<name>_ocr.txt.
    --model-dir DIR Use ONNX/config files from DIR instead of downloading.
    --max-tokens N  Cap generated tokens (default 2048).
    --max-pixels N  Image pixel budget for the vision encoder (default is the
                    model's max). Lower trades accuracy for speed.
    --vision V      Vision-encoder variant: q8 (default) or q4.

Output:
    Plaintext, UTF-8. Default location is ./textsnaps/ (created if missing)
    under the current working directory; override with -o (with several
    inputs, -o names a directory; one output path per line). The filename is
    "<name>_ocr.txt", where <name> is the image filename stem (for image
    inputs) or the webpage slug (for HTML inputs).

    When the input image comes from the clipboard (textsnap run with no
    arguments), the OCR text is ALSO copied back to the clipboard so it can
    be pasted immediately -- the .txt file is still written as well.

Model files:
    The 3 ONNX components (~1.1 GB) are auto-downloaded on first run and
    cached in ~/.cache/textsnap. The decoder uses the q4 variant; the vision
    encoder defaults to q8 (--vision q4 selects the smaller 4-bit build).
    The embedding table ships fp32 only and is memory-mapped rather than
    run through ONNX Runtime.

    Portable mode: if the model files are found next to this script
    (./onnx/* + ./tokenizer.json), they are used directly -- no
    download, no --model-dir flag, no setup. Copy the textsnap folder
    together with its model files to any machine and run it offline.
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
    r = requests.get(url, timeout=60, headers={"User-Agent": "textsnap/1.0"})
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
                         headers={"User-Agent": "textsnap/1.0"}).text
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
    """Download every pinned model file (all vision variants included) and
    write a fresh model_checksums.sha256.

    Used to regenerate the manifest after deliberately moving to a new model
    revision. Writes next to this module by default. Note: this fetches the
    optional q8 vision encoder too (443 MB beyond the default set).
    """
    from huggingface_hub import hf_hub_download

    files = []
    for f in ([f for v in VISION_VARIANTS.values() for f in v]
              + DECODER_FILES + EMBEDDING_FILES + AUX_FILES):
        if f not in files:
            files.append(f)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    lines = [
        f"# textsnap model checksums for {HF_REPO} @ {HF_REVISION}",
        "# Regenerate with: textsnap --generate-checksums",
    ]
    for fname in files:
        local = hf_hub_download(repo_id=HF_REPO, filename=fname,
                                revision=HF_REVISION, local_dir=str(CACHE_DIR))
        digest = _sha256_file(local)
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


def main():
    ap = argparse.ArgumentParser(
        prog="textsnap",
        description="Lean CPU OCR via PaddleOCR-VL-1.5 ONNX (quantized).",
    )
    ap.add_argument("inputs", nargs="*", metavar="input",
                    help="image file(s), image URL(s), or webpage URL(s). "
                         "Several inputs are processed in one run with the "
                         "model loaded once. Omit to read from the "
                         "clipboard.")
    ap.add_argument("-o", "--output", default=None,
                    help="output .txt path (single input) or directory. "
                         "Default: ./textsnaps/<name>_ocr.txt.")
    ap.add_argument("-v", "--verbose", action="store_true",
                    help="print progress diagnostics to stderr. "
                         "By default only the output path is printed.")
    ap.add_argument("--plaintext", action="store_true",
                    help="output plain text instead of native markdown.")
    ap.add_argument("--model-dir", default=None,
                    help="use ONNX/config files from this directory. "
                         "If omitted, textsnap uses model files found next "
                         "to the script (portable mode), else the OS cache "
                         "(downloading on first run).")
    ap.add_argument("--max-tokens", type=int, default=2048,
                    help="max generated tokens (default 2048).")
    ap.add_argument("--max-pixels", type=int, default=MAX_PIXELS,
                    help=f"image pixel budget fed to the vision encoder "
                         f"(default {MAX_PIXELS}). Lower = faster but less "
                         f"accurate; too low makes the model hallucinate. "
                         f"The image is only ever shrunk, never enlarged.")
    ap.add_argument("--vision", choices=list(VISION_VARIANTS),
                    default=os.environ.get("TEXTSNAP_VISION", DEFAULT_VISION),
                    help=f"vision-encoder variant (default {DEFAULT_VISION}; "
                         f"env TEXTSNAP_VISION). q4 is a smaller download "
                         f"(231 vs 443 MB) but slower here. "
                         f"Outputs differ slightly between variants.")
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

    inputs = _expand_inputs(args.inputs) if args.inputs else [None]
    single = len(inputs) == 1
    if (not single and args.output and Path(args.output).suffix
            and not Path(args.output).is_dir()):
        ap.error("with several inputs, -o must name a directory")

    engine = None
    used, failures = set(), []
    for i, arg in enumerate(inputs, 1):
        label = arg if arg is not None else "clipboard"
        if not single:
            log(f"[textsnap] ({i}/{len(inputs)}) {label}")
        # One input: errors end the run exactly as before. Several: a bad
        # input is reported and skipped, and the exit status is 1 at the end.
        # (KeyboardInterrupt is not an Exception, so Ctrl-C still stops.)
        try:
            kind, img, stem = load_input(arg)
        except (Exception, SystemExit) as e:
            if single:
                raise
            _report_failure(label, e, failures)
            continue

        # Load the model lazily, once, after the first input loads -- a typo
        # in the only argument shouldn't cost a model load. A model that
        # fails to load is fatal for the whole batch.
        if engine is None:
            model_dir = get_model_dir(args.model_dir,
                                      verify=not args.no_verify,
                                      vision=args.vision)
            engine = OCREngine(model_dir, vision=args.vision)

        try:
            result = engine.recognize(img, max_tokens=args.max_tokens,
                                      max_pixels=args.max_pixels)
        except (Exception, SystemExit) as e:
            if single:
                raise
            _report_failure(label, e, failures)
            continue

        if args.plaintext:
            result = to_plaintext(result)

        out_path = _output_path(stem, args.output, single, used)
        out_path.write_text(result, encoding="utf-8")
        log(f"[textsnap] Wrote {out_path}  ({len(result)} chars)")

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
