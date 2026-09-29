"""Offline tests for textsnap using small hand-built ONNX models that mirror
the real export's layout (external-data embedding table, MatMulNBits vocab
projection, past/present KV cache I/O). No network or model download needed.

    python test_textsnap.py

If a pre-OCREngine copy of the script is saved as textsnap_orig.py next to
this file, the engine is also checked token-for-token against it; otherwise
the optimized paths are checked against the unoptimized fallbacks."""
import sys, types, os, io, shutil, contextlib, tempfile
from pathlib import Path
import numpy as np

# ---- stub the deps that aren't installed (and aren't exercised here) -------
class _Enc:
    def __init__(self, ids): self.ids = ids
class Tokenizer:
    SPECIAL = {"<|IMAGE_START|>": 501, "<|IMAGE_END|>": 502}
    @classmethod
    def from_file(cls, p):
        assert Path(p).is_file(), p
        return cls()
    def encode(self, text, add_special_tokens=False):
        if text in self.SPECIAL:
            return _Enc([self.SPECIAL[text]])
        return _Enc([10 + (ord(c) % 400) for c in text])
    def decode(self, ids, skip_special_tokens=True):
        return "".join(chr(65 + i % 26) for i in ids)
tok = types.ModuleType("tokenizers"); tok.Tokenizer = Tokenizer
sys.modules["tokenizers"] = tok
sys.modules["huggingface_hub"] = types.ModuleType("huggingface_hub")
sys.modules["readability"] = types.ModuleType("readability")
# wordfreq stub: a tiny English lexicon is enough to exercise the gate.
_LEX = set("""the a an of to and in is it for on as with was day reminder
moon tells sky sea tide me quarterly report region revenue total page this
that be are from by at or not have has but your our more""".split())
wf = types.ModuleType("wordfreq")
wf.zipf_frequency = lambda w, lang: 5.0 if (lang == "en" and w in _LEX) else 0.0
sys.modules.setdefault("wordfreq", wf)

sys.path.insert(0, str(Path(__file__).parent))
import textsnap as ts
try:
    import textsnap_orig as orig
except ImportError:
    orig = None
import onnxruntime as ort
from PIL import Image

V, H, L = 512, 128, 3
for m in (ts, orig) if orig else (ts,):
    m.HIDDEN_SIZE, m.IMAGE_TOKEN_ID, m.EOS_TOKEN_ID = H, 500, 2

E, I = ts._pb_enc_len, ts._pb_enc_int
DT = {np.float32: 1, np.uint8: 2, np.int64: 7}

def tensor(name, arr):
    arr = np.ascontiguousarray(arr)
    return (b"".join(I(1, d) for d in arr.shape) + I(2, DT[arr.dtype.type])
            + E(8, name.encode()) + E(9, arr.tobytes()))

def tensor_ext(name, shape, location, offset, length):
    ext = b"".join(E(13, E(1, k.encode()) + E(2, v.encode())) for k, v in
                   (("location", location), ("offset", str(offset)),
                    ("length", str(length))))
    return (b"".join(I(1, d) for d in shape) + I(2, 1) + E(8, name.encode())
            + ext + I(14, 1))

def a_int(n, v): return E(1, n.encode()) + I(3, v) + I(20, 2)
def a_ints(n, vs): return E(1, n.encode()) + b"".join(I(8, v) for v in vs) + I(20, 7)

def node(op, ins, outs, name, attrs=(), domain=""):
    b = b"".join(E(1, x.encode()) for x in ins) + b"".join(E(2, x.encode()) for x in outs)
    b += E(3, name.encode()) + E(4, op.encode()) + b"".join(E(5, a) for a in attrs)
    return b + (E(7, domain.encode()) if domain else b"")

def vi(name, elem, dims):
    shape = b"".join(E(1, I(1, d) if isinstance(d, int) else E(2, d.encode())) for d in dims)
    return E(1, name.encode()) + E(2, E(1, I(1, elem) + E(2, shape)))

def model(nodes, inits, inputs, outputs, opsets=(("", 17), ("com.microsoft", 1))):
    g = (b"".join(E(1, n) for n in nodes) + E(2, b"g")
         + b"".join(E(5, t) for t in inits)
         + b"".join(E(11, x) for x in inputs) + b"".join(E(12, x) for x in outputs))
    return I(1, 8) + E(2, b"toy") + E(7, g) + b"".join(
        E(8, E(1, d.encode()) + I(2, v)) for d, v in opsets)

def matmulnbits(W, block=32):
    K, N = W.shape
    nb = K // block
    B = np.zeros((N, nb, block // 2), np.uint8)
    S = np.zeros((N, nb), np.float32)
    for n in range(N):
        for b in range(nb):
            w = W[b*block:(b+1)*block, n]
            sc = max(np.abs(w).max() / 7, 1e-8)
            q = np.clip(np.round(w / sc) + 8, 0, 15).astype(np.uint8)
            B[n, b] = q[0::2] | (q[1::2] << 4)
            S[n, b] = sc
    return B, S.reshape(-1)

rng = np.random.default_rng(0)

def build_decoder(lm="nbits", extra_logits_user=False, identity_chain=False):
    nodes, inits = [], []
    inits.append(tensor("kv_shape", np.array([1, -1, 2, 128], np.int64)))
    ins = [vi("inputs_embeds", 1, [1, "s", H]), vi("attention_mask", 7, [1, "t"])]
    outs = [vi("logits", 1, [1, "s", V])]
    for i in range(L):
        for kind in ("key", "value"):
            w = f"W{kind}{i}"
            inits.append(tensor(w, (rng.standard_normal((H, 256)) * .1).astype(np.float32)))
            nodes.append(node("MatMul", ["inputs_embeds", w], [f"{w}_mm"], f"{w}_mm"))
            nodes.append(node("Reshape", [f"{w}_mm", "kv_shape"], [f"{w}_r"], f"{w}_r"))
            nodes.append(node("Transpose", [f"{w}_r"], [f"{w}_t"], f"{w}_t", [a_ints("perm", [0, 2, 1, 3])]))
            nodes.append(node("Concat", [f"past_key_values.{i}.{kind}", f"{w}_t"],
                              [f"present.{i}.{kind}"], f"cat{i}{kind}", [a_int("axis", 2)]))
            ins.append(vi(f"past_key_values.{i}.{kind}", 1, [1, 2, "p", 128]))
            outs.append(vi(f"present.{i}.{kind}", 1, [1, 2, "t", 128]))
    inits.append(tensor("W1", (rng.standard_normal((H, H)) * .3).astype(np.float32)))
    nodes += [node("MatMul", ["inputs_embeds", "W1"], ["h0"], "h0"),
              node("ReduceMean", ["present.0.key"], ["m1"], "m1", [a_ints("axes", [1]), a_int("keepdims", 0)]),
              node("ReduceMean", ["m1"], ["m2"], "m2", [a_ints("axes", [1]), a_int("keepdims", 1)]),
              node("Add", ["h0", "m2"], ["h1"], "h1"),
              node("Tanh", ["h1"], ["hidden"], "final_norm")]
    Wlm = (rng.standard_normal((H, V)) * .5).astype(np.float32)
    lm_out = "lm_raw" if identity_chain else "logits"
    if lm == "nbits":
        B, S = matmulnbits(Wlm)
        inits += [tensor("lm_B", B), tensor("lm_S", S)]
        nodes.append(node("MatMulNBits", ["hidden", "lm_B", "lm_S"], [lm_out], "lm_head",
                          [a_int("K", H), a_int("N", V), a_int("bits", 4), a_int("block_size", 32)],
                          "com.microsoft"))
    else:
        inits.append(tensor("Wlm", Wlm))
        nodes.append(node("MatMul", ["hidden", "Wlm"], [lm_out], "lm_head"))
    if identity_chain:
        nodes.append(node("Identity", ["lm_raw"], ["logits"], "id"))
    if extra_logits_user:
        nodes.append(node("ReduceMax", ["logits"], ["lmax"], "rm", [a_int("keepdims", 0)]))
        outs.append(vi("lmax", 1, [1]))
    return model(nodes, inits, ins, outs)

def build_model_dir(d, embed_inline=False):
    d = Path(d); (d / "onnx").mkdir(parents=True, exist_ok=True)
    table = (rng.standard_normal((V, H))).astype(np.float32)
    if embed_inline:
        init = tensor("embed_tokens.weight", table)
    else:
        (d / "onnx/embedding.onnx.data").write_bytes(b"\0" * 64 + table.tobytes())
        init = tensor_ext("embed_tokens.weight", table.shape, "embedding.onnx.data", 64, table.nbytes)
    (d / "onnx/embedding.onnx").write_bytes(model(
        [node("Gather", ["embed_tokens.weight", "input_ids"], ["inputs_embeds"], "gather")],
        [init], [vi("input_ids", 7, [1, "s"])], [vi("inputs_embeds", 1, [1, "s", H])]))
    Wv = (rng.standard_normal((2352, H)) * .02).astype(np.float32)
    (d / "onnx/vision_encoder_q4.onnx").write_bytes(model(
        [node("Reshape", ["pixel_values", "pshape"], ["p2"], "r"),
         node("MatMul", ["p2", "Wv"], ["image_embeds"], "mm")],
        [tensor("pshape", np.array([-1, 2352], np.int64)), tensor("Wv", Wv)],
        [vi("pixel_values", 1, [1, "n", 3, 14, 14]), vi("grid_thw", 7, [1, 3])],
        [vi("image_embeds", 1, ["m", H])]))
    # Same toy graph under both variant names (q8 is the default).
    shutil.copy(d / "onnx/vision_encoder_q4.onnx", d / "onnx/vision_encoder_q8.onnx")
    (d / "onnx/decoder_q4.onnx").write_bytes(build_decoder())
    (d / "tokenizer.json").write_text("{}"); (d / "config.json").write_text("{}")
    return table

def sess(b):
    return ort.InferenceSession(b, providers=["CPUExecutionProvider"])

# ---------------------------------------------------------------------------
def test_embedding_table(tmp):
    table = build_model_dir(tmp / "ext")
    et, why = ts.EmbeddingTable.from_onnx(tmp / "ext/onnx/embedding.onnx")
    assert et is not None, why
    ids = np.array([[0, 5, 511, 500, 5]], np.int64)
    ref = sess(str(tmp / "ext/onnx/embedding.onnx")).run(None, {"input_ids": ids})[0]
    out = et.lookup(ids)
    assert out.shape == (1, 5, H) and out.dtype == np.float32 and out.flags.writeable
    assert np.array_equal(out, ref) and np.array_equal(out[0, 2], table[511])
    build_model_dir(tmp / "inl", embed_inline=True)
    et2, why = ts.EmbeddingTable.from_onnx(tmp / "inl/onnx/embedding.onnx")
    assert et2 is not None, why
    assert np.array_equal(et2.lookup(ids), sess(str(tmp / "inl/onnx/embedding.onnx")).run(None, {"input_ids": ids})[0])
    try:
        et.lookup(np.array([[V]])); raise AssertionError("no range check")
    except ValueError:
        pass
    # A graph that scales the output must NOT be served from the table.
    scaled = model([node("Gather", ["t", "input_ids"], ["g"], "g"),
                    node("Mul", ["g", "s"], ["out"], "mul")],
                   [tensor("t", table), tensor("s", np.array([2.0], np.float32))],
                   [vi("input_ids", 7, [1, "s"])], [vi("out", 1, [1, "s", H])])
    (tmp / "scaled.onnx").write_bytes(scaled)
    et3, why = ts.EmbeddingTable.from_onnx(tmp / "scaled.onnx")
    assert et3 is None and "Gather" in why, why
    print("  embedding table: external + inline match ORT; scaled graph rejected ->", why)

def test_logits_slice(tmp):
    rs = np.random.default_rng(1)
    for kw in ({"lm": "nbits"}, {"lm": "matmul"}, {"lm": "nbits", "identity_chain": True}):
        p = tmp / "dec.onnx"; p.write_bytes(build_decoder(**kw))
        patched, why = ts._last_position_logits_model(p)
        assert patched is not None, why
        full, cut = sess(str(p)), sess(patched)
        for S, P in ((37, 0), (1, 12)):
            feed = {"inputs_embeds": rs.standard_normal((1, S, H)).astype(np.float32),
                    "attention_mask": np.ones((1, S + P), np.int64)}
            for i in range(L):
                for k in ("key", "value"):
                    feed[f"past_key_values.{i}.{k}"] = rs.standard_normal((1, 2, P, 128)).astype(np.float32)
            a, b = full.run(None, feed), cut.run(None, feed)
            assert a[0].shape == (1, S, V) and b[0].shape == (1, 1, V)
            assert np.allclose(a[0][:, -1:], b[0], atol=1e-5), np.abs(a[0][:, -1:] - b[0]).max()
            for x, y in zip(a[1:], b[1:]):
                assert np.array_equal(x, y)
        print(f"  logits slice {kw}: exact match, prefill logits (1,37,{V}) -> (1,1,{V})")
    p.write_bytes(build_decoder(extra_logits_user=True))
    patched, why = ts._last_position_logits_model(p)
    assert patched is None and "used inside the graph" in why, why
    print("  logits consumed inside graph -> patch refused:", why)

def run_capture(fn):
    out, err = io.StringIO(), io.StringIO()
    code = 0
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        try:
            fn()
        except SystemExit as e:
            code = e.code if isinstance(e.code, int) else 1
            if not isinstance(e.code, int) and e.code is not None:
                err.write(str(e.code))
    return code, out.getvalue(), err.getvalue()

def test_parity_and_batch(tmp):
    md = tmp / "model"; build_model_dir(md)
    imgs = []
    for k, (w, h) in enumerate(((300, 200), (640, 480), (123, 777))):
        a = np.random.default_rng(10 + k).integers(0, 255, (h, w, 3), np.uint8)
        sub = tmp / f"in{k}"; sub.mkdir(); p = sub / "page.png"
        Image.fromarray(a).save(p); imgs.append(p)
    def reference():
        if orig is not None:     # original script, one image at a time
            return [orig.run_ocr(Image.open(p).convert("RGB"), md, max_tokens=120)
                    for p in imgs]
        return [ts.run_ocr(Image.open(p).convert("RGB"), md, max_tokens=120)
                for p in imgs]
    os.environ["TEXTSNAP_EMBED"] = "ort"; os.environ["TEXTSNAP_LOGITS_SLICE"] = "0"
    try:
        ref = reference()
    finally:
        del os.environ["TEXTSNAP_EMBED"], os.environ["TEXTSNAP_LOGITS_SLICE"]
    eng = ts.OCREngine(md)
    assert eng.embed_table is not None and eng.logits_sliced
    new = [eng.recognize(Image.open(p).convert("RGB"), max_tokens=120) for p in imgs]
    assert new == ref, (new, ref)
    assert all(len(t) > 10 for t in ref), ref
    print(f"  parity vs {'original script' if orig else 'unoptimized paths'}: "
          f"{len(ref)} images identical "
          f"(lengths {[len(t) for t in ref]}, stop={eng.last_stats['stop']})")
    # Same, with both optimizations disabled -> still identical.
    os.environ["TEXTSNAP_EMBED"] = "ort"; os.environ["TEXTSNAP_LOGITS_SLICE"] = "0"
    try:
        e2 = ts.OCREngine(md)
        assert e2.embed_table is None and not e2.logits_sliced
        assert [e2.recognize(Image.open(p).convert("RGB"), max_tokens=120) for p in imgs] == ref
    finally:
        del os.environ["TEXTSNAP_EMBED"], os.environ["TEXTSNAP_LOGITS_SLICE"]
    print("  fallback paths (ORT embedding, full logits) also identical")

    # ---- CLI batch: 3 images (same stem!) + 1 bad path, -o dir -----------
    out_dir = tmp / "out"
    loads = []
    real = ts.OCREngine
    class Counting(real):
        def __init__(s, *a, **k): loads.append(1); super().__init__(*a, **k)
    ts.OCREngine = Counting
    try:
        sys.argv = ["textsnap", str(imgs[0]), str(tmp / "missing.png"), str(imgs[1]),
                    str(imgs[2]), "--model-dir", str(md), "-o", str(out_dir), "--max-tokens", "120", "--structured"]
        code, out, err = run_capture(ts.main)
    finally:
        ts.OCREngine = real
    lines = out.split()
    assert code == 1 and len(loads) == 1, (code, loads, err)
    assert [Path(x).name for x in lines] == ["page_ocr.txt", "page_ocr_2.txt", "page_ocr_3.txt"], lines
    assert [Path(x).read_text() for x in lines] == ref
    assert "FAILED" in err and "missing.png" in err and "1 of 4 inputs failed" in err, err
    print(f"  batch CLI: 1 model load, stdout={[Path(x).name for x in lines]}, exit={code}")
    print("    stderr:", err.strip().replace("\n", " | "))

    # ---- glob expansion (unexpanded pattern, as on Windows) ---------------
    sys.argv = ["textsnap", str(tmp / "in*" / "*.png"), "--model-dir", str(md),
                "-o", str(tmp / "out_glob"), "--max-tokens", "120", "--plaintext", "--structured"]
    code, out, err = run_capture(ts.main)
    assert code == 0 and len(out.split()) == 3, (code, out, err)
    print("  glob pattern expanded to", len(out.split()), "inputs")

    # ---- single input: behaviour unchanged --------------------------------
    sys.argv = ["textsnap", str(imgs[0]), "--model-dir", str(md), "-o", str(tmp / "one.txt"), "--max-tokens", "120", "--markdown"]
    code, out, err = run_capture(ts.main)
    assert code == 0 and out.strip() == str(tmp / "one.txt") and (tmp / "one.txt").read_text() == ref[0]
    sys.argv = ["textsnap", str(tmp / "nope.png"), "--model-dir", str(md)]
    code, out, err = run_capture(ts.main)
    assert code == 1 and "neither an existing file" in err and out == "", (code, out, err)
    sys.argv = ["textsnap", str(imgs[0]), str(imgs[1]), "-o", str(tmp / "x.txt"), "--model-dir", str(md)]
    code, out, err = run_capture(ts.main)
    assert code == 2 and "must name a directory" in err
    print("  single-input path/-o/error behaviour unchanged; multi-input -o file rejected")

    # ---- vision variants ---------------------------------------------------
    assert ts.DEFAULT_VISION == "q8" and ts.OCREngine(md).vision_variant == "q8"
    e4 = ts.OCREngine(md, vision="q4")
    assert e4.recognize(Image.open(imgs[0]).convert("RGB"), max_tokens=120) == ref[0]
    (md / "onnx/vision_encoder_q4.onnx").unlink()
    code, out, err = run_capture(lambda: ts.OCREngine(md, vision="q4"))
    assert code == 1 and "not found" in err, err
    code, out, err = run_capture(lambda: ts.OCREngine(md, vision="int3"))
    assert code == 1 and "unknown vision variant" in err, err
    print("  default is q8; --vision q4 loads the q4 file; missing file / bad name give clear errors")

def test_model_files_and_checksums(tmp):
    md = tmp / "pm"; build_model_dir(md)
    assert ts._looks_like_model_dir(md)
    (md / "onnx/vision_encoder_q8.onnx").rename(md / "onnx/y")
    assert not ts._looks_like_model_dir(md), "default portable set must include q8"
    assert ts._looks_like_model_dir(md, "q4")
    (md / "onnx/y").rename(md / "onnx/vision_encoder_q8.onnx")
    assert "onnx/vision_encoder_q8.onnx" in ts.model_files()
    (md / "onnx/embedding.onnx.data").rename(md / "onnx/x")
    assert not ts._looks_like_model_dir(md), "portable check must require the .data sidecar"
    assert "onnx/embedding.onnx.data" in ts.model_files()
    assert "onnx/vision_encoder_q8.onnx" in ts.model_files("q8")
    assert "fp32" not in ts.VISION_VARIANTS
    assert ts.HF_REPO == "kouhxp/PaddleOCR-VL-1.5-ONNX" and len(ts.HF_REVISION) == 40
    for f in set(sum(ts.VISION_VARIANTS.values(), [])) | set(ts.EMBEDDING_FILES):
        assert f in ts.EMBEDDED_CHECKSUMS, f
    manifest = ts._load_checksums(Path(__file__).parent / "model_checksums.sha256")
    assert manifest == ts.EMBEDDED_CHECKSUMS, set(manifest.items()) ^ set(ts.EMBEDDED_CHECKSUMS.items())
    # An old manifest (no sidecar entry) must not un-pin the sidecar.
    old = tmp / "model_checksums.sha256"
    old.write_text("d737d600be1bd90ec1e3b537ffe1645a6d780de688904ca4301353df6086f46e  onnx/vision_encoder_q4.onnx\n")
    ts._find_checksum_manifest = lambda: old
    sums, src = ts._pinned_checksums()
    assert sums["onnx/embedding.onnx.data"].startswith("a2299447")
    print("  file registry, portable check, manifest/embedded digests consistent; old manifest merged:", src)


# ---------------------------------------------------------------------------
# PP-OCRv6 path: toy det/rec graphs with the real I/O contract
#   det: x (1,3,H,W) BGR-normalized -> (1,1,H,W) probability map
#   rec: x (N,3,48,W)               -> (N,W/4,C) softmax over CTC classes
# ---------------------------------------------------------------------------
CHARS = ["!", "'", "#", "$", "\\", "~", "\u3000"] + list("abcdefghijklmnopqrstuvwxyz")

def _yml_scalar(c):
    if c == "'":
        return "'" * 4                     # YAML: '' inside single quotes
    if c in "!#~":
        return "'" + c + "'"
    return c

def rec_yml(chars=CHARS):
    return ("Global:\n  model_name: PP-OCRv6_medium_rec\nPreProcess:\n  transform_ops:\n"
            "  - RecResizeImg:\n      image_shape:\n      - 3\n      - 48\n      - 320\n"
            "PostProcess:\n  name: CTCLabelDecode\n  character_dict:\n"
            + "".join("  - " + _yml_scalar(c) + "\n" for c in chars))

DET_YML = ("Global:\n  model_name: PP-OCRv6_medium_det\nPostProcess:\n  box_thresh: 0.45\n"
           "  max_candidates: 3000\n  name: DBPostProcess\n  thresh: 0.2\n  unclip_ratio: 1.4\n"
           "PreProcess:\n  transform_ops:\n  - DecodeImage:\n      img_mode: BGR\n")

def build_ppocr_dir(root, n_chars=len(CHARS), with_space=True):
    d = Path(root) / "ppocr"
    (d / "det").mkdir(parents=True, exist_ok=True); (d / "rec").mkdir(exist_ok=True)
    # det: dark pixels -> high text probability.
    (d / "det/inference.onnx").write_bytes(model(
        [node("ReduceMean", ["x"], ["m"], "rm", [a_ints("axes", [1]), a_int("keepdims", 1)]),
         node("Mul", ["m", "k"], ["z"], "mul"), node("Add", ["z", "b"], ["z2"], "add"),
         node("Sigmoid", ["z2"], ["p"], "sig")],
        [tensor("k", np.array([-8.0], np.float32)), tensor("b", np.array([-2.0], np.float32))],
        [vi("x", 1, [1, 3, "h", "w"])], [vi("p", 1, [1, 1, "h", "w"])]))
    (d / "det/inference.yml").write_text(DET_YML)
    # rec: 4-px column pooling; bright -> class 1, darker -> class 2.
    C = n_chars + 1 + (1 if with_space else 0)
    w = np.full((1, C), 0.0, np.float32); w[0, 1], w[0, 2] = 6.0, -6.0
    # Decision point at v=0.5 (normalized): unclipped crops are ~2.4x the
    # bar's height, so a "dark" column averages well above -1.
    b = np.full((C,), -50.0, np.float32); b[:3] = (0.0, -3.0, 3.0)
    (d / "rec/inference.onnx").write_bytes(model(
        [node("AveragePool", ["x"], ["ap"], "ap", [a_ints("kernel_shape", [48, 4]), a_ints("strides", [48, 4])]),
         node("ReduceMean", ["ap"], ["rm"], "rm", [a_ints("axes", [1, 2]), a_int("keepdims", 0)]),
         node("Unsqueeze", ["rm", "ax"], ["u"], "u"),
         node("MatMul", ["u", "w"], ["l0"], "mm"), node("Add", ["l0", "b"], ["l"], "add"),
         node("Softmax", ["l"], ["y"], "sm", [a_int("axis", -1)])],
        [tensor("w", w), tensor("b", b), tensor("ax", np.array([-1], np.int64))],
        [vi("x", 1, ["n", 3, 48, "w"])], [vi("y", 1, ["n", "t", C])]))
    (d / "rec/inference.yml").write_text(rec_yml(CHARS[:n_chars]))
    return d

def page_image(path, lines=((40, 30, 360, 60), (40, 100, 300, 130))):
    a = np.full((200, 420, 3), 255, np.uint8)
    for x0, y0, x1, y1 in lines:
        a[y0:y1, x0:x1] = 0
        a[y0 + 8:y1 - 8, x0 + 60:x0 + 90] = 255   # a bright gap inside the "line"
    Image.fromarray(a).save(path)
    return path

def test_yaml_and_decode(tmp):
    import yaml
    txt = rec_yml()
    got = ts.parse_inference_yml(txt)["PostProcess"]
    ref = yaml.safe_load(txt)["PostProcess"]
    assert got["character_dict"] == [str(c) for c in ref["character_dict"]] == CHARS, got["character_dict"][:8]
    det = ts.parse_inference_yml(DET_YML)["PostProcess"]
    assert det["thresh"] == "0.2" and det["box_thresh"] == "0.45" and det["unclip_ratio"] == "1.4"
    # CTC: repeats collapse, blanks split, score = mean prob of kept chars.
    cs = ["<blank>", "a", "b", " "]
    seq = [1, 1, 0, 1, 2, 2, 3, 0]
    probs = np.full((1, len(seq), 4), 0.01, np.float32)
    for t, k in enumerate(seq): probs[0, t, k] = 0.9 if k else 0.97
    (text, score), = ts.ctc_decode(probs, cs)
    assert text == "aab " and abs(score - 0.9) < 1e-6, (text, score)
    (text, score), = ts.ctc_decode(np.eye(4, dtype=np.float32)[[0, 0, 0]][None], cs)
    assert text == "" and score == 0.0
    print("  inference.yml parser matches PyYAML; CTC decode collapses/blanks/scores correctly")

def test_det_geometry(tmp):
    import cv2
    # A rotated, filled rectangle in a probability map comes back as one quad
    # that covers it, grown by the unclip distance and scaled to the original.
    pred = np.zeros((200, 300), np.float32)
    rect = ((150, 100), (160, 30), 12.0)
    cv2.fillPoly(pred, [cv2.boxPoints(rect).astype(np.int32)], 0.9)
    boxes, scores = ts.db_boxes(pred, 600, 400, 0.2, 0.45, 1.4)
    assert len(boxes) == 1 and scores[0] > 0.8, (boxes, scores)
    q = boxes[0]
    w, h = np.linalg.norm(q[1] - q[0]), np.linalg.norm(q[3] - q[0])
    d = 160 * 30 * 1.4 / (2 * (160 + 30))
    assert abs(w - 2 * (160 + 2 * d)) < 8 and abs(h - 2 * (30 + 2 * d)) < 8, (w, h)
    assert q[0][0] < q[1][0] and q[0][1] < q[3][1], q          # TL, TR, BR, BL
    assert ts.det_resize(20, 50) == (64, 160) and ts.det_resize(5000, 3000) == (4000, 2400)
    # Reading order: two cells on one row, a paragraph gap, then another row.
    B = lambda x0, y0, x1, y1: np.float32([[x0, y0], [x1, y0], [x1, y1], [x0, y1]])
    txt = ts.assemble_lines([B(200, 10, 300, 30), B(10, 12, 150, 31), B(10, 80, 100, 100), B(10, 40, 90, 60)],
                            ["world", "hello", "para two", "next"])
    assert txt == "hello world\nnext\n\npara two", repr(txt)
    print(f"  DB quad {w:.0f}x{h:.0f} (expected ~{2*(160+2*d):.0f}x{2*(30+2*d):.0f}); reading order + paragraph break ok")

def test_quality_gate(tmp):
    ok = lambda t, s=0.98: ts.line_ok(t, s)
    assert ok("day as a reminder of the")
    assert not ok("day as a reminder of the", 0.5)                  # low confidence
    for bad in ("|||", "llIl1", "l1Il|", "-------", "aaaaaah", "x", "xqzt vbnm plkj wrtz",
                "a1#b c2$d", "@@##$$%%"):
        assert not ok(bad), bad
    assert ok("Total $1,299.00") and ok("v1.2-beta") and ok("东京の天気は晴れ")
    assert ts.line_ok("the moon tells", 0.99, ("fr",)) is False      # wrong language list
    assert ts.line_ok("the moon tells", 0.99, ("fr", "en"))
    res = {"rec_texts": ["The Quarterly Report", "", "|||", "revenue by region is on this page"],
           "rec_scores": [0.99, 0.0, 0.95, 0.97]}
    q = ts.page_quality(res)
    assert abs(q - (20 + 33) / (20 + 3 + 33)) < 1e-9, q               # empty line ignored, '|||' weighed by length
    assert ts.page_quality({"rec_texts": [], "rec_scores": []}) == 0.0
    print(f"  line checks behave; page quality weighted by chars = {q:.3f}")

def test_ppocr_engine_and_fallback(tmp):
    root = tmp / "both"; build_model_dir(root); build_ppocr_dir(root)
    img = page_image(tmp / "doc.png")
    eng = ts.PPOCREngine(root / "ppocr")
    assert eng.det_params == {"thresh": 0.2, "box_thresh": 0.45, "unclip_ratio": 1.4, "max_candidates": 3000}
    res = eng.recognize(Image.open(img))
    assert len(res["rec_polys"]) == 2 and len(res["rec_texts"]) == 2, res
    ys = sorted(float(b[:, 1].mean()) for b in res["rec_polys"])
    assert abs(ys[0] - 45) < 6 and abs(ys[1] - 115) < 6, ys
    # bright margin -> "!", dark bar -> "'", bright gap -> "!", dark bar -> "'"
    assert all(set(t) <= {"!", "'"} and t.startswith("!'") and "'!'" in t
               for t in res["rec_texts"]), res["rec_texts"]
    assert res["text"].count("\n") >= 1 and ts.page_quality(res) == 0.0
    # Model without the trailing space class is accepted too; a wrong dict is not.
    build_ppocr_dir(tmp / "nospace", with_space=False)
    ts.PPOCREngine(tmp / "nospace/ppocr").recognize(Image.open(img))
    build_ppocr_dir(tmp / "bad", n_chars=len(CHARS) - 3)
    (tmp / "bad/ppocr/rec/inference.yml").write_text(rec_yml())
    code, _, err = run_capture(lambda: ts.PPOCREngine(tmp / "bad/ppocr"))
    assert code == 1 and "classes" in err, err
    print("  engine: 2 lines found at the right rows; charset/class-count check works")

    vlm_ref = ts.run_ocr(Image.open(img).convert("RGB"), root, max_tokens=60)
    def cli(*extra):
        sys.argv = ["textsnap", str(img), "--model-dir", str(root), "-o", str(tmp / "o.txt"),
                    "--max-tokens", "60", "-v", *extra]
        code, out, err = run_capture(ts.main)
        assert code == 0, err
        return (tmp / "o.txt").read_text(), err
    # Garbage PP-OCR text -> quality 0 -> VLM re-read, flattened to plain text.
    text, err = cli()
    assert text == ts.to_plaintext(vlm_ref) and "Falling back" in err and ", vlm)" in err, err[-400:]
    # --min-quality 0: never fall back; output is PP-OCR's assembled text.
    text, err = cli("--min-quality", "0")
    assert text == res["text"] and ", ppocr)" in err and "Falling back" not in err
    # --structured: VLM only, native markdown, PP-OCR never loaded.
    loads = []
    real = ts.PPOCREngine
    class Counting(real):
        def __init__(s, *a, **k): loads.append(1); super().__init__(*a, **k)
    ts.PPOCREngine = Counting
    try:
        text, err = cli("--structured")
    finally:
        ts.PPOCREngine = real
    assert text == vlm_ref and loads == [], (loads, text[:40])
    # VLM missing -> keep PP-OCR text, warn once, don't fail the input.
    only = tmp / "only_ppocr"; build_ppocr_dir(only)
    sys.argv = ["textsnap", str(img), str(img), "--model-dir", str(only), "-o", str(tmp / "od")]
    code, out, err = run_capture(ts.main)
    assert code == 0 and len(out.split()) == 2, (code, err)
    assert err.count("PaddleOCR-VL unavailable") == 1, err
    assert Path(out.split()[0]).read_text() == res["text"]
    # Bad threshold is rejected by argparse.
    sys.argv = ["textsnap", str(img), "--min-quality", "1.5"]
    code, _, err = run_capture(ts.main)
    assert code == 2 and "between 0 and 1" in err
    print("  CLI: fallback on low quality, --min-quality 0, --structured (no PP-OCR load), "
          "missing VLM degrades gracefully")

def test_ppocr_pinning(tmp):
    assert ts.PPOCR_HF_REPO == "kouhxp/PP-OCRv6_medium-ONNX"
    for f in ts.PPOCR_FILES:
        assert f in ts.EMBEDDED_CHECKSUMS
    pinned = ts._HEX40.match(ts.PPOCR_HF_REVISION)
    if not pinned:
        # Unpinned mirror: a download must refuse rather than fetch "main".
        old = ts.CACHE_DIR; ts.CACHE_DIR = tmp / "cache"
        real_dirs = ts._script_dirs; ts._script_dirs = lambda: []
        try:
            code, _, err = run_capture(lambda: ts.get_ppocr_dir())
        finally:
            ts.CACHE_DIR = old; ts._script_dirs = real_dirs
        assert code == 1 and "not pinned" in err, err
        f = tmp / "x.onnx"; f.write_bytes(b"x")
        code, _, err = run_capture(lambda: ts.verify_files({"det/inference.onnx": f}, ts.EMBEDDED_CHECKSUMS))
        assert code == 1 and "placeholder" in err, err
        print("  PP-OCRv6 mirror not pinned yet -> downloads and verification refuse")
    else:
        assert all(ts._HEX64.match(ts.EMBEDDED_CHECKSUMS[f]) for f in ts.PPOCR_FILES)
        print("  PP-OCRv6 mirror pinned:", ts.PPOCR_HF_REVISION[:12])
    code, _, err = run_capture(lambda: ts.get_ppocr_dir(str(tmp / "nowhere")))
    assert code == 1 and "PP-OCRv6 files not found" in err


if __name__ == "__main__":
    tmp = Path(tempfile.mkdtemp())
    for t in (test_embedding_table, test_logits_slice, test_parity_and_batch,
              test_model_files_and_checksums, test_yaml_and_decode, test_det_geometry,
              test_quality_gate, test_ppocr_engine_and_fallback, test_ppocr_pinning):
        print(t.__name__)
        t(tmp)
    print("ALL PASSED")
