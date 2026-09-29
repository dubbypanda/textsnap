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
                    str(imgs[2]), "--model-dir", str(md), "-o", str(out_dir), "--max-tokens", "120"]
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
                "-o", str(tmp / "out_glob"), "--max-tokens", "120", "--plaintext"]
    code, out, err = run_capture(ts.main)
    assert code == 0 and len(out.split()) == 3, (code, out, err)
    print("  glob pattern expanded to", len(out.split()), "inputs")

    # ---- single input: behaviour unchanged --------------------------------
    sys.argv = ["textsnap", str(imgs[0]), "--model-dir", str(md), "-o", str(tmp / "one.txt"), "--max-tokens", "120"]
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

if __name__ == "__main__":
    tmp = Path(tempfile.mkdtemp())
    for t in (test_embedding_table, test_logits_slice, test_parity_and_batch,
              test_model_files_and_checksums):
        print(t.__name__)
        t(tmp)
    print("ALL PASSED")
