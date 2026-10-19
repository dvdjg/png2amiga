#!/usr/bin/env python3
"""SHAM5b (flicker) converter front-end.

Ports the idea from Leonard's abc-bitmap-converter: quantise the source to
RGB555 with error diffusion, split each 5-bit channel into two RGB444
frames (floor/ceil, with a checkerboard phase swap), encode both frames as
SHAM (HAM6 + per-scanline copper palette), and emit the pair plus a
flicker GIF and the perceived (temporal average) preview.

The two frames are separate hardware states; a single static PNG cannot
represent the 50 Hz alternation. The tool therefore writes:

  <prefix>-a.png      frame A preview (SHAM encode)
  <prefix>-b.png      frame B preview (SHAM encode)
  <prefix>-avg.png    per-pixel average of A and B (what the eye sees)
  <prefix>-flicker.gif animated A/B pair (20 ms per frame)

The 444 source frames and the SHAM encoding are delegated to the
png2amiga binary (`--mode ham6 --sliced --dither none`), so the copper
palette planning and HAM output match the rest of the converter.

Usage:
  python tools/sham5b.py input.png --out-dir out/sham5b
  python tools/sham5b.py input.png --prefix out/room_1__sham5b --gif
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from PIL import Image


# ---------------------------------------------------------------------------
# RGB888 -> RGB555 with optional error diffusion (sRGB space, as abc does)
# ---------------------------------------------------------------------------

KERNELS = {
    "floyd-steinberg": [
        (1, 0, 7 / 16), (-1, 1, 3 / 16), (0, 1, 5 / 16), (1, 1, 1 / 16),
    ],
    "sierra": [
        (1, 0, 5 / 32), (2, 0, 3 / 32),
        (-1, 1, 2 / 32), (0, 1, 4 / 32), (1, 1, 2 / 32), (2, 1, 3 / 32),
        (-1, 2, 2 / 32), (0, 2, 3 / 32), (1, 2, 2 / 32), (2, 2, 3 / 32),
    ],
    "jarvis": [
        (1, 0, 7 / 48), (2, 0, 5 / 48),
        (-2, 1, 3 / 48), (-1, 1, 5 / 48), (0, 1, 7 / 48), (1, 1, 5 / 48), (2, 1, 3 / 48),
        (-2, 2, 1 / 48), (-1, 2, 3 / 48), (0, 2, 5 / 48), (1, 2, 3 / 48), (2, 2, 1 / 48),
    ],
}

BAYER16 = [
    [0, 191, 48, 239, 12, 203, 60, 251, 3, 194, 51, 242, 15, 206, 63, 254],
    [127, 64, 175, 112, 139, 76, 187, 124, 130, 67, 178, 115, 142, 79, 190, 127],
    [32, 223, 16, 207, 44, 235, 28, 219, 35, 226, 19, 210, 47, 238, 31, 222],
    [159, 96, 143, 80, 171, 108, 155, 92, 162, 99, 146, 83, 174, 111, 158, 95],
    [8, 199, 56, 247, 4, 195, 52, 243, 11, 202, 59, 250, 7, 198, 55, 246],
    [135, 72, 183, 120, 131, 68, 179, 116, 138, 75, 186, 123, 134, 71, 182, 119],
    [40, 231, 24, 215, 36, 227, 20, 211, 43, 234, 27, 218, 39, 230, 23, 214],
    [167, 104, 151, 88, 163, 100, 147, 84, 170, 107, 154, 91, 166, 103, 150, 87],
    [2, 193, 50, 241, 14, 205, 62, 253, 1, 192, 49, 240, 13, 204, 61, 252],
    [129, 66, 177, 114, 141, 78, 189, 126, 128, 65, 176, 113, 140, 77, 188, 125],
    [34, 225, 18, 209, 46, 237, 30, 221, 33, 224, 17, 208, 45, 236, 29, 220],
    [161, 98, 145, 82, 173, 110, 157, 94, 160, 97, 144, 81, 172, 109, 156, 93],
    [10, 201, 58, 249, 6, 197, 54, 245, 9, 200, 57, 248, 5, 196, 53, 244],
    [137, 74, 185, 122, 133, 70, 181, 118, 136, 73, 184, 121, 132, 69, 180, 117],
    [42, 233, 26, 217, 38, 229, 22, 213, 41, 232, 25, 216, 37, 228, 21, 212],
    [169, 106, 153, 90, 165, 102, 149, 86, 168, 105, 152, 89, 164, 101, 148, 85],
]


def quantize_channel(value: float) -> int:
    """Nearest RGB555 level (0..31) for an 8-bit sample in 0..255."""
    return max(0, min(31, int((value + 4.0) // 8.2258)))


def quantize_555(img: Image.Image, method: str) -> Image.Image:
    """Return a 555-quantised copy (channels are 0..31)."""
    if method == "bayer":
        src = img.convert("RGB")
        out = Image.new("RGB", src.size)
        sp, op = src.load(), out.load()
        for y in range(src.height):
            for x in range(src.width):
                r, g, b = sp[x, y]
                t = BAYER16[y & 15][x & 15]
                # abc's ordered rule: add one 5-bit step when below threshold.
                op[x, y] = (
                    quantize_channel(r) + (1 if (r & 0xFF) < t else 0),
                    quantize_channel(g) + (1 if (g & 0xFF) < t else 0),
                    quantize_channel(b) + (1 if (b & 0xFF) < t else 0),
                )
        return out

    src = img.convert("RGB")
    w, h = src.size
    buf = [[[float(c) for c in src.getpixel((x, y))] for x in range(w)] for y in range(h)]
    kernel = KERNELS.get(method)
    for y in range(h):
        for x in range(w):
            for ci in range(3):
                v = buf[y][x][ci]
                q = quantize_channel(max(0.0, min(255.0, v)))
                q8 = q * (255.0 / 31.0)
                err = v - q8
                buf[y][x][ci] = q
                if kernel:
                    for dx, dy, wt in kernel:
                        nx, ny = x + dx, y + dy
                        if 0 <= nx < w and 0 <= ny < h:
                            buf[ny][nx][ci] += err * wt
    out = Image.new("RGB", (w, h))
    op = out.load()
    for y in range(h):
        for x in range(w):
            op[x, y] = tuple(int(round(max(0, min(31, buf[y][x][ci])))) for ci in range(3))
    return out


def split555(img555: Image.Image) -> tuple[Image.Image, Image.Image]:
    """Split a 555 image into two 444 frames (abc Split555).

    A = value >> 1, B = A + (value & 1), with a checkerboard swap so the
    rounding error is spread across both space and time.
    """
    w, h = img555.size
    a = Image.new("RGB", (w, h))
    b = Image.new("RGB", (w, h))
    sp, ap, bp = img555.load(), a.load(), b.load()
    for y in range(h):
        for x in range(w):
            r, g, bl = sp[x, y]
            ar, ag, ab = r >> 1, g >> 1, bl >> 1
            br = min(15, ar + (r & 1))
            bg = min(15, ag + (g & 1))
            bb = min(15, ab + (bl & 1))
            if (x ^ y) & 1:
                ar, br = br, ar
                ag, bg = bg, ag
                ab, bb = bb, ab
            ap[x, y] = (ar * 17, ag * 17, ab * 17)
            bp[x, y] = (br * 17, bg * 17, bb * 17)
    return a, b


def find_png2amiga(explicit: str | None) -> str:
    if explicit:
        return explicit
    here = Path(__file__).resolve().parent.parent
    for cand in (here / "build" / "png2amiga", here / "build" / "png2amiga.exe",
                 here / "build-msvc" / "png2amiga.exe"):
        if cand.exists():
            return str(cand)
    exe = shutil.which("png2amiga")
    if exe:
        return exe
    raise SystemExit("png2amiga binary not found; pass --png2amiga")


def run(cmd: list[str]) -> None:
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        sys.stderr.write(proc.stdout)
        sys.stderr.write(proc.stderr)
        raise SystemExit(f"command failed ({proc.returncode}): {' '.join(cmd)}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input", help="source PNG/JPEG (24-bit or indexed)")
    ap.add_argument("--prefix", help="output prefix (default: <out-dir>/<stem>__sham5b)")
    ap.add_argument("--out-dir", default=".", help="output directory for generated files")
    ap.add_argument("--png2amiga", help="path to the png2amiga binary")
    ap.add_argument("--dither", default="sierra",
                    choices=["none", "floyd-steinberg", "sierra", "jarvis", "bayer"],
                    help="RGB888 -> RGB555 dither (default: sierra, as abc)")
    ap.add_argument("--slice-changes", type=int, default=0,
                    help="forwarded to png2amiga --slice-changes (0 = auto)")
    ap.add_argument("--best", action="store_true", help="forwarded to png2amiga --best")
    ap.add_argument("--ham-metric", help="forwarded to png2amiga --ham-metric (oklab2, oklab-lum, abc342, srgb-mse)")
    ap.add_argument("--ham-smooth", type=float, help="forwarded to png2amiga --ham-smooth")
    ap.add_argument("--gif", action="store_true", help="also write the animated flicker GIF/APNG")
    ap.add_argument("--frame-ms", type=float, default=16.666,
                    help="flicker frame duration for the preview (default: 16.666 ms, 60 Hz)")
    ap.add_argument("--keep-444", action="store_true", help="keep the split 444 source frames")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(args.input).stem
    prefix = Path(args.prefix) if args.prefix else out_dir / f"{stem}__sham5b"
    prefix.parent.mkdir(parents=True, exist_ok=True)

    img = Image.open(args.input).convert("RGB")
    img555 = quantize_555(img, args.dither)
    frame_a, frame_b = split555(img555)

    with tempfile.TemporaryDirectory(prefix="sham5b-") as tmp:
        tmp = Path(tmp)
        src_a, src_b = tmp / "a444.png", tmp / "b444.png"
        frame_a.save(src_a)
        frame_b.save(src_b)
        if args.keep_444:
            frame_a.save(prefix.parent / f"{prefix.name}-a444.png")
            frame_b.save(prefix.parent / f"{prefix.name}-b444.png")

        exe = find_png2amiga(args.png2amiga)
        out_a = prefix.parent / f"{prefix.name}-a.png"
        out_b = prefix.parent / f"{prefix.name}-b.png"
        common = ["--mode", "ham6", "--sliced", "--dither", "none", "--no-scale"]
        if args.slice_changes:
            common += ["--slice-changes", str(args.slice_changes)]
        if args.best:
            common += ["--best"]
        if args.ham_metric:
            common += ["--ham-metric", args.ham_metric]
        if args.ham_smooth is not None:
            common += ["--ham-smooth", str(args.ham_smooth)]
        run([exe, str(src_a), "-o", str(out_a)] + common)
        run([exe, str(src_b), "-o", str(out_b)] + common)

    # Temporal average in sRGB (same recipe as abc's ColorMix).
    pa, pb = Image.open(out_a).convert("RGB"), Image.open(out_b).convert("RGB")
    if pa.size != pb.size:
        pb = pb.resize(pa.size, Image.NEAREST)
    avg = Image.new("RGB", pa.size)
    apxa, apxb, apxo = pa.load(), pb.load(), avg.load()
    for y in range(pa.height):
        for x in range(pa.width):
            apxo[x, y] = tuple((apxa[x, y][i] + apxb[x, y][i]) >> 1 for i in range(3))
    out_avg = prefix.parent / f"{prefix.name}-avg.png"
    avg.save(out_avg)

    gif = None
    apng = None
    if args.gif:
        # GIF stores frame delays in centiseconds, so 16.666 ms is not
        # representable; round to the nearest centisecond (usually 20 ms).
        gif_ms = max(10, int(round(args.frame_ms / 10.0)) * 10)
        gif = prefix.parent / f"{prefix.name}-flicker.gif"
        pa.save(gif, save_all=True, append_images=[pb], duration=gif_ms, loop=0, optimize=False)
        # APNG keeps millisecond delays: the 60 Hz preview the GIF cannot
        # express.
        apng = prefix.parent / f"{prefix.name}-flicker-60hz.png"
        pa.save(apng, save_all=True, append_images=[pb], duration=args.frame_ms, loop=0)

    print(json.dumps({
        "input": str(args.input),
        "frame_a": str(out_a),
        "frame_b": str(out_b),
        "average": str(out_avg),
        "gif": str(gif) if gif else None,
        "apng": str(apng) if apng else None,
        "gif_frame_ms": (max(10, int(round(args.frame_ms / 10.0)) * 10) if args.gif else None),
        "frame_ms": args.frame_ms,
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
