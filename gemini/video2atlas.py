"""Turn a green-screen (or black-screen) Gemini/Veo clip into a keyed sprite-sheet atlas.

usage: python3 video2atlas.py in.mp4 out_name frames fps_from fps_to cell [bg=green|black] [start] [dur]
Writes ../assets/a_<out_name>.webp and prints JSON metadata (cols, rows, frames, aspect).
"""
import sys, json, subprocess, tempfile, os, glob
import numpy as np
from PIL import Image
from scipy import ndimage

src, name, N, cell = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
bgmode = sys.argv[5] if len(sys.argv) > 5 else 'green'
start = float(sys.argv[6]) if len(sys.argv) > 6 else 0.0
dur = float(sys.argv[7]) if len(sys.argv) > 7 else None
tmp = tempfile.mkdtemp()
probe = json.loads(subprocess.check_output(['ffprobe', '-v', 'error', '-show_entries', 'format=duration', '-of', 'json', src]))
total = float(probe['format']['duration'])
dur = dur or (total - start - 0.05)
subprocess.run(['ffmpeg', '-v', 'error', '-ss', str(start), '-t', str(dur), '-i', src, '-vf', f'fps={N/dur}', f'{tmp}/f_%04d.png'], check=True)
files = sorted(glob.glob(f'{tmp}/f_*.png'))[:N]


def key(im):
    im = im.astype(np.float32)
    r, g, b = im[..., 0], im[..., 1], im[..., 2]
    if bgmode == 'black':
        lum = im.max(axis=2)
        a = np.clip((lum - 10) / 30, 0, 1)
        return im, a
    border = np.concatenate([im[:6].reshape(-1, 3), im[-6:].reshape(-1, 3), im[:, :6].reshape(-1, 3), im[:, -6:].reshape(-1, 3)])
    bg = np.median(border, axis=0)
    mx = np.maximum(r, b); dom = g - mx
    dist = np.linalg.norm(im - bg, axis=2)
    cand = (dom > 45) & (dist < 160)
    lab, n = ndimage.label(cand)
    edge = set(np.unique(np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]]))) - {0}
    bgm = np.isin(lab, list(edge))
    if n:
        sizes = ndimage.sum(np.ones_like(dist), lab, range(1, n + 1)); md = ndimage.mean(dist, lab, range(1, n + 1))
        for i in range(n):
            if (i + 1) not in edge and sizes[i] > 60 and md[i] < 80:
                bgm |= (lab == i + 1)
    near = ndimage.binary_dilation(bgm, iterations=3)
    a = np.where(bgm, 0.0, 1.0)
    soft = np.clip((80 - dom) / 60, 0, 1)
    a = np.where(near & ~bgm, np.minimum(a, soft), a)
    a = ndimage.gaussian_filter(a, 0.7)
    im[..., 1] = np.where(near | (dom > 20), np.minimum(g, mx + np.clip(dom, 0, 40) * 0.25), g)
    return im, a


frames = []
for f in files:
    im, a = key(np.asarray(Image.open(f).convert('RGB')))
    frames.append(np.dstack([im, a * 255]).clip(0, 255).astype(np.uint8))
# common crop across all frames so the creature does not jitter
alpha = np.max([fr[..., 3] for fr in frames], axis=0)
ys, xs = np.nonzero(alpha > 10)
x0, x1, y0, y1 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
cw, chh = x1 - x0, y1 - y0
aspect = cw / chh
if aspect >= 1:
    W, H = cell, max(8, int(round(cell / aspect)))
else:
    W, H = max(8, int(round(cell * aspect))), cell
cols = int(np.ceil(np.sqrt(len(frames) * H / W)))
rows = int(np.ceil(len(frames) / cols))
atlas = Image.new('RGBA', (cols * W, rows * H), (0, 0, 0, 0))
for i, fr in enumerate(frames):
    img = Image.fromarray(fr[y0:y1, x0:x1], 'RGBA').resize((W, H), Image.LANCZOS)
    atlas.paste(img, ((i % cols) * W, (i // cols) * H))
out = os.path.join(os.path.dirname(__file__), '..', 'assets', f'a_{name}.webp')
atlas.save(out, 'WEBP', quality=84, method=6)
meta = {'name': name, 'cols': cols, 'rows': rows, 'frames': len(frames), 'aspect': round(aspect, 4), 'w': cols * W, 'h': rows * H, 'bytes': os.path.getsize(out)}
print(json.dumps(meta))
