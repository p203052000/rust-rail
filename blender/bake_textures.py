"""Bake seamless PBR textures for 鏽軌逃生 with Blender (bpy) + Cycles.

Each material is a procedural node graph sampled on a 4D torus mapping
(u,v -> cos/sin pairs) so the result tiles without seams. We bake the colour
and a height field, then derive a tangent-space normal map with numpy.
Run:  python3 bake_textures.py <out_dir>
"""
import bpy, math, sys, os
import numpy as np

OUT = sys.argv[-1] if len(sys.argv) > 1 else "out"
os.makedirs(OUT, exist_ok=True)
RES = 1024
TAU = 2 * math.pi

bpy.ops.wm.read_factory_settings(use_empty=True)
scn = bpy.context.scene
scn.render.engine = 'CYCLES'
scn.cycles.device = 'CPU'
scn.cycles.samples = 1
scn.cycles.bake_type = 'EMIT'
scn.view_settings.view_transform = 'Standard'

bpy.ops.mesh.primitive_plane_add(size=2)
plane = bpy.context.active_object


def new_mat(name):
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    nt = m.node_tree
    for n in list(nt.nodes):
        nt.nodes.remove(n)
    return m, nt


def torus(nt, ru, rv):
    """UV -> (vector, w) on a 4D torus; ru/rv give anisotropy."""
    tc = nt.nodes.new('ShaderNodeTexCoord')
    sep = nt.nodes.new('ShaderNodeSeparateXYZ')
    nt.links.new(tc.outputs['UV'], sep.inputs[0])

    def op(kind, a, b=None, val=None):
        n = nt.nodes.new('ShaderNodeMath')
        n.operation = kind
        nt.links.new(a, n.inputs[0])
        if b is not None:
            nt.links.new(b, n.inputs[1])
        elif val is not None:
            n.inputs[1].default_value = val
        return n.outputs[0]

    a = op('MULTIPLY', sep.outputs['X'], val=TAU)
    b = op('MULTIPLY', sep.outputs['Y'], val=TAU)
    x = op('MULTIPLY', op('COSINE', a), val=ru)
    y = op('MULTIPLY', op('SINE', a), val=ru)
    z = op('MULTIPLY', op('COSINE', b), val=rv)
    w = op('MULTIPLY', op('SINE', b), val=rv)
    comb = nt.nodes.new('ShaderNodeCombineXYZ')
    nt.links.new(x, comb.inputs[0]); nt.links.new(y, comb.inputs[1]); nt.links.new(z, comb.inputs[2])
    return comb.outputs[0], w


def noise(nt, vec, w, scale, detail=8, rough=0.55, distort=0.0):
    n = nt.nodes.new('ShaderNodeTexNoise')
    n.noise_dimensions = '4D'
    nt.links.new(vec, n.inputs['Vector']); nt.links.new(w, n.inputs['W'])
    n.inputs['Scale'].default_value = scale
    n.inputs['Detail'].default_value = detail
    n.inputs['Roughness'].default_value = rough
    n.inputs['Distortion'].default_value = distort
    return n.outputs['Fac']


def voronoi(nt, vec, w, scale, feature='F1', rnd=1.0, out='Distance'):
    n = nt.nodes.new('ShaderNodeTexVoronoi')
    n.voronoi_dimensions = '4D'
    n.feature = feature
    nt.links.new(vec, n.inputs['Vector']); nt.links.new(w, n.inputs['W'])
    n.inputs['Scale'].default_value = scale
    n.inputs['Randomness'].default_value = rnd
    return n.outputs[out]


def ramp(nt, fac, stops, interp='LINEAR'):
    r = nt.nodes.new('ShaderNodeValToRGB')
    r.color_ramp.interpolation = interp
    els = r.color_ramp.elements
    while len(els) > 1:
        els.remove(els[-1])
    for i, (pos, col) in enumerate(stops):
        e = els[0] if i == 0 else els.new(pos)
        e.position = pos
        e.color = (*col, 1.0)
    nt.links.new(fac, r.inputs['Fac'])
    return r.outputs['Color']


def mix(nt, a, b, fac, blend='MIX'):
    m = nt.nodes.new('ShaderNodeMix')
    m.data_type = 'RGBA'
    m.blend_type = blend
    nt.links.new(fac, m.inputs['Factor'])
    nt.links.new(a, m.inputs[6]); nt.links.new(b, m.inputs[7])
    return m.outputs[2]


def math_n(nt, kind, a, b=None, val=None, clamp=False):
    n = nt.nodes.new('ShaderNodeMath')
    n.operation = kind
    n.use_clamp = clamp
    if isinstance(a, (int, float)):
        n.inputs[0].default_value = a
    else:
        nt.links.new(a, n.inputs[0])
    if b is not None:
        if isinstance(b, (int, float)):
            n.inputs[1].default_value = b
        else:
            nt.links.new(b, n.inputs[1])
    elif val is not None:
        n.inputs[1].default_value = val
    return n.outputs[0]


def srgb(h):
    h = h.lstrip('#')
    c = [int(h[i:i + 2], 16) / 255 for i in (0, 2, 4)]
    return tuple(((x + 0.055) / 1.055) ** 2.4 if x > 0.04045 else x / 12.92 for x in c)


def bake(nt, mat, socket, name, noncolor=False):
    em = nt.nodes.new('ShaderNodeEmission')
    out = nt.nodes.new('ShaderNodeOutputMaterial')
    nt.links.new(socket, em.inputs['Color'])
    nt.links.new(em.outputs[0], out.inputs['Surface'])
    img = bpy.data.images.new(name, RES, RES, float_buffer=noncolor)
    if noncolor:
        img.colorspace_settings.name = 'Non-Color'
    tn = nt.nodes.new('ShaderNodeTexImage')
    tn.image = img
    for n in nt.nodes:
        n.select = False
    tn.select = True
    nt.nodes.active = tn
    plane.data.materials.clear()
    plane.data.materials.append(mat)
    bpy.context.view_layer.objects.active = plane
    plane.select_set(True)
    bpy.ops.object.bake(type='EMIT')
    px = np.empty(RES * RES * 4, dtype=np.float32)
    img.pixels.foreach_get(px)
    nt.nodes.remove(em); nt.nodes.remove(out); nt.nodes.remove(tn)
    return px.reshape(RES, RES, 4)[::-1]  # top row first


def save_color(arr, path):
    from PIL import Image
    a = np.clip(arr[..., :3], 0, 1)
    Image.fromarray((a * 255 + 0.5).astype(np.uint8)).save(path, quality=88)


def save_normal(h, path, strength):
    from PIL import Image
    hh = h[..., 0].astype(np.float64)
    dx = (np.roll(hh, 1, 1) - np.roll(hh, -1, 1)) * strength
    dy = (np.roll(hh, -1, 0) - np.roll(hh, 1, 0)) * strength
    n = np.dstack([dx, dy, np.ones_like(hh)])
    n /= np.linalg.norm(n, axis=2, keepdims=True)
    Image.fromarray(((n * 0.5 + 0.5) * 255 + 0.5).astype(np.uint8)).save(path, quality=92)


def rust():
    mat, nt = new_mat('rust')
    v, w = torus(nt, 1.0, 1.0)
    big = noise(nt, v, w, 1.6, 10, 0.62, 0.3)
    pits = voronoi(nt, v, w, 14.0)
    flake = noise(nt, v, w, 2.6, 6, 0.5)
    fine = noise(nt, v, w, 18.0, 4, 0.7)
    base = ramp(nt, big, [(0.0, srgb('#2a1408')), (0.38, srgb('#5c2a12')), (0.55, srgb('#9a4a1c')), (0.7, srgb('#c8702e')), (1.0, srgb('#e09a52'))])
    paint = ramp(nt, flake, [(0.0, srgb('#3a3a34')), (1.0, srgb('#55524a'))])
    pmask = math_n(nt, 'GREATER_THAN', flake, val=0.6)
    col = mix(nt, base, paint, pmask)
    pitmask = math_n(nt, 'LESS_THAN', pits, val=0.08)
    col = mix(nt, col, ramp(nt, fine, [(0, srgb('#140a05')), (1, srgb('#2a140a'))]), pitmask)
    col = mix(nt, col, ramp(nt, fine, [(0, (0.0, 0.0, 0.0)), (1, (1.0, 1.0, 1.0))]), math_n(nt, 'MULTIPLY', fine, val=0.25), 'OVERLAY')
    save_color(bake(nt, mat, col, 'rust_c'), f'{OUT}/rust_c.jpg')
    h = math_n(nt, 'ADD', math_n(nt, 'MULTIPLY', big, val=0.5), math_n(nt, 'MULTIPLY', fine, val=0.25))
    h = math_n(nt, 'ADD', h, math_n(nt, 'MULTIPLY', pmask, val=0.12))
    h = math_n(nt, 'SUBTRACT', h, math_n(nt, 'MULTIPLY', pitmask, val=0.18))
    save_normal(bake(nt, mat, h, 'rust_h', True), f'{OUT}/rust_n.jpg', 9.0)


def rock():
    mat, nt = new_mat('rock')
    v, w = torus(nt, 1.0, 1.0)
    big = noise(nt, v, w, 1.4, 12, 0.6, 0.6)
    cr = voronoi(nt, v, w, 1.7, 'DISTANCE_TO_EDGE')
    fine = noise(nt, v, w, 14.0, 6, 0.65)
    moss = noise(nt, v, w, 3.0, 6, 0.55)
    col = ramp(nt, big, [(0.0, srgb('#2e2a26')), (0.45, srgb('#5a544c')), (0.75, srgb('#857c70')), (1.0, srgb('#a49a8c'))])
    crack = math_n(nt, 'LESS_THAN', cr, val=0.022)
    col = mix(nt, col, ramp(nt, fine, [(0, srgb('#0e0c0a')), (1, srgb('#1e1a16'))]), crack)
    mmask = math_n(nt, 'MULTIPLY', math_n(nt, 'GREATER_THAN', moss, val=0.58), math_n(nt, 'LESS_THAN', big, val=0.55))
    col = mix(nt, col, ramp(nt, fine, [(0, srgb('#1e3018')), (1, srgb('#4a6a30'))]), mmask)
    save_color(bake(nt, mat, col, 'rock_c'), f'{OUT}/rock_c.jpg')
    h = math_n(nt, 'ADD', big, math_n(nt, 'MULTIPLY', fine, val=0.3))
    h = math_n(nt, 'SUBTRACT', h, math_n(nt, 'MULTIPLY', crack, val=0.35))
    save_normal(bake(nt, mat, h, 'rock_h', True), f'{OUT}/rock_n.jpg', 10.0)


def wood():
    mat, nt = new_mat('wood')
    v, w = torus(nt, 0.35, 4.5)  # stretched grain along v
    grain = noise(nt, v, w, 2.2, 10, 0.6, 1.2)
    rings = math_n(nt, 'SINE', math_n(nt, 'MULTIPLY', grain, val=40.0))
    fine = noise(nt, v, w, 9.0, 4, 0.6)
    col = ramp(nt, grain, [(0.0, srgb('#1c130c')), (0.5, srgb('#3e2a1a')), (1.0, srgb('#6a4a30'))])
    col = mix(nt, col, ramp(nt, rings, [(0, srgb('#22160e')), (1, srgb('#5a3e28'))]), math_n(nt, 'MULTIPLY', fine, val=0.5))
    save_color(bake(nt, mat, col, 'wood_c'), f'{OUT}/wood_c.jpg')
    h = math_n(nt, 'ADD', math_n(nt, 'MULTIPLY', rings, val=0.15), math_n(nt, 'MULTIPLY', grain, val=0.5))
    save_normal(bake(nt, mat, h, 'wood_h', True), f'{OUT}/wood_n.jpg', 7.0)


def paint():
    mat, nt = new_mat('paint')
    v, w = torus(nt, 1.0, 1.0)
    peel = noise(nt, v, w, 2.4, 8, 0.6, 0.4)
    cells = voronoi(nt, v, w, 9.0, 'DISTANCE_TO_EDGE')
    vs, ws = torus(nt, 6.0, 0.4)
    streak = noise(nt, vs, ws, 1.5, 6, 0.6)
    fine = noise(nt, v, w, 16.0, 4, 0.6)
    white = ramp(nt, fine, [(0, srgb('#bdbab0')), (1, srgb('#e2e0d6'))])
    rustc = ramp(nt, fine, [(0, srgb('#4a200c')), (1, srgb('#9a4a1e'))])
    peeled = math_n(nt, 'GREATER_THAN', peel, val=0.6)
    col = mix(nt, white, rustc, peeled)
    col = mix(nt, col, rustc, math_n(nt, 'MULTIPLY', math_n(nt, 'GREATER_THAN', streak, val=0.55), val=0.55))
    col = mix(nt, col, ramp(nt, fine, [(0, srgb('#2a2620')), (1, srgb('#3a342c'))]), math_n(nt, 'MULTIPLY', math_n(nt, 'LESS_THAN', cells, val=0.02), val=0.5))
    save_color(bake(nt, mat, col, 'paint_c'), f'{OUT}/paint_c.jpg')
    h = math_n(nt, 'ADD', math_n(nt, 'MULTIPLY', math_n(nt, 'SUBTRACT', 1.0, peeled), val=0.3), math_n(nt, 'MULTIPLY', fine, val=0.15))
    save_normal(bake(nt, mat, h, 'paint_h', True), f'{OUT}/paint_n.jpg', 8.0)


for f in (rust, rock, wood, paint):
    f()
    print('baked', f.__name__, flush=True)
