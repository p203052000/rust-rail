"""Model hero assets for 鏽軌逃生 in Blender and export them to a compact binary.

Exports non-indexed triangles (position, split normal, uv) converted from
Blender's Z-up to three.js Y-up:  (x, y, z)_three = (x, z, -y)_blender.
Run:  python3 build_models.py <out_dir>
"""
import bpy, bmesh, math, sys, os, json, random
import numpy as np
from mathutils import Vector

OUT = sys.argv[-1] if len(sys.argv) > 1 else "out"
os.makedirs(OUT, exist_ok=True)
random.seed(7)
bpy.ops.wm.read_factory_settings(use_empty=True)


def clear():
    for o in list(bpy.data.objects):
        bpy.data.objects.remove(o, do_unlink=True)


def activate(o):
    bpy.ops.object.select_all(action='DESELECT')
    o.select_set(True)
    bpy.context.view_layer.objects.active = o


def apply_all(o):
    activate(o)
    for m in list(o.modifiers):
        bpy.ops.object.modifier_apply(modifier=m.name)


def smart_uv(o, scale=1.0, angle=66):
    activate(o)
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.select_all(action='SELECT')
    bpy.ops.uv.smart_project(angle_limit=math.radians(angle), island_margin=0.02, scale_to_bounds=False)
    bpy.ops.object.mode_set(mode='OBJECT')
    uv = o.data.uv_layers.active.data
    for d in uv:
        d.uv = d.uv * scale


def auto_smooth(o, deg):
    activate(o)
    bpy.ops.object.shade_auto_smooth(angle=math.radians(deg))


def export(o, name, store):
    dg = bpy.context.evaluated_depsgraph_get()
    ev = o.evaluated_get(dg)
    me = ev.to_mesh()
    me.calc_loop_triangles()
    uvl = me.uv_layers.active.data if me.uv_layers.active else None
    cn = me.corner_normals
    pos, nor, uv = [], [], []
    mw = o.matrix_world
    nm = mw.to_3x3().inverted().transposed()
    for tri in me.loop_triangles:
        for li in tri.loops:
            vi = me.loops[li].vertex_index
            p = mw @ me.vertices[vi].co
            n = (nm @ cn[li].vector).normalized()
            pos += [p.x, p.z, -p.y]
            nor += [n.x, n.z, -n.y]
            if uvl:
                uv += [uvl[li].uv.x, uvl[li].uv.y]
            else:
                uv += [0.0, 0.0]
    ev.to_mesh_clear()
    store[name] = (np.array(pos, np.float32), np.array(nor, np.float32), np.array(uv, np.float32))
    print(name, len(pos) // 9, 'tris', flush=True)


def tex(kind, scale, **kw):
    t = bpy.data.textures.new(kind + str(random.random()), kind)
    t.noise_scale = scale
    for k, v in kw.items():
        setattr(t, k, v)
    return t


store = {}

# ---------------------------------------------------------------- ore cart tub
clear()
bm = bmesh.new()
bmesh.ops.create_cube(bm, size=1.0)
for v in bm.verts:
    v.co.x *= 1.36; v.co.y *= 1.28; v.co.z *= 0.70
top = [f for f in bm.faces if f.normal.z > 0.9]
bmesh.ops.delete(bm, geom=top, context='FACES')
bmesh.ops.subdivide_edges(bm, edges=bm.edges[:], cuts=14, use_grid_fill=True)
me = bpy.data.meshes.new('tub'); bm.to_mesh(me); bm.free()
tub = bpy.data.objects.new('tub', me); bpy.context.collection.objects.link(tub)
tub.location = (0, 0.33, -0.71)
d = tub.modifiers.new('dent', 'DISPLACE'); d.texture = tex('CLOUDS', 0.35, noise_depth=2); d.strength = 0.03; d.mid_level = 0.5; d.texture_coords = 'LOCAL'
s = tub.modifiers.new('solid', 'SOLIDIFY'); s.thickness = 0.03; s.offset = -1
b = tub.modifiers.new('bev', 'BEVEL'); b.width = 0.008; b.segments = 2; b.limit_method = 'ANGLE'
apply_all(tub)
parts = [tub]
# rounded rim along the top edges
for (x0, y0, x1, y1) in [(-0.68, 0.97, 0.68, 0.97), (-0.68, -0.31, 0.68, -0.31), (-0.68, -0.31, -0.68, 0.97), (0.68, -0.31, 0.68, 0.97)]:
    L = math.hypot(x1 - x0, y1 - y0)
    bpy.ops.mesh.primitive_cylinder_add(radius=0.03, depth=L + 0.06, vertices=16, location=((x0 + x1) / 2, (y0 + y1) / 2, -0.36))
    c = bpy.context.active_object
    c.rotation_euler = (0, math.pi / 2, 0) if y0 == y1 else (math.pi / 2, 0, 0)
    parts.append(c)
# vertical straps + rivets
for x in (-0.45, 0.0, 0.45):
    bpy.ops.mesh.primitive_cube_add(size=1, location=(x, 0.955, -0.71))
    st = bpy.context.active_object; st.scale = (0.07, 0.012, 0.66); parts.append(st)
    for k in range(5):
        bpy.ops.mesh.primitive_uv_sphere_add(radius=0.011, segments=10, ring_count=6, location=(x, 0.945, -0.42 - k * 0.14))
        r = bpy.context.active_object; r.scale = (1, 0.55, 1); parts.append(r)
for side in (-1, 1):
    for y in (-0.05, 0.33, 0.71):
        bpy.ops.mesh.primitive_cube_add(size=1, location=(side * 0.655, y, -0.71))
        st = bpy.context.active_object; st.scale = (0.012, 0.07, 0.66); parts.append(st)
        for k in range(5):
            bpy.ops.mesh.primitive_uv_sphere_add(radius=0.011, segments=10, ring_count=6, location=(side * 0.645, y, -0.42 - k * 0.14))
            r = bpy.context.active_object; r.scale = (0.55, 1, 1); parts.append(r)
for i in range(14):  # rivets along the outer rim
    x = -0.62 + i * 0.095
    bpy.ops.mesh.primitive_uv_sphere_add(radius=0.013, segments=10, ring_count=6, location=(x, 0.995, -0.40))
    parts.append(bpy.context.active_object)
for p in parts[1:]:
    apply_all(p)
activate(tub)
for p in parts:
    p.select_set(True)
bpy.ops.object.join()
bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
smart_uv(tub, 3.4)
auto_smooth(tub, 40)
export(tub, 'cart', store)

# ---------------------------------------------------------------- rocks
for i in range(4):
    clear()
    bpy.ops.mesh.primitive_ico_sphere_add(subdivisions=5, radius=1.0)
    r = bpy.context.active_object
    r.scale = (1.0, 0.8 + random.random() * 0.4, 0.55 + random.random() * 0.35)
    bpy.ops.object.transform_apply(scale=True)
    m1 = r.modifiers.new('v', 'DISPLACE'); m1.texture = tex('VORONOI', 0.6 + random.random() * 0.3, distance_metric='DISTANCE'); m1.strength = 0.45; m1.texture_coords = 'LOCAL'
    m2 = r.modifiers.new('n', 'DISPLACE'); m2.texture = tex('CLOUDS', 0.25, noise_depth=4); m2.strength = 0.18; m2.texture_coords = 'LOCAL'
    m3 = r.modifiers.new('n2', 'DISPLACE'); m3.texture = tex('CLOUDS', 0.07, noise_depth=2); m3.strength = 0.04; m3.texture_coords = 'LOCAL'
    dc = r.modifiers.new('d', 'DECIMATE'); dc.ratio = 0.12
    apply_all(r)
    for v in r.data.vertices:  # flatten the base so it sits in water
        if v.co.z < -0.25:
            v.co.z = -0.25 + (v.co.z + 0.25) * 0.15
    smart_uv(r, 0.7)
    auto_smooth(r, 55)
    export(r, f'rock{i}', store)

# ---------------------------------------------------------------- icebergs
for i in range(2):
    clear()
    bpy.ops.mesh.primitive_cube_add(size=2)
    r = bpy.context.active_object
    sub = r.modifiers.new('s', 'SUBSURF'); sub.levels = 4; sub.subdivision_type = 'SIMPLE'
    cast = r.modifiers.new('c', 'CAST'); cast.factor = 0.55
    m1 = r.modifiers.new('v', 'DISPLACE'); m1.texture = tex('VORONOI', 0.9, distance_metric='DISTANCE'); m1.strength = 0.6; m1.texture_coords = 'LOCAL'
    m2 = r.modifiers.new('n', 'DISPLACE'); m2.texture = tex('CLOUDS', 0.4, noise_depth=3); m2.strength = 0.25; m2.texture_coords = 'LOCAL'
    dc = r.modifiers.new('d', 'DECIMATE'); dc.decimate_type = 'COLLAPSE'; dc.ratio = 0.08
    apply_all(r)
    r.scale = (1.0, 0.9, 1.7 + i * 0.5)
    bpy.ops.object.transform_apply(scale=True)
    for v in r.data.vertices:
        if v.co.z < -0.6:
            v.co.z = -0.6
    smart_uv(r, 0.5, 40)
    auto_smooth(r, 25)
    export(r, f'berg{i}', store)

# ---------------------------------------------------------------- write
index, blobs, off = {}, [], 0
for name, (p, n, u) in store.items():
    cnt = len(p) // 3
    index[name] = {'count': cnt, 'offset': off}
    for arr in (p, n, u):
        blobs.append(arr.tobytes())
        off += arr.nbytes
with open(f'{OUT}/models.bin', 'wb') as f:
    for bl in blobs:
        f.write(bl)
with open(f'{OUT}/models.json', 'w') as f:
    json.dump(index, f)
print('wrote', off, 'bytes')
