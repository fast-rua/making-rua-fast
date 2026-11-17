"""MuJoCo scene of the bimanual I2RT YAM station (`yam_station_linear_4310_d405`, i2rt v1.4.0).

i2rt's station MJCF carries kinematics and inertia only. This module adds what a simulation needs: the table, the
objects, one torque motor per arm joint and per gripper, joint armature and friction, finger friction, and the three
D405 cameras. The world frame is the cell's public frame: the origin midway between the arm bases on the table top,
+x forward, +z up, so the left base is at y = +0.305 and the right one at y = -0.305 (ORA's `yam_station` convention).
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field

import mujoco
import numpy as np

I2RT_ROOT = os.environ.get("I2RT_ROOT", os.path.expanduser("~/robot-cell/i2rt"))   # an i2rt v1.4.0 checkout
STATION_XML = os.path.join(I2RT_ROOT, "i2rt/robot_models/station/yam_station_linear_4310_d405/"
                                      "yam_station_linear_4310_d405.xml")
STATION_URDF = STATION_XML[:-4] + ".urdf"
STATION_TO_WORLD_Y = 0.305          # station frame (left base at the origin) -> public world frame
ARMS = ("left", "right")
ARM_JOINTS = {arm: [f"{arm}_joint{i}" for i in range(1, 7)] for arm in ARMS}
FINGER_JOINTS = {arm: (f"{arm}_joint7", f"{arm}_joint8") for arm in ARMS}
TCP_SITES = {arm: f"tcp_{arm}" for arm in ARMS}
CAMERAS = {"left_wrist": "left_camera", "right_wrist": "right_camera", "top": "top_camera"}   # name -> body
FINGER_OPEN_M = 0.04695              # per finger; the opening between the pads is twice the joint7 travel
# Motor limits (DM4340 on joints 1-3, DM4310 on 4-6 and the gripper), from ORA's yam_station table.
TORQUE_LIMIT_NM = [28.0, 28.0, 28.0, 10.0, 10.0, 10.0]
ARMATURE = [0.012, 0.012, 0.012, 0.004, 0.004, 0.004]
GRIPPER_FORCE_LIMIT_N = 50.0             # squeeze per pad, as i2rt's force limiter means it (default 50 N)
PAD_HALF = (0.010, 0.006, 0.0415)         # cloth pad: box half extents in the tip body frame (y = jaw axis, +z up the
                                          # finger): 2 cm wide, 12 mm thick, from the fingertip up to the housing
PAD_HALF_T = 0.006                        # half thickness of the finger pads (along the jaw axis)
PAD_FRICTION = (0.9, 0.0015, 0.001)       # rubber-like pads; torsional: a ~2 mm contact patch
PAD_SOLMIX = 1000.0                       # the pads' stiff solref sets their contacts (not averaged with a soft
                                          # object's, which let a squeezed handle sink 0.6 mm into the pads)
TORSIONAL_FRICTION = 0.0015               # every geom's: MuJoCo takes the larger of a pair's values, so the pads' own
                                          # 0.0015 reaches their contacts only if the objects' is no larger
CONTACT_SOLREF = (0.005, 1.0)             # contacts of every object and fixture that sets none of its own: wood and
CONTACT_SOLIMP = (0.95, 0.99, 0.001)      # plastic are hard, and at MuJoCo's default (0.02 s, impedance 0.9-0.95)
                                          # stacked blocks sank 1-2 mm into each other, a dropped key 10 mm into the
                                          # table and a pressed power strip 2.4 mm (now 0.01, 1.4 and 0.2 mm)
CONTACT_SOLMIX = 100.0                    # ... and their values win over the table's (so an object on the table is
                                          # stiff too) but not over the pads' (PAD_SOLMIX). The table keeps MuJoCo's
                                          # default for the cloth, which lies on it as before: on a stiff table the
                                          # towel's corner pinch caught no vertex between the finger faces
MUJOCO_SOLIMP = (0.9, 0.95, 0.001)        # MuJoCo's default impedance: an object part that sets solref alone keeps it
HOUSING_BOX = ((0.0, 0.0, 0.0265), (0.051, 0.0325, 0.037))   # gripper housing (gripper frame), for cloth contact
CLOTH_GRIP_SLOTS = 12                     # per arm: cloth vertices a closed gripper can hold (see the backend)
CLOTH_GRIP_SOLIMP = (0.99, 0.999, 0.001)  # their welds' impedance, as the cloth's own edges (stiffer blew up there)
FINGERTIP_BELOW_FACET = 0.002             # the fingertip's end is 2 mm below the facet centre (the TCP when closed)
PAD_MARGIN = 0.003                        # cloth pads: contacts start 3 mm out from the box, which sits 3 mm back
                                          # (and 3 mm up), so cloth meets the real finger face: without the margin,
                                          # squeezed cloth sinks into the box and stays caught after the jaws open
# i2rt's default gains for the YAM (yam.yaml): stiff shoulder and elbow, soft wrist.
DEFAULT_KP = [80.0, 80.0, 80.0, 10.0, 10.0, 10.0]
DEFAULT_KD = [5.0, 5.0, 5.0, 1.5, 1.5, 1.5]
GRIPPER_KP, GRIPPER_KD = 1000.0, 20.0           # N/m, N s/m on the finger slide, per pad (the motor: twice that)
# Soft stop: no stiffness, only this damping on top of the gravity compensation. More than i2rt's grav-comp idle
# (0.1, 0.1, 0.1, 0.3, 0.05, 0.05), so a gravity-model error drifts slowly; still easy to push away by hand.
SOFT_KD = [2.0, 2.0, 2.0, 0.5, 0.5, 0.5]


# The real fingertip's contact face (i2rt's linear-4310 tip mesh, which matches its CAD, ray cast along the jaw axis;
# heights h above the fingertip's end, x across the finger, mm): solid and flush up to 15 mm, 12 mm wide at the end and
# widening 0.45 mm per mm of height; above that a window opens in the middle between two side rails, ~10 mm wide, that
# slant outward 0.4 mm per mm up to 59 mm and lean back 0.06 mm per mm (2.7 mm behind the tip's face at the top); from
# 60 to 72 mm a plate 61-68 mm wide, 2.7-3.4 mm back. As convex prisms 12 mm deep: (outline, set back at 15 mm, lean).
FINGER_LEAN = 0.061                       # mm set back per mm of height
FINGER_PADS = (
    (((-6.0, 0.0), (6.0, 0.0), (12.75, 15.0), (-12.75, 15.0)), 0.0, 0.0),                     # the tip
    (((2.61, 15.0), (12.5, 15.0), (30.1, 59.0), (19.77, 59.0)), 0.2, FINGER_LEAN),            # the rails (0.2 mm
    (((-2.61, 15.0), (-12.5, 15.0), (-30.1, 59.0), (-19.77, 59.0)), 0.2, FINGER_LEAN),        #   back: the scan's spread)
    (((-30.5, 60.0), (30.5, 60.0), (34.0, 72.0), (-34.0, 72.0)), 2.74 - 45 * FINGER_LEAN, FINGER_LEAN))   # the plate


def _finger_outline(h):
    """The face's half-width and its window's half-width at heights h (mm)."""
    h = np.asarray(h, float)
    tip, rail, plate = (h >= 0) & (h <= 15), (h > 15) & (h <= 59), (h >= 60) & (h <= 72)
    half = np.where(tip, 6.0 + 0.45 * h, np.where(rail, 12.5 + 0.4 * (h - 15), np.where(plate, 30.5 + 0.29 * (h - 60),
                                                                                          -1.0)))
    window = np.where(rail, 2.61 + 0.39 * (h - 15), 0.0)
    return half, window


def on_finger_face(x, h, tol: float = 0.0):
    """Whether points x across the finger and h above its end (m, tip body frame; arrays or numbers) lie on the real
    finger face, or within tol (m) of it (beside it, in its window, below its end or above its top)."""
    h = np.asarray(h, float) * 1000
    x = np.abs(np.asarray(x, float)) * 1000
    if not tol:
        half, window = _finger_outline(h)
        return (x <= half) & (x >= window)
    t = tol * 1000
    hs = [np.clip(h + dh, 0.0, 72.0) for dh in (-t, 0.0, t)]          # the outline's widest and its window's
    half = np.max([_finger_outline(k)[0] for k in hs], axis=0)       # narrowest within tol up or down
    window = np.min([_finger_outline(k)[1] for k in hs], axis=0)
    return (x <= half + t) & (x >= window - t) & (h >= -t) & (h <= 72.0 + t)


def finger_half_width(h: float) -> float:
    """How far the real finger face reaches to either side of its centre line at h above its end (m)."""
    return max(float(_finger_outline(h * 1000)[0]), 0.0) / 1000


@dataclass
class Part:
    """One more geom of an object, placed in the object's frame (e.g. a paddle's blade next to its handle)."""
    name: str
    shape: str = "box"
    size: tuple = (0.02, 0.02, 0.02)
    pos: tuple = (0.0, 0.0, 0.0)
    quat: tuple = (1, 0, 0, 0)
    mass: float = 0.01
    rgba: tuple = (0.5, 0.5, 0.5, 1)
    friction: tuple = (1.0, 0.01, 0.001)
    solref: tuple | None = None    # contact softness/bounce (MuJoCo solref, solimp); the higher priority geom of a
    solimp: tuple | None = None    #   touching pair sets the contact's parameters
    priority: int = 0
    condim: int = 4


@dataclass
class Obj:
    """A free rigid object on the table: shape box|cylinder|sphere, size in MuJoCo's convention (half extents)."""
    name: str
    shape: str = "box"
    size: tuple = (0.02, 0.02, 0.02)
    pos: tuple = (0.35, 0.0, 0.02)
    quat: tuple = (1, 0, 0, 0)
    mass: float = 0.05
    rgba: tuple = (0.85, 0.2, 0.2, 1)
    friction: tuple = (1.0, 0.01, 0.001)
    fixed: bool = False            # True: welded to the world (a fixture)
    parts: tuple = ()              # further geoms (Part) of the same rigid body, in its frame
    solref: tuple | None = None    # contact parameters of the main geom (see Part)
    solimp: tuple | None = None
    priority: int = 0
    condim: int = 4
    hollow: bool = False           # a thin shell (a ping-pong ball): inertia of the surface, not of a solid
    air: bool = False              # air drag and lift (MuJoCo's ellipsoid fluid model; needs SceneConfig.air)
    drag: float = 0.25             # with air: MuJoCo's blunt drag coefficient (C rho A v^2: 0.25 is a sphere's Cd 0.5)
    magnus: float = 0.25           # ... and its Magnus lift coefficient
    bounce_noise: tuple | None = None  # (speed sigma, direction sigma in deg): each time a contact of it ends, its
                                   # velocity is scaled and turned by random amounts (an uneven surface, a seam)
    vel: tuple = (0, 0, 0, 0, 0, 0)  # linear and angular velocity (world) it starts with, or is let go with
    release_t: float | None = None   # held still at pos until this time (a person drops it), then let go with vel
    handover: bool = False         # held still at pos until a gripper closes on it (a person hands it over)


@dataclass
class Cloth:
    """A cloth (a MuJoCo flex): a flat triangulated sheet given in its own frame, laid on the table at pos/yaw."""
    name: str
    points: np.ndarray                  # N x 2, metres, in the cloth frame
    triangles: np.ndarray               # M x 3 vertex indices
    pos: tuple = (0.3, 0.0, 0.004)      # where the cloth frame's origin goes (z: just above the table)
    yaw: float = 0.0
    mass: float = 0.08                  # whole cloth, kg
    radius: float = 0.003               # collision half thickness
    young: float = 5e3                  # Pa
    poisson: float = 0.2
    thickness: float = 0.002            # for bending stiffness
    damping: float = 0.0005
    friction: tuple = (1.2, 0.01, 0.001)
    # the edge-length constraints (they keep the cloth from stretching). MuJoCo's soft constraints are as stiff as
    # the mass they move, and a cloth vertex weighs a fraction of a gram, so at the defaults (0.02 s, impedance
    # 0.9-0.95) a towel dragged by two corners stretched its edges near the hands to twice their length, and over a
    # table edge it stayed ~20% longer. A 0.008 s time constant (2 steps) and impedance 0.99-0.999: 0.2% on average,
    # under 6% anywhere (stiffer, 0.999-0.9999, blew up)
    edge_solref: tuple | None = (0.008, 1.0)
    edge_solimp: tuple | None = (0.99, 0.999, 0.001)
    rgba: tuple = (0.2, 0.5, 0.85, 1)
    initial: np.ndarray | None = None   # N x 3 world positions to start from (e.g. crumpled); None: flat at pos/yaw

    def world_points(self) -> np.ndarray:
        c, s = np.cos(self.yaw), np.sin(self.yaw)
        xy = self.points @ np.array([[c, s], [-s, c]]) + np.asarray(self.pos[:2])
        return np.c_[xy, np.full(len(xy), self.pos[2])]


@dataclass
class Rope:
    """A rope or cord: a chain of rigid capsule links joined end to end by ball joints (connect constraints), each link
    a free body. In these maximal coordinates a contact on one link involves only that link's dofs, so a 1.5 m rope of
    150 links steps in about 1 ms; MuJoCo's cable (a chain of ball joints) took 21 ms per step at 60 links lying on the
    table, every contact row reaching back to the root. Bending and twist are an elastic rod whose couples the backend
    applies before every physics step (explicit, so each link's rotational inertia is raised to ROPE_INERTIA_RULE x its
    joint stiffness x dt^2; that slows only the shortest waves). Links (collision bit 4) touch the table, objects, the
    rigid finger pads, the gripper housings and each other (not their direct neighbours), not the arm links (their
    hulls are far larger than the arm) nor the cloth pads."""
    name: str
    points: np.ndarray                  # N x 3 world centre line at the start (any spacing): resampled to equal links
    radius: float = 0.004
    link: float = 0.01                  # link length (m); the last link takes the remainder
    density: float = 1100.0             # kg/m^3
    young: float = 1e6                  # Pa: bending stiffness E I / l per joint; 0 = a free chain (beads on a cord)
    poisson: float = 0.4                # twist: G J / l, G = E / 2 (1 + nu), J = 2 I
    rest: str = "straight"              # "straight", or "initial": the start shape is stress free
    friction: tuple = (1.0, 0.005, 0.0001)
    rgba: tuple = (0.85, 0.55, 0.25, 1)
    fixed_links: tuple = ()             # links held by the world (a clamp, a staple): no joint
    drag: float = 1e-4                  # each link's damping against the world, N s/m and N m s (implicit): air drag
                                        # for the rope (a 12 mm link: ~8 s), and it stops a link spinning in place
                                        # (~20 ms), which the explicit rod and the connect rows otherwise keep going
    # contact softness of the links (see Part), set by the links in every pair (ROPE_SOLMIX, above the pads' own).
    # A link weighs ~0.6 g, and MuJoCo's soft contact yields in proportion to the mass it moves: at the pads'
    # impedance (0.9-0.95) a 15 N pinch crushed the 8 mm rope to 0.3 mm and it slid out; at (0.95, 0.99) 50 N crushed
    # it to 1.2 mm; at (0.95, 0.999) a jaws-down pinch at the fingertips holds it at 6.2-6.8 mm from 5 to 50 N, through
    # a lift and a carry (with ROPE_NOSLIP)
    solref: tuple | None = (0.004, 1.0)
    solimp: tuple | None = (0.95, 0.999, 0.001)
    stripes: int = 0                    # every `stripes`-th link darker (so a twist or slide shows on camera)
    joint_solref: tuple | None = None   # the links' ball joints (connect rows); None = ROPE_CONNECT_SOLREF. A free chain
                                        # (young 0) draped over a post gained energy at 0.004 s (2 x dt) until it tore
                                        # apart within 1 s; at 0.008 it settles to 1 mm/s in 2 s

    def nodes(self) -> np.ndarray:
        """The link end points (n + 1 x 3), at equal arc length along the given centre line."""
        P = np.asarray(self.points, float)
        cum = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))]
        n = max(1, int(round(cum[-1] / self.link)))
        s = np.linspace(0.0, cum[-1], n + 1)
        return np.stack([np.interp(s, cum, P[:, i]) for i in range(3)], 1)


ROPE_INERTIA_RULE = 4.0                 # link rotational inertia >= this x joint stiffness x dt^2 (2 is the edge)
ROPE_CONNECT_SOLREF = (0.004, 1.0)
ROPE_SOLMIX = 1e6                       # above the pads' PAD_SOLMIX: a rope link's contacts take its own solref/solimp
ROPE_NOSLIP = 30                        # noslip iterations in a scene with ropes: with the station's 2 (and with 10) a
                                        # pinched link crept out of the jaws within ~10 s of carrying (5-15 N)


@dataclass
class SceneConfig:
    objects: list = field(default_factory=list)
    cloths: list = field(default_factory=list)
    ropes: list = field(default_factory=list)
    table_top_z: float = 0.0
    table_x: tuple = (-0.20, 0.90)      # table extent along x (forward); a front edge within reach gives a ledge
    timestep: float = 0.002
    camera_width: int = 640
    camera_height: int = 480
    camera_fovy_deg: float = 58.0
    initial_q: dict = field(default_factory=dict)   # joint name -> start value (default: READY_Q, jaws closed)
    air: bool = False                   # air at room conditions (density 1.204 kg/m^3, viscosity 1.8e-5 Pa s)
    watch: tuple = ()                   # objects whose contacts and path the backend logs for the success check
    wind: dict | None = None            # moving air (needs air): {"mean": (x, y, z) m/s, "gust": horizontal sigma m/s,
                                        #   "gust_z": vertical sigma, "tau": s}; gusts are an Ornstein-Uhlenbeck process
    noise_seed: int | None = None       # seeds the backend's random streams (bounce noise, gusts), drawn in step order
    noise_info: dict = field(default_factory=dict)  # what the scene drew (for the record; never sent to an agent)
    fence: dict | None = None           # a workspace fence for the TCPs (controller.parse_fence); robotd --fence wins
    start_grips: dict = field(default_factory=dict)  # arm -> squeeze (N per pad): the gripper closes on what is in it
                                        # as the clock starts (a tool handed over before the episode)
    gravity_comp_scale: float = 1.0     # the arms' gravity compensation x this (tests: a wrong model leaves a steady
                                        # joint offset, as the real YAM's i2rt feed-forward does)
    joint_torque_bias: dict = field(default_factory=dict)  # joint name -> a constant torque (Nm) on top of it (tests)
    joint_stiction: dict = field(default_factory=dict)  # joint name -> (breakaway, sliding) friction (Nm): a joint at
                                        # rest holds against up to breakaway and slides against less (tests: the real
                                        # YAM's joints stick ~0.3-0.5 deg off a target, then jump past it)


def _shift_world_children(spec: mujoco.MjSpec, dy: float) -> None:
    for body in spec.worldbody.bodies:
        body.pos = [body.pos[0], body.pos[1] + dy, body.pos[2]]


def build_spec(cfg: SceneConfig | None = None) -> mujoco.MjSpec:
    cfg = cfg or SceneConfig()
    spec = mujoco.MjSpec.from_file(STATION_XML)
    spec.option.timestep = cfg.timestep
    spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    spec.option.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
    spec.option.noslip_iterations = 2
    spec.visual.global_.offwidth = max(cfg.camera_width, 640)
    spec.visual.global_.offheight = max(cfg.camera_height, 480)
    _shift_world_children(spec, STATION_TO_WORLD_Y)

    # Joint dynamics: reflected rotor inertia, a little damping and friction (real motors are not ideal).
    for arm in ARMS:
        for i, name in enumerate(ARM_JOINTS[arm]):
            j = spec.joint(name)
            j.armature = ARMATURE[i]
            j.damping = [0.05, 0.0, 0.0]
            j.frictionloss = 0.02
        for name in FINGER_JOINTS[arm]:
            j = spec.joint(name)
            j.armature = 0.002
            j.damping = [2.0, 0.0, 0.0]
            j.solref_limit = [0.004, 1.0]       # the closed stop is hard: at MuJoCo's default (0.02 s) a grip whose
                                                # object slips out drove the fingers 6 mm past each other
    # Finger pads. MuJoCo collides a mesh as its convex hull, and the L-shaped fingertip meshes have hulls far larger
    # than the real pads (open "7 cm", the hulls almost touch). So the tip meshes are visual only, and each finger
    # gets pads whose inner faces lie on its facet: the facets meet at the TCP site when the jaws are closed,
    # and each finger slides out along its body y axis, so the pad gap is exactly 2 x joint7 (the API's width).
    # Rigid objects touch the finger pads <arm>_<tip>_pad<k>: convex prisms on the real fingertip's face (FINGER_PADS;
    # MuJoCo gives a flat face pressed on one a contact at each corner of their overlap), no margin, so they touch only
    # where the finger is (one box 20 mm wide down to the tip, with a 3 mm margin below it, pinched coins and keys
    # where the finger has no surface). Cloth touches the cloth pad <arm>_<tip>_cloth.
    facets = _facet_points()
    for geom in spec.geoms:
        body = geom.parent
        if body is not None and ("tip_left" in body.name or "tip_right" in body.name):
            geom.contype, geom.conaffinity = 0, 0
    for arm in ARMS:
        for side, inward in (("tip_left", 1.0), ("tip_right", -1.0)):
            name = f"{arm}_{side}"
            for k, (outline, back, lean) in enumerate(FINGER_PADS):
                front = [facets[name] + np.array([x, -inward * (back + lean * (h - 15)), h]) / 1000
                         - [0, 0, FINGERTIP_BELOW_FACET] for x, h in outline]
                verts = np.concatenate([front, np.array(front) - [0, inward * 2 * PAD_HALF_T, 0]])
                spec.add_mesh(name=f"{name}_pad{k}", uservert=verts.ravel().tolist())
                spec.body(name).add_geom(name=f"{name}_pad{k}", type=mujoco.mjtGeom.mjGEOM_MESH,
                                         meshname=f"{name}_pad{k}", friction=list(PAD_FRICTION), condim=4,
                                         solref=[0.004, 1.0], solmix=PAD_SOLMIX, contype=1, conaffinity=1, group=3,
                                         rgba=[0.15, 0.15, 0.15, 1], mass=0.0)
            # cloth pad (collision bit 2 only): inner face on the facet, up the finger so cloth cannot drape over it;
            # the box starts PAD_MARGIN above the fingertip, so its contacts reach down to the real fingertip only
            centre = facets[name] + np.array([0.0, -inward * (PAD_HALF[1] + PAD_MARGIN),
                                              PAD_HALF[2] - FINGERTIP_BELOW_FACET + PAD_MARGIN / 2])
            size = [PAD_HALF[0], PAD_HALF[1], PAD_HALF[2] - PAD_MARGIN / 2]
            spec.body(name).add_geom(name=f"{name}_cloth", type=mujoco.mjtGeom.mjGEOM_BOX, size=size,
                                     margin=PAD_MARGIN, pos=centre.tolist(), friction=list(PAD_FRICTION), condim=4,
                                     solref=[0.004, 1.0], contype=0, conaffinity=2, group=3, rgba=[0, 0, 0, 0],
                                     mass=0.0)

    # The housing's mesh collides as its hull (too big); cloth gets this box instead, so it cannot slip between the
    # finger roots and the housing (collision bit 2 only: cloth and table; invisible to the cameras).
    for arm in ARMS:
        spec.body(f"{arm}_gripper").add_geom(name=f"{arm}_housing", type=mujoco.mjtGeom.mjGEOM_BOX,
                                            pos=list(HOUSING_BOX[0]), size=list(HOUSING_BOX[1]), contype=2,
                                            conaffinity=2, group=3, rgba=[0, 0, 0, 0], mass=0.0, margin=PAD_MARGIN)

    # The bases hang off the world body, so MuJoCo's parent-child contact filter does not cover base/link1.
    for arm in ARMS:
        spec.add_exclude(bodyname1=f"{arm}_base", bodyname2=f"{arm}_link1")
    for eq in spec.equalities:                  # the finger coupling: stiff, so both fingers move together
        eq.solref = [0.004, 1.0]

    wb = spec.worldbody
    tex = spec.add_texture(name="grid", type=mujoco.mjtTexture.mjTEXTURE_2D, builtin=mujoco.mjtBuiltin.mjBUILTIN_CHECKER,
                           width=256, height=256, rgb1=[0.82, 0.82, 0.8], rgb2=[0.74, 0.74, 0.72])
    mat = spec.add_material(name="floor", texrepeat=[6, 6])
    mat.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = "grid"
    wb.add_geom(name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[3, 3, 0.1], pos=[0, 0, -0.75], material="floor")
    tx0, tx1 = cfg.table_x
    wb.add_geom(name="table", type=mujoco.mjtGeom.mjGEOM_BOX, size=[(tx1 - tx0) / 2, 0.7, 0.025],
                pos=[(tx0 + tx1) / 2, 0.0, cfg.table_top_z - 0.025], rgba=[0.62, 0.5, 0.38, 1],
                friction=[0.8, 0.01, 0.001])
    wb.add_light(name="key", pos=[0.4, 0.0, 2.0], dir=[0, 0, -1], diffuse=[0.8, 0.8, 0.8],
                 type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL)
    wb.add_light(name="fill", pos=[-1.0, 1.0, 1.5], dir=[0.6, -0.6, -0.6], diffuse=[0.3, 0.3, 0.3],
                 type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL)

    for c in cfg.cloths:
        _add_cloth(spec, c)
    if cfg.cloths:
        # MuJoCo 3.17 integrates flex elasticity only with the discrete integrator (no PGS, no noslip with flex).
        # CG instead of Newton: Newton factorises the whole system (1,000+ cloth dofs) every step and was 2-4x
        # slower; cloth resting on the table keeps ~3 contacts per triangle. A 4 ms step (one per control tick) is
        # stable here: a single-layer shirt on a 2 cm grid with self-collision then runs about 2x real time.
        spec.option.integrator = mujoco.mjtIntegrator.mjINT_DISCRETE
        spec.option.noslip_iterations = 0
        spec.option.solver = mujoco.mjtSolver.mjSOL_CG
        spec.option.timestep = 0.004

    if cfg.air:
        spec.option.density, spec.option.viscosity = 1.204, 1.8e-5
    # Every geom added from here on (objects, fixtures, ropes, the tasks' own geoms) takes the stiff CONTACT_SOLREF,
    # CONTACT_SOLIMP and CONTACT_SOLMIX unless it sets its own; the station's links, the pads, the table and the cloth
    # keep theirs (see CONTACT_SOLMIX).
    spec.default.geom.solref = list(CONTACT_SOLREF)
    spec.default.geom.solimp = list(CONTACT_SOLIMP) + [0.5, 2.0]
    spec.default.geom.solmix = CONTACT_SOLMIX
    for o in cfg.objects:
        body = wb.add_body(name=o.name, pos=list(o.pos), quat=list(o.quat))
        if not o.fixed:
            body.add_freejoint(name=f"{o.name}_free")
        for p in object_parts(o):
            g = body.add_geom(name=p.name, type=GEOM_TYPES[p.shape], size=list(p.size), pos=list(p.pos),
                              quat=list(p.quat), mass=p.mass, rgba=list(p.rgba), friction=list(p.friction),
                              condim=p.condim, priority=p.priority)
            if p.solref is not None:
                g.solref = list(p.solref)
            if p.solimp is not None or p.solref is not None:     # (dmin, dmax, width[, midpoint, power])
                g.solimp = (list(p.solimp or MUJOCO_SOLIMP) + [0.5, 2.0])[:5]
            if o.hollow:
                g.typeinertia = mujoco.mjtGeomInertia.mjINERTIA_SHELL
            if o.air:
                # blunt drag, slender drag, angular drag, Kutta lift, Magnus lift. MuJoCo's blunt drag is
                # C rho A v^2 (no 1/2): 0.25 is a sphere's drag coefficient of 0.5 (a table-tennis ball then falls at
                # most ~8.5 m/s); Magnus 0.25 gives a lift coefficient ~0.3 at a spin ratio r w / v of 0.5
                g.fluid_ellipsoid = 1.0
                g.fluid_coefs = [o.drag, 0.25, 1.5, 1.0, o.magnus]
        if (o.release_t is not None or o.handover) and not o.fixed:
            # a person holds it still where it starts: a weld to the world, let go by the backend
            # data: the anchor (in body2's frame, here the world: the object's start position; at the world origin
            # the object swung round it like a pendulum), the relative pose from the start (zero quaternion), torque
            # scale 1
            spec.add_equality(type=mujoco.mjtEq.mjEQ_WELD, name=f"{o.name}_hold", name1=o.name,
                              objtype=mujoco.mjtObj.mjOBJ_BODY, solref=[0.005, 1.0],
                              data=list(map(float, o.pos)) + [0.0] * 7 + [1.0, 0.0])

    for r in cfg.ropes:
        _add_rope(spec, r)
    if cfg.ropes:
        spec.option.noslip_iterations = max(spec.option.noslip_iterations, ROPE_NOSLIP)
        # CG, not Newton: where a rope lies across itself (a knot) its contacts couple links far apart along the chain,
        # which fills in Newton's sparse factor (the knot scene ran 5x slower than real time, 3x faster with CG); the
        # pinch holds (5-50 N, lift, carry, shake) and the knot primitives came out the same with either
        spec.option.solver = mujoco.mjtSolver.mjSOL_CG

    object_geoms = {p.name for o in cfg.objects for p in object_parts(o)}
    # Cloth (collision bit 2) touches the table, the floor, objects and the cloth pads, but not the arm links: their
    # collision geometry is each mesh's convex hull, far larger than the real parts (the gripper housing's hull
    # reaches down around the fingers and traps cloth between itself and the pads).
    for g in spec.geoms:
        if g.name in ("table", "floor") or g.name in object_geoms:
            g.conaffinity = g.conaffinity | 2
    # Ropes (collision bit 4) touch the table, the floor, objects, the finger pads (the rigid ones), the gripper
    # housings and each other: not the arm links (hulls) and not the cloth pads (a margin box)
    if cfg.ropes:
        for g in spec.geoms:
            if g.name in ("table", "floor") or g.name in object_geoms or re.search(r"_pad\d+$", g.name) \
                    or g.name.endswith("_housing"):
                g.conaffinity = g.conaffinity | 4
    for g in spec.geoms:
        g.friction = [g.friction[0], TORSIONAL_FRICTION, g.friction[2]]

    # Cloth held in a closed gripper: a pinch traps only a few cloth vertices, and in simulation they slip out under
    # the cloth's weight although a real pinch holds. So the backend welds the vertices between the closed pads to a
    # finger until the pinch lets go, using these inactive connect constraints (retargeted at run time). They are as
    # stiff as the solver steps stably (CLOTH_GRIP_SOLIMP): at 0.01 s and MuJoCo's default impedance, a welded vertex
    # sagged 4 mm under the towel's weight, below the fingertips and out of the jaws.
    if cfg.cloths:
        for arm in ARMS:
            for k in range(CLOTH_GRIP_SLOTS):
                eq = spec.add_equality(type=mujoco.mjtEq.mjEQ_CONNECT, name=f"{arm}_cloth_grip_{k}",
                                       name1=f"{arm}_tip_left", name2=f"{cfg.cloths[0].name}_flex_0",
                                       objtype=mujoco.mjtObj.mjOBJ_BODY, data=[0.0] * 12)
                eq.active = False
                eq.solref = [2 * spec.option.timestep, 1.0]
                eq.solimp = list(CLOTH_GRIP_SOLIMP) + [0.5, 2.0]

    # Cameras: the station's camera bodies are optical frames (+z forward, +y down); a MuJoCo camera looks along
    # its -z with +y up, so it sits in the optical frame turned half a turn about x.
    for cam, body_name in CAMERAS.items():
        spec.body(body_name).add_camera(name=cam, quat=[0, 1, 0, 0], fovy=cfg.camera_fovy_deg)

    # Motors: joint-space PD servos, force = kp*(ctrl - q) - kd*dq, clamped to the motor's torque. kp and kd are
    # changed at run time (the driver's per-command gains), as i2rt's command_joint_state(pos, vel, kp, kd) does on
    # the real arm; the backend adds gravity compensation, as i2rt does.
    for arm in ARMS:
        for i, name in enumerate(ARM_JOINTS[arm]):
            lim = TORQUE_LIMIT_NM[i]
            spec.add_actuator(name=f"{name}_motor", target=name, trntype=mujoco.mjtTrn.mjTRN_JOINT,
                              gaintype=mujoco.mjtGain.mjGAIN_FIXED, gainprm=[DEFAULT_KP[i]] + [0] * 9,
                              biastype=mujoco.mjtBias.mjBIAS_AFFINE,
                              biasprm=[0, -DEFAULT_KP[i], -DEFAULT_KD[i]] + [0] * 7,
                              forcelimited=True, forcerange=[-lim, lim])
        # the gripper motor drives joint7, and joint8 follows it (an equality): its force is both pads' squeeze, so
        # gains and force limit are twice the per-pad values the driver uses (the backend scales them)
        spec.add_actuator(name=f"{arm}_gripper_motor", target=FINGER_JOINTS[arm][0], trntype=mujoco.mjtTrn.mjTRN_JOINT,
                          gaintype=mujoco.mjtGain.mjGAIN_FIXED, gainprm=[2 * GRIPPER_KP] + [0] * 9,
                          biastype=mujoco.mjtBias.mjBIAS_AFFINE,
                          biasprm=[0, -2 * GRIPPER_KP, -2 * GRIPPER_KD] + [0] * 7, forcelimited=True,
                          forcerange=[-2 * GRIPPER_FORCE_LIMIT_N, 2 * GRIPPER_FORCE_LIMIT_N])
    return spec


def _add_cloth(spec: mujoco.MjSpec, c: Cloth) -> None:
    P = c.world_points()
    xml = f"""<mujoco><worldbody>
<flexcomp name="flex" type="direct" dim="2" radius="{c.radius}" mass="{c.mass}" rgba="{' '.join(map(str, c.rgba))}"
  point="{' '.join(f'{v:.5f}' for v in P.ravel())}" element="{' '.join(map(str, np.asarray(c.triangles).ravel()))}">
  <contact condim="3" solref="0.003 1" solimp="0.95 0.99 0.0001" selfcollide="auto" contype="2" conaffinity="2"
           friction="{' '.join(map(str, c.friction))}"/>
  <edge equality="true"{f' solref="{c.edge_solref[0]} {c.edge_solref[1]}"' if c.edge_solref else ''}{f' solimp="{" ".join(map(str, c.edge_solimp))}"' if c.edge_solimp else ''}/>
  <elasticity young="{c.young}" poisson="{c.poisson}" thickness="{c.thickness}" damping="{c.damping}" elastic2d="bend"/>
</flexcomp></worldbody></mujoco>"""
    spec.attach(mujoco.MjSpec.from_string(xml), frame=spec.worldbody.add_frame(), prefix=f"{c.name}_")


def rope_stiffness(r: Rope, link: float) -> np.ndarray:
    """A joint's stiffness about the link's own x (twist), y and z (bending), N m / rad."""
    I = np.pi * r.radius ** 4 / 4
    G = r.young / (2 * (1 + r.poisson))
    return np.array([G * 2 * I, r.young * I, r.young * I]) / link


def _add_rope(spec: mujoco.MjSpec, r: Rope) -> None:
    """The rope's links (bodies <name>_000, ...: x along the link, origin at its middle), its joints (connect
    equalities <name>_j001, ... at each shared end) and the neighbour contact exclusions."""
    P = r.nodes()
    n = len(P) - 1
    dt = spec.option.timestep
    stripe = tuple(0.55 * c for c in r.rgba[:3]) + (r.rgba[3],)
    y = None
    for k in range(n):
        a, b = P[k], P[k + 1]
        L = float(np.linalg.norm(b - a))
        x = (b - a) / L
        if y is None:                       # the first link's y horizontal, then parallel transport (no twist)
            y = np.cross([0.0, 0.0, 1.0], x)
            y = y / np.linalg.norm(y) if np.linalg.norm(y) > 1e-6 else np.array([0.0, 1.0, 0.0])
        else:
            y = y - x * (x @ y)
            y = y / np.linalg.norm(y)
        R = np.stack([x, y, np.cross(x, y)], 1)
        quat = np.zeros(4)
        mujoco.mju_mat2Quat(quat, R.ravel())
        body = spec.worldbody.add_body(name=f"{r.name}_{k:03d}", pos=((a + b) / 2).tolist(), quat=quat.tolist())
        if k not in r.fixed_links:
            j = body.add_freejoint(name=f"{r.name}_{k:03d}_free")
            j.damping = [r.drag, 0.0, 0.0]
        m = r.density * np.pi * r.radius ** 2 * L
        own = m * np.array([r.radius ** 2 / 2, L ** 2 / 12 + r.radius ** 2 / 4, L ** 2 / 12 + r.radius ** 2 / 4])
        body.explicitinertial = True
        body.ipos, body.iquat = [0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]
        body.mass = m
        body.inertia = np.maximum(own, ROPE_INERTIA_RULE * rope_stiffness(r, L) * dt ** 2).tolist()
        rgba = stripe if r.stripes and k % r.stripes == 0 else r.rgba
        body.add_geom(name=f"{r.name}_{k:03d}_g", type=mujoco.mjtGeom.mjGEOM_CAPSULE, size=[r.radius, 0, 0],
                      fromto=[-L / 2, 0, 0, L / 2, 0, 0], mass=0.0, density=0.0, friction=list(r.friction), condim=4,
                      contype=4, conaffinity=4, solmix=ROPE_SOLMIX, rgba=list(rgba))
        g = body.geoms[-1]
        if r.solref is not None:
            g.solref = list(r.solref)
        if r.solimp is not None:
            g.solimp = (list(r.solimp) + [0.5, 2.0])[:5]
        if k:
            Lp = float(np.linalg.norm(P[k] - P[k - 1]))
            spec.add_equality(type=mujoco.mjtEq.mjEQ_CONNECT, name=f"{r.name}_j{k:03d}", name1=f"{r.name}_{k - 1:03d}",
                              name2=f"{r.name}_{k:03d}", objtype=mujoco.mjtObj.mjOBJ_BODY,
                              data=[Lp / 2, 0, 0] + [0.0] * 9, solref=list(r.joint_solref or ROPE_CONNECT_SOLREF))
            spec.add_exclude(bodyname1=f"{r.name}_{k - 1:03d}", bodyname2=f"{r.name}_{k:03d}")


def _facet_points() -> dict:
    """Each fingertip's facet centre in its own body frame: where the TCP site is when the jaws are closed."""
    m = mujoco.MjSpec.from_file(STATION_XML).compile()
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    out = {}
    for arm in ARMS:
        p = d.site(TCP_SITES[arm]).xpos
        for side in ("tip_left", "tip_right"):
            b = d.body(f"{arm}_{side}")
            out[f"{arm}_{side}"] = b.xmat.reshape(3, 3).T @ (p - b.xpos)
    return out


GEOM_TYPES = {"box": mujoco.mjtGeom.mjGEOM_BOX, "cylinder": mujoco.mjtGeom.mjGEOM_CYLINDER,
              "sphere": mujoco.mjtGeom.mjGEOM_SPHERE, "capsule": mujoco.mjtGeom.mjGEOM_CAPSULE}


def object_parts(o: Obj) -> list:
    """An object's geoms: its main one (named <name>_geom, at the body origin) and then its parts."""
    main = Part(f"{o.name}_geom", o.shape, o.size, mass=o.mass, rgba=o.rgba, friction=o.friction, solref=o.solref,
                solimp=o.solimp, priority=o.priority, condim=o.condim)
    return [main] + list(o.parts)


def build(cfg: SceneConfig | None = None) -> tuple[mujoco.MjModel, mujoco.MjData]:
    model = build_spec(cfg).compile()
    return model, mujoco.MjData(model)


def kinematic_model() -> mujoco.MjModel:
    """The bare station model in the public world frame (no table, objects or motors): for kinematics and the arms'
    dynamics (link inertias plus the motors' rotor inertia, which dominates at the wrist)."""
    spec = mujoco.MjSpec.from_file(STATION_XML)
    _shift_world_children(spec, STATION_TO_WORLD_Y)
    for arm in ARMS:
        for i, name in enumerate(ARM_JOINTS[arm]):
            spec.joint(name).armature = ARMATURE[i]
    return spec.compile()


# A ready pose: forearms forward, TCP about 0.30 m ahead of each base and 0.20 m above the table, jaws down.
READY_Q = {"left": np.array([0.0, 1.0, 0.9, -0.5, 0.0, 0.0]), "right": np.array([0.0, 1.0, 0.9, -0.5, 0.0, 0.0])}
