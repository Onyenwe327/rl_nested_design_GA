"""
pipeline/xml_builder.py — MjSpec-based XML builder (gear-ratio aware)

Key properties:
- All position actuators use kp = 30
- ctrlrange is in radians (±2.094 ≈ ±120°)
- Root body is placed at the origin (0, 0, 0)
- Foot sphere on the last link for contact

Supports BOTH:
  A) New API:
     params["motor_type"] in {"robstride02","robstride03"}
     params["gear_ratio"]  in {9,15,20} (or any positive int)
  B) Legacy API:
     params["motor"] object with attributes:
       .peak_torque_Nm, .noload_rpm, .mass_kg, .J_rotor
     and optional params["gear_ratio"] (default 1)

Modeling choice for robstride03:
- Use official OUTPUT-SIDE baseline as reference (T0_out, rpm0_out, R0=9).
- External gear ratio R_ext rescales output:
    T_out = T0_out * (R_ext / R0) * η
    rpm_out = rpm0_out * (R0 / R_ext)
- Armature reflection enabled by default: J_out = J_rotor * R_ext^2.

This keeps numbers consistent with the vendor's published 60 N·m peak torque
and ~195 rpm no-load speed at the QDD 9:1 integrated reducer baseline.
"""

from __future__ import annotations
from typing import Dict, List, Optional, Any
from pathlib import Path
import math
import mujoco

# ---------------------- defaults / fallbacks ----------------------
RANGE_RAD = 2.094            # 120 degrees in radians
RANGE_DEG = 120.0
POS_KP = 30.0
FOOT_RADIUS = 0.08
DEFAULT_DENSITY = 1200.0     # kg/m^3 (plastic-like)
DEFAULT_LINK_RADIUS = 0.05   # meters
DEFAULT_NOLOAD_RPM = 410.0
DEFAULT_PEAK_TORQUE = 17.0   # N·m
DEFAULT_ARMATURE = 1.8e-4    # kg·m^2
GEARBOX_EFFICIENCY = 0.95    # simple constant efficiency (can be tabled later)
REFLECT_ARMATURE = True      # reflect J_rotor by R_ext^2 to output side

# ---------------------- motor specs (OUTPUT-SIDE baselines) ----------------------
# robstride03 numbers use official output-side baseline (QDD 9:1 integrated):
#   peak torque ≈ 60 N·m, no-load speed ≈ 195 rpm, internal_ratio_base = 9, mass ≈ 0.88 kg.
# robstride02 treated as no fixed reducer (internal_ratio_base=1) with legacy project numbers.
_MOTOR_SPECS: Dict[str, Dict[str, float]] = {
    "robstride02": {
        "peak_torque_nm_out_base": 17.0,
        "noload_rpm_out_base": 410.0,
        "internal_ratio_base": 1,      # treat as direct for baseline
        "mass_kg": 0.405,
        "J_rotor": 0.0042,
    },
    "robstride03": {
        "peak_torque_nm_out_base": 60.0,
        "noload_rpm_out_base": 195.0,
        "internal_ratio_base": 1,      # vendor QDD baseline
        "mass_kg": 0.88,               # 880 g
        "J_rotor": 0.02,               # vendor doesn't publish rotor inertia; safe default
    },
    "robstride06": {
        "peak_torque_nm_out_base": 36.0,
        "noload_rpm_out_base": 480.0,
        "internal_ratio_base": 1,
        "mass_kg": 0.62,
        "J_rotor": 0.012,
    },
}

# ---------------------- utilities ----------------------
def _compute_link_masses(
    link_lengths: List[float],
    link_masses_from_params: Optional[List[float]],
    density: float,
    radius: float,
) -> List[float]:
    """Use provided masses if available; otherwise compute solid-cylinder mass."""
    if link_masses_from_params is not None:
        assert len(link_masses_from_params) == len(link_lengths), \
            "If provided, link_masses must match link_lengths length."
        return [float(m) for m in link_masses_from_params]

    cross_area = math.pi * radius * radius
    return [density * cross_area * float(L) for L in link_lengths]


def _setup_spec_visuals(spec: "mujoco.MjSpec", link_length_sum: float) -> None:
    """Visual and global options (offscreen buffer, lighting)."""
    spec.modelname = "single_leg_mjspec"
    spec.option.gravity = [0, 0, -9.81]
    spec.option.integrator = mujoco.mjtIntegrator.mjINT_RK4
    spec.option.timestep = 0.002

    try:
        spec.visual.global_.offwidth = 1280
        spec.visual.global_.offheight = 720
        spec.visual.headlight.ambient  = [0.4, 0.4, 0.4]
        spec.visual.headlight.diffuse  = [0.8, 0.8, 0.8]
        spec.visual.headlight.specular = [0.2, 0.2, 0.2]
    except Exception:
        pass

    # Top light
    try:
        spec.worldbody.add_light(
            name="top_light",
            pos=[0.0, 0.0, link_length_sum + 1.5],
            diffuse=[0.9, 0.9, 0.9],
            specular=[0.2, 0.2, 0.2],
        )
    except Exception:
        pass

    # Ground plane
    spec.worldbody.add_geom(
        name="ground",
        type=mujoco.mjtGeom.mjGEOM_PLANE,
        size=[5.0, 5.0, 0.1],
        rgba=[0.9, 0.9, 0.9, 1.0],
        friction=[1.0, 0.1, 0.1],
    )


def _build_root(spec: "mujoco.MjSpec"):
    """Root body placed at origin (0,0,0) with slide joints on X/Z only."""
    root = spec.worldbody.add_body(name="root", pos=[0, 0, 0])
    root.add_joint(name="root_tx", type=mujoco.mjtJoint.mjJNT_SLIDE, axis=[1, 0, 0], damping=0.0)
    root.add_joint(name="root_tz", type=mujoco.mjtJoint.mjJNT_SLIDE, axis=[0, 0, 1], damping=0.0)
    root.add_geom(name="root_geom", type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[0.08, 0, 0], rgba=[0.8, 0.2, 0.2, 1.0], mass=2.5)
    return root


def _build_leg_chain(
    root,
    dof: int,
    link_lengths: List[float],
    link_masses: List[float],
    link_radius: float,
    joint_damping: float,
    joint_armature: float,
    motor_mass: float,
):
    """
    Build a vertical chain along -Z.
    Each joint: hinge around Y axis with range ±120° (radians).
    - Adds a capsule for each link with mass=link_masses[i]
    - Adds a small sphere at each hinge with mass=motor_mass (to lump motor mass)
    """
    parent = root
    for i, L in enumerate(link_lengths):
        pos = [0, 0, -link_lengths[i - 1]] if i > 0 else [0, 0, 0]
        child = parent.add_body(name=f"link_{i}", pos=pos)

        # Hinge joint (radians)
        child.add_joint(
            name=f"joint_{i}",
            type=mujoco.mjtJoint.mjJNT_HINGE,
            axis=[0, 1, 0],
            limited=True,
            range=[-RANGE_DEG, RANGE_DEG],
            damping=joint_damping,
            armature=joint_armature,
        )

        # Capsule link (from current body origin down along -Z by L)
        child.add_geom(
            name=f"geom_{i}",
            type=mujoco.mjtGeom.mjGEOM_CAPSULE,
            fromto=[0, 0, 0, 0, 0, -L],
            size=[link_radius, 0, 0],
            rgba=[0.2, 0.6, 1.0, 1.0] if i % 2 == 0 else [0.3, 0.75, 1.0, 1.0],
            mass=link_masses[i],
            friction=[1.0, 0.1, 0.1],
        )

        # Lumped motor mass at the joint location (small, dark sphere)
        child.add_geom(
            name=f"motor_lump_{i}",
            type=mujoco.mjtGeom.mjGEOM_SPHERE,
            pos=[0.0, 0.0, 0.0],
            size=[0.02, 0.0, 0.0],
            rgba=[0.1, 0.1, 0.1, 1.0],
            mass=motor_mass,
            friction=[1.0, 0.1, 0.1],
        )

        # Foot on last link
        if i == dof - 1:
            child.add_geom(
                name="foot_sphere",
                type=mujoco.mjtGeom.mjGEOM_SPHERE,
                pos=[0.0, 0.0, -L],
                size=[FOOT_RADIUS, 0.0, 0.0],
                rgba=[0.1, 0.1, 0.1, 1.0],
                friction=[1.0, 0.1, 0.1],
            )

        parent = child


def _inject_actuators(xml_text: str, dof: int, torque_cap_out: float) -> str:
    """Add per-hinge position actuators with torque cap from gearbox output."""
    lines = ['  <actuator>']
    for i in range(dof):
        lines.append(
            f'    <position name="mtr_{i}" joint="joint_{i}" '
            f'kp="{POS_KP:.1f}" gear="1" '
            f'ctrllimited="true" ctrlrange="{-RANGE_DEG:.3f} {RANGE_DEG:.3f}" '
            f'forcelimited="true" forcerange="-{torque_cap_out:.3f} {torque_cap_out:.3f}"/>'
        )
    lines.append('  </actuator>\n')
    act_block = "\n".join(lines)
    return xml_text.replace("</worldbody>", "</worldbody>\n" + act_block, 1)


def _resolve_base_from_params(params: Dict[str, Any]) -> Dict[str, float]:
    """
    Resolve OUTPUT-SIDE baseline motor spec dict with keys:
      peak_torque_nm_out_base, noload_rpm_out_base, internal_ratio_base, mass_kg, J_rotor

    Sources:
      - params["motor_type"] (preferred)
      - params["motor"] (legacy object) -> converted to a pseudo "out_base" with internal_ratio_base=1
    """
    if "motor_type" in params:
        mtype = str(params["motor_type"]).lower()
        if mtype not in _MOTOR_SPECS:
            raise ValueError(f"Unknown motor_type={mtype}. Known: {list(_MOTOR_SPECS.keys())}")
        return dict(_MOTOR_SPECS[mtype])

    # Legacy: object with attributes (treated as output-side baseline with R0=1)
    motor_obj = params.get("motor", None)
    if motor_obj is None:
        # final fallback
        return {
            "peak_torque_nm_out_base": DEFAULT_PEAK_TORQUE,
            "noload_rpm_out_base": DEFAULT_NOLOAD_RPM,
            "internal_ratio_base": 1,
            "mass_kg": 0.4,
            "J_rotor": DEFAULT_ARMATURE,
        }
    return {
        "peak_torque_nm_out_base": float(getattr(motor_obj, "peak_torque_Nm", DEFAULT_PEAK_TORQUE)),
        "noload_rpm_out_base": float(getattr(motor_obj, "noload_rpm", DEFAULT_NOLOAD_RPM)),
        "internal_ratio_base": 1,
        "mass_kg": float(getattr(motor_obj, "mass_kg", 0.4)),
        "J_rotor": float(getattr(motor_obj, "J_rotor", DEFAULT_ARMATURE)),
    }


def _motor_output_specs_from_out_base(base: dict, gear_ratio_ext: int) -> dict:
    """
    Given output-side baseline (T0_out, rpm0_out) tied to R0, apply external ratio R_ext:
        T_out   = T0_out * (R_ext / R0) * eta
        rpm_out = rpm0_out * (R0 / R_ext)
    Armature reflection (if enabled): J_out = J_rotor * R_ext^2
    """
    R0 = max(1, int(base.get("internal_ratio_base", 1)))
    R_ext = max(1, int(gear_ratio_ext))
    T0 = float(base["peak_torque_nm_out_base"])
    rpm0 = float(base["noload_rpm_out_base"])
    eta = float(GEARBOX_EFFICIENCY)

    torque_out = T0 * (R_ext / R0) * eta
    rpm_out = rpm0 * (R0 / R_ext)

    # joint damping heuristic uses baseline numbers (kept consistent with prior practice)
    joint_damping = T0 / (rpm0 * 2 * math.pi / 60.0 + 1e-9)

    J_base = float(base.get("J_rotor", DEFAULT_ARMATURE))
    armature_out = (J_base * (R_ext ** 2)) if REFLECT_ARMATURE else J_base

    return {
        "torque_cap_nm": float(torque_out),
        "noload_rpm_out": float(rpm_out),
        "joint_damping": float(joint_damping),
        "armature_out": float(armature_out),
        "motor_mass": float(base.get("mass_kg", 0.4)),
        "gear_ratio": R_ext,
        "efficiency": eta,
    }


def build_xml_from_params(params: Dict[str, Any], out_xml_path: str) -> None:
    """
    Compile MJCF with MjSpec → inject actuators → save XML.

    Expected keys in params:
      - dof_per_leg (int)
      - link_lengths (List[float])
      - EITHER:
          (new) motor_type: {"robstride02","robstride03"} AND gear_ratio (int >= 1)
        OR
          (legacy) motor: MotorSpec-like object with .peak_torque_Nm, .noload_rpm, .mass_kg, .J_rotor
                         AND optionally gear_ratio (fallback=1)
      - (optional) link_masses (List[float])
      - (optional) material_density (float, default 1200)
      - (optional) link_radius (float, default 0.05)
    """
    # ---- required dimensions ----
    dof = int(params["dof_per_leg"])
    link_lengths = [float(x) for x in params["link_lengths"]]
    assert dof == len(link_lengths), "len(link_lengths) must equal dof_per_leg"
    link_length_sum = sum(link_lengths)

    # ---- motor & gearbox ----
    base = _resolve_base_from_params(params)
    gear_ratio = int(params.get("gear_ratio", 1))
    out_spec = _motor_output_specs_from_out_base(base, gear_ratio)

    # ---- link mass model ----
    link_masses_param = params.get("link_masses", None)
    material_density = float(params.get("material_density", DEFAULT_DENSITY))
    link_radius = float(params.get("link_radius", DEFAULT_LINK_RADIUS))
    link_masses = _compute_link_masses(
        link_lengths=link_lengths,
        link_masses_from_params=link_masses_param,
        density=material_density,
        radius=link_radius,
    )
    for m in link_masses:
        assert m > 0.0, "Computed/Provided link mass must be > 0"

    # ---- build MjSpec tree ----
    spec = mujoco.MjSpec()
    _setup_spec_visuals(spec, link_length_sum)
    root = _build_root(spec)
    _build_leg_chain(
        root=root,
        dof=dof,
        link_lengths=link_lengths,
        link_masses=link_masses,
        link_radius=link_radius,
        joint_damping=out_spec["joint_damping"],
        joint_armature=out_spec["armature_out"],
        motor_mass=out_spec["motor_mass"],
    )

    # Optional camera
    try:
        spec.worldbody.add_camera(
            name="side_cam",
            pos=[3.0, 0.0, link_length_sum * 0.6],
            xyaxes=[0.0, -1.0, 0.0, 0.0, 0.0, 1.0],
            fovy=45.0,
        )
    except Exception:
        pass

    # ---- Compile + XML ----
    spec.compile()
    xml_text = spec.to_xml()

    # ---- Inject actuators with gearbox torque cap ----
    xml_text = _inject_actuators(xml_text, dof, torque_cap_out=out_spec["torque_cap_nm"])

    # ---- Save and validate ----
    out = Path(out_xml_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(xml_text, encoding="utf-8")

    _ = mujoco.MjModel.from_xml_path(str(out))  # validity check


# ---------------------- self-test ----------------------
if __name__ == "__main__":
    # Test A: robstride03 with external ratio 9 (≈ vendor baseline)
    paramsA = {
        "dof_per_leg": 3,
        "link_lengths": [0.25, 0.25, 0.25],
        "motor_type": "robstride03",
        "gear_ratio": 9,
    }
    build_xml_from_params(paramsA, "/tmp/single_leg_RB03_R9.xml")
    print("Wrote /tmp/single_leg_RB03_R9.xml")

    # Test B: robstride03 with external ratio 15 (more torque, less speed)
    paramsB = dict(paramsA)
    paramsB["gear_ratio"] = 15
    build_xml_from_params(paramsB, "/tmp/single_leg_RB03_R15.xml")
    print("Wrote /tmp/single_leg_RB03_R15.xml")

    # Test C: robstride02 with external ratio 20
    paramsC = {
        "dof_per_leg": 3,
        "link_lengths": [0.22, 0.22, 0.22],
        "motor_type": "robstride02",
        "gear_ratio": 20,
    }
    build_xml_from_params(paramsC, "/tmp/single_leg_RB02_R20.xml")
    print("Wrote /tmp/single_leg_RB02_R20.xml")
