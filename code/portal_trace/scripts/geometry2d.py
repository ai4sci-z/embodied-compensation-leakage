"""Exact cross-sections of MuJoCo geoms in a plane, and clearances between them.

Used by `compute_task_tol` (9.5.4) and the cl_pm query (9.5.5). Both feed
numbers that end up as the denominator of kappa or as the bounds of the
intermediate amplitude band, so approximating a shape here would quietly move a
preregistered threshold. Every geom type is either handled exactly or refused.

A cross-section is a convex polygon dilated by a radius. That covers a box and a
mesh (polygon, radius zero), a sphere (point, radius r), a capsule (segment,
radius r) and a cylinder whose axis is perpendicular to the plane (point, radius
r), and -- the reason for choosing this representation -- it is closed under
Minkowski sums, since (A + disc a) + (B + disc b) = (A + B) + disc(a+b).

Clearance is computed as a Minkowski difference, not as a difference of two
radii. Writing `r_hole(u) - r_inserted(u)` looks equivalent and is not: the hole
side is naturally a radial function (how far along u until a wall) and the
inserted side a support function (how far the outline reaches along u), and
subtracting one from the other is only correct when both shapes are discs. On a
square peg in a square hole it understates the clearance by nearly a factor of
two in the off-axis directions. The distance until contact when the part
translates along u is a ray cast against the wall dilated by the reflected part,
which is what `first_contact_distance` does.
"""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np
from scipy.spatial import ConvexHull

__all__ = [
    "CrossSection",
    "PlaneBasis",
    "cross_section_of_geom",
    "plane_basis",
    "first_contact_distance",
    "min_contact_distance",
]

_PERP_TOL = 1e-12
# Directional queries against a dilated polygon are solved by bisection on a
# convex function. 1e-14 m is six orders below the 0.01 mm agreement the 9.5.4
# check demands, so it does not enter any reported digit.
_BISECT_TOL = 1e-14
_BISECT_MAX_ITER = 200


@dataclass(frozen=True)
class PlaneBasis:
    """An orthonormal basis of the plane normal to `axis`."""

    origin: np.ndarray  # a point on the axis, world frame
    axis: np.ndarray    # unit
    b1: np.ndarray      # unit, perpendicular to axis
    b2: np.ndarray      # unit, axis x b1

    def project(self, points: np.ndarray) -> np.ndarray:
        """World points -> 2-D coordinates in (b1, b2), relative to `origin`."""
        rel = np.atleast_2d(points) - self.origin
        return np.stack([rel @ self.b1, rel @ self.b2], axis=-1)

    def direction(self, theta: float) -> np.ndarray:
        """A unit 2-D direction at angle `theta` from b1."""
        return np.array([np.cos(theta), np.sin(theta)])

    def to_world(self, xy: np.ndarray) -> np.ndarray:
        xy = np.atleast_2d(xy)
        return self.origin + xy[:, 0:1] * self.b1 + xy[:, 1:2] * self.b2


def plane_basis(origin: np.ndarray, axis: np.ndarray) -> PlaneBasis:
    axis = np.asarray(axis, dtype=np.float64)
    norm = np.linalg.norm(axis)
    if norm < 1e-9:
        raise ValueError("insertion direction must be non-degenerate")
    axis = axis / norm
    # Pick the world axis least aligned with `axis` so the cross product is well
    # conditioned.
    seed = np.eye(3)[int(np.argmin(np.abs(axis)))]
    b1 = np.cross(axis, seed)
    b1 /= np.linalg.norm(b1)
    b2 = np.cross(axis, b1)
    return PlaneBasis(origin=np.asarray(origin, dtype=np.float64), axis=axis, b1=b1, b2=b2)


def _reduce_to_hull(pts: np.ndarray) -> np.ndarray:
    pts = np.unique(np.round(pts, 12), axis=0)
    if len(pts) <= 2:
        return pts
    try:
        return pts[ConvexHull(pts).vertices]
    except Exception:
        # Collinear point sets are legitimate: a box seen edge-on projects to a
        # segment. Keep the two extremes.
        d = pts[-1] - pts[0]
        n = np.linalg.norm(d)
        if n < _PERP_TOL:
            return pts[:1]
        t = (pts - pts[0]) @ (d / n)
        return np.stack([pts[int(np.argmin(t))], pts[int(np.argmax(t))]])


@dataclass(frozen=True)
class CrossSection:
    """A convex polygon in plane coordinates, dilated by `radius`."""

    points: np.ndarray  # (n, 2), convex hull vertices
    radius: float
    geom_name: str
    geom_type: str

    def __post_init__(self) -> None:
        pts = np.atleast_2d(np.asarray(self.points, dtype=np.float64))
        if pts.ndim != 2 or pts.shape[1] != 2:
            raise ValueError(f"{self.geom_name}: points must be (n, 2), got {pts.shape}")
        object.__setattr__(self, "points", _reduce_to_hull(pts))
        if self.radius < 0:
            raise ValueError(f"{self.geom_name}: negative radius")

    # -- basic queries ----------------------------------------------------

    def support(self, u: np.ndarray) -> float:
        """max over the cross-section of <x, u>, for a unit direction u."""
        return float(np.max(self.points @ u) + self.radius)

    def distance_from(self, point: np.ndarray) -> float:
        """Euclidean distance from `point` to the set; 0 inside."""
        return max(0.0, _polygon_distance(self.points, np.asarray(point, float)) - self.radius)

    def contains(self, point: np.ndarray) -> bool:
        return self.distance_from(point) <= 0.0

    def inner_distance_from(self, point: np.ndarray) -> float:
        """Distance from an interior `point` to the boundary. Raises if outside."""
        p = np.asarray(point, dtype=np.float64)
        if not self.contains(p):
            raise ValueError(f"{self.geom_name}: point is outside, no inner distance")
        if len(self.points) < 3:
            raise ValueError(f"{self.geom_name}: degenerate cross-section has no interior")
        return _polygon_boundary_distance_inside(self.points, p) + self.radius

    # -- combination ------------------------------------------------------

    def reflected(self) -> CrossSection:
        return CrossSection(-self.points, self.radius, f"-{self.geom_name}", self.geom_type)

    def minkowski_sum(self, other: CrossSection) -> CrossSection:
        """Exact: (A + disc a) + (B + disc b) = conv(A + B) + disc(a + b)."""
        sums = (self.points[:, None, :] + other.points[None, :, :]).reshape(-1, 2)
        return CrossSection(
            sums,
            self.radius + other.radius,
            f"{self.geom_name}+{other.geom_name}",
            "minkowski",
        )

    # -- ray casting -------------------------------------------------------

    def ray_boundary(self, u: np.ndarray) -> float | None:
        """Distance from the plane origin along u to the first boundary point.

        When the origin is outside this is the entry distance; when it is inside
        it is the exit distance. Both cases occur: an inserted part sits outside
        every wall, while Stack's centroid sits inside the supporting face.
        Returns None when the ray misses.
        """
        u = np.asarray(u, dtype=np.float64)
        if self.radius == 0.0:
            if len(self.points) < 3:
                return None  # a segment has zero measure; a ray misses it a.s.
            return _ray_polygon_boundary(self.points, u)
        if len(self.points) == 1:
            return _ray_disc_boundary(self.points[0], self.radius, u)
        return _ray_dilated_polygon_boundary(self.points, self.radius, u)


# ---------------------------------------------------------------------------
# Clearance between two parts
# ---------------------------------------------------------------------------


def first_contact_distance(
    moving: list[CrossSection], fixed: list[CrossSection], u: np.ndarray
) -> float:
    """How far `moving` may translate along u from its current pose before contact.

    Contact between a convex A translated by t*u and a convex B happens exactly
    when t*u lands in B + (-A), so this is a ray cast against every pairwise
    Minkowski sum. Unions are handled by taking the minimum over pairs, which is
    exact because contact with the union is contact with some member.

    `inf` means the direction is open -- no wall lies that way.
    """
    best = np.inf
    for b in fixed:
        for a in moving:
            hit = b.minkowski_sum(a.reflected()).ray_boundary(u)
            if hit is not None:
                best = min(best, hit)
    return float(best)


def min_contact_distance(
    moving: list[CrossSection], fixed: list[CrossSection]
) -> tuple[float, np.ndarray]:
    """The minimum of `first_contact_distance` over all directions, in closed form.

    For a convex set S not containing the origin, min over directions of the ray
    entry distance is just dist(origin, S), attained along the direction of the
    closest point. So the global minimum needs no directional sweep, which
    matters for mesh tasks where a sweep would have to be dense enough to catch a
    narrow facet.
    """
    best = np.inf
    best_dir = np.array([1.0, 0.0])
    for b in fixed:
        for a in moving:
            s = b.minkowski_sum(a.reflected())
            closest = _polygon_closest_point(s.points, np.zeros(2))
            d = max(0.0, float(np.linalg.norm(closest)) - s.radius)
            if d < best:
                best = d
                n = float(np.linalg.norm(closest))
                best_dir = closest / n if n > _PERP_TOL else best_dir
    return float(best), best_dir


# ---------------------------------------------------------------------------
# Polygon primitives
# ---------------------------------------------------------------------------


def _segment_closest_point(a: np.ndarray, b: np.ndarray, p: np.ndarray) -> np.ndarray:
    d = b - a
    dd = float(d @ d)
    if dd < _PERP_TOL:
        return a
    t = float(np.clip((p - a) @ d / dd, 0.0, 1.0))
    return a + t * d


def _polygon_inside(points: np.ndarray, p: np.ndarray) -> bool:
    if len(points) < 3:
        return False
    edges = np.stack([points, np.roll(points, -1, axis=0)], axis=1)
    # ConvexHull returns counter-clockwise vertices, so an interior point is to
    # the left of every directed edge.
    for a, b in edges:
        if np.cross(b - a, p - a) < -1e-12:
            return False
    return True


def _polygon_closest_point(points: np.ndarray, p: np.ndarray) -> np.ndarray:
    if len(points) == 1:
        return points[0]
    if _polygon_inside(points, p):
        return p
    edges = np.stack([points, np.roll(points, -1, axis=0)], axis=1)
    if len(points) == 2:
        edges = edges[:1]
    cands = np.stack([_segment_closest_point(a, b, p) for a, b in edges])
    return cands[int(np.argmin(np.linalg.norm(cands - p, axis=1)))]


def _polygon_distance(points: np.ndarray, p: np.ndarray) -> float:
    return float(np.linalg.norm(_polygon_closest_point(points, p) - p))


def _polygon_boundary_distance_inside(points: np.ndarray, p: np.ndarray) -> float:
    edges = np.stack([points, np.roll(points, -1, axis=0)], axis=1)
    return float(
        min(np.linalg.norm(_segment_closest_point(a, b, p) - p) for a, b in edges)
    )


def _first_crossing(t_lo: float, t_hi: float) -> float | None:
    if t_lo > t_hi or t_hi < 0:
        return None
    return float(t_hi) if t_lo <= 0.0 else float(t_lo)


def _ray_disc_boundary(center: np.ndarray, radius: float, u: np.ndarray) -> float | None:
    if radius <= 0:
        return None
    b = float(center @ u)
    c = float(center @ center) - radius * radius
    disc = b * b - c
    if disc < 0:
        return None
    root = float(np.sqrt(disc))
    return _first_crossing(b - root, b + root)


def _ray_polygon_boundary(points: np.ndarray, u: np.ndarray) -> float | None:
    """Slab clipping against a convex polygon."""
    edges = np.stack([points, np.roll(points, -1, axis=0)], axis=1)
    t_lo, t_hi = -np.inf, np.inf
    centroid = points.mean(axis=0)
    for p, q in edges:
        e = q - p
        n = np.array([-e[1], e[0]], dtype=np.float64)
        norm = np.linalg.norm(n)
        if norm < _PERP_TOL:
            continue
        n /= norm
        if n @ (centroid - p) > 0:  # orient outward
            n = -n
        denom = float(n @ u)
        rhs = float(n @ p)
        if abs(denom) < _PERP_TOL:
            if rhs < -1e-12:
                return None  # parallel to this edge and outside it
            continue
        t = rhs / denom
        if denom > 0:
            t_hi = min(t_hi, t)
        else:
            t_lo = max(t_lo, t)
    return _first_crossing(t_lo, t_hi)


def _ray_dilated_polygon_boundary(
    points: np.ndarray, radius: float, u: np.ndarray
) -> float | None:
    """First crossing of the offset curve of a polygon.

    g(t) = dist(t*u, polygon) is convex, so {t : g(t) <= radius} is an interval.
    Its endpoints are found by locating the minimiser and bisecting on each
    monotone branch. Bisection rather than case analysis over edge slabs and
    vertex arcs, because the case analysis has half a dozen branches whose
    degenerate cases are exactly the ones that occur here (a ray hitting a
    vertex arc tangentially), and getting one of them subtly wrong would show up
    as a plausible-looking tolerance rather than as a crash.
    """

    def g(t: float) -> float:
        return _polygon_distance(points, t * u) - radius

    # A bracket: beyond the polygon's support along u plus the radius, g grows.
    reach = float(np.max(np.linalg.norm(points, axis=1))) + radius
    hi = 2.0 * reach + 1.0

    # Locate the minimiser of the convex g on [0, hi] by ternary search.
    lo_t, hi_t = 0.0, hi
    for _ in range(_BISECT_MAX_ITER):
        if hi_t - lo_t < _BISECT_TOL:
            break
        m1 = lo_t + (hi_t - lo_t) / 3.0
        m2 = hi_t - (hi_t - lo_t) / 3.0
        if g(m1) <= g(m2):
            hi_t = m2
        else:
            lo_t = m1
    t_star = 0.5 * (lo_t + hi_t)
    if g(t_star) > 0.0:
        return None  # the ray never comes within `radius` of the polygon

    if g(0.0) <= 0.0:
        # Origin inside: the first crossing ahead is the exit.
        a, b = t_star, hi
        if g(b) < 0.0:
            return None  # unbounded, cannot happen for a bounded polygon
        for _ in range(_BISECT_MAX_ITER):
            if b - a < _BISECT_TOL:
                break
            m = 0.5 * (a + b)
            if g(m) <= 0.0:
                a = m
            else:
                b = m
        return 0.5 * (a + b)

    a, b = 0.0, t_star  # g(a) > 0 >= g(b)
    for _ in range(_BISECT_MAX_ITER):
        if b - a < _BISECT_TOL:
            break
        m = 0.5 * (a + b)
        if g(m) > 0.0:
            a = m
        else:
            b = m
    return 0.5 * (a + b)


# ---------------------------------------------------------------------------
# MuJoCo geoms
# ---------------------------------------------------------------------------


def cross_section_of_geom(
    model: mujoco.MjModel, data: mujoco.MjData, geom_id: int, basis: PlaneBasis
) -> CrossSection:
    """Exact cross-section of one geom, in the coordinates of `basis`.

    Reads the geom's world pose out of `mjData`, so `mj_forward` must have run.
    The projection discards the coordinate along the insertion axis, so what
    comes back is the outline swept over the geom's whole axial extent. For a
    prismatic part that is the cross-section itself; for a mesh that tapers it is
    the widest section, which is the conservative reading and the one a fit
    tolerance wants.
    """
    gtype = int(model.geom_type[geom_id])
    size = np.asarray(model.geom_size[geom_id], dtype=np.float64)
    pos = np.asarray(data.geom_xpos[geom_id], dtype=np.float64)
    rot = np.asarray(data.geom_xmat[geom_id], dtype=np.float64).reshape(3, 3)
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) or f"geom{geom_id}"

    T = mujoco.mjtGeom

    if gtype == T.mjGEOM_BOX:
        signs = np.array(np.meshgrid([-1, 1], [-1, 1], [-1, 1])).T.reshape(-1, 3)
        verts = pos + (signs * size[:3]) @ rot.T
        return CrossSection(basis.project(verts), 0.0, name, "box")

    if gtype == T.mjGEOM_SPHERE:
        return CrossSection(basis.project(pos), float(size[0]), name, "sphere")

    if gtype in (T.mjGEOM_CYLINDER, T.mjGEOM_CAPSULE):
        radius, half_len = float(size[0]), float(size[1])
        axis = rot[:, 2]
        alignment = abs(float(axis @ basis.axis))
        if gtype == T.mjGEOM_CAPSULE:
            ends = np.stack([pos - half_len * axis, pos + half_len * axis])
            return CrossSection(basis.project(ends), radius, name, "capsule")
        if alignment > 1.0 - 1e-9:
            # Axis perpendicular to the plane: the projection is exactly a disc.
            return CrossSection(basis.project(pos), radius, name, "cylinder")
        raise NotImplementedError(
            f"{name}: a cylinder whose axis is tilted "
            f"{np.degrees(np.arccos(min(alignment, 1.0))):.2f} deg from the insertion "
            "direction projects to a stadium with elliptical caps, which this "
            "representation cannot hold exactly. Refusing rather than approximating, "
            "because this number is the denominator of kappa"
        )

    if gtype == T.mjGEOM_MESH:
        mesh_id = int(model.geom_dataid[geom_id])
        adr = int(model.mesh_vertadr[mesh_id])
        num = int(model.mesh_vertnum[mesh_id])
        local = np.asarray(model.mesh_vert[adr : adr + num], dtype=np.float64)
        verts = pos + local @ rot.T
        return CrossSection(basis.project(verts), 0.0, name, "mesh")

    if gtype == T.mjGEOM_ELLIPSOID:
        raise NotImplementedError(
            f"{name}: an ellipsoid cross-section is an exact ellipse, not a dilated "
            "polygon; add an exact branch before using a task that contains one"
        )

    raise NotImplementedError(
        f"{name}: geom type {T(gtype).name} has no exact cross-section here"
    )
