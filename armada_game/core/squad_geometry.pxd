# cython: language_level=3
"""
Exact closed-form geometry for the squadron rules (TODO.md, G1-G19).

Everything here is double precision, allocation-free and nogil, so it can run
inside the hot loops of move enumeration and engagement scans.

A ship's base and token are both rectangles centred on (ship.x, ship.y): the
ship template is built centred, which `_ship_coordinate()` confirms to ~5e-5 mm.
A disk of radius r overlaps a rectangle exactly when its centre is closer than
r to it - the Minkowski sum of the two is a rounded rectangle - and the
distance from a point to a rectangle is a rotation into the ship's frame
followed by an axis-aligned clamp.  No polygons, no square roots.
"""
from libc.math cimport cos, sin, fabs


cdef struct Rect:
    double cx, cy   # centre, world frame
    double c, s     # cos / sin of the ship orientation
    double hw, hl   # half width, half length


cdef inline Rect make_rect(double cx, double cy, double theta, double hw, double hl) noexcept nogil:
    cdef Rect rect
    rect.cx = cx
    rect.cy = cy
    rect.c = cos(theta)
    rect.s = sin(theta)
    rect.hw = hw
    rect.hl = hl
    return rect


cdef inline double rect_dist_sq(const Rect* rect, double px, double py) noexcept nogil:
    """Squared distance from (px, py) to the rectangle; 0 inside it."""
    cdef double dx = px - rect.cx
    cdef double dy = py - rect.cy
    # world = R(-theta) @ local + centre, so local = R(theta) @ (world - centre)
    cdef double ex = fabs(rect.c * dx - rect.s * dy) - rect.hw
    cdef double ey = fabs(rect.s * dx + rect.c * dy) - rect.hl
    if ex < 0.0: ex = 0.0
    if ey < 0.0: ey = 0.0
    return ex * ex + ey * ey


cdef inline void rect_bounds(const Rect* rect, double pad,
                             double* xmin, double* ymin, double* xmax, double* ymax) noexcept nogil:
    """Axis-aligned bounds of the rectangle grown by `pad` on every side."""
    cdef double ex = fabs(rect.c) * rect.hw + fabs(rect.s) * rect.hl + pad
    cdef double ey = fabs(rect.s) * rect.hw + fabs(rect.c) * rect.hl + pad
    xmin[0] = rect.cx - ex
    xmax[0] = rect.cx + ex
    ymin[0] = rect.cy - ey
    ymax[0] = rect.cy + ey


cdef inline bint _slab(double delta, double start, double half,
                       double* t0, double* t1) noexcept nogil:
    """Liang-Barsky: narrow [t0, t1] to where |start + t * delta| <= half."""
    cdef double ta, tb, tmp
    if delta == 0.0:
        return -half <= start <= half
    ta = (-half - start) / delta
    tb = (half - start) / delta
    if ta > tb:
        tmp = ta
        ta = tb
        tb = tmp
    if ta > t0[0]: t0[0] = ta
    if tb < t1[0]: t1[0] = tb
    return t0[0] <= t1[0]


cdef inline bint segment_hits_rect(const Rect* rect, double ax, double ay, double bx, double by) noexcept nogil:
    """True when segment a-b touches the filled rectangle."""
    cdef double dax = ax - rect.cx
    cdef double day = ay - rect.cy
    cdef double dbx = bx - rect.cx
    cdef double dby = by - rect.cy
    cdef double p0x = rect.c * dax - rect.s * day
    cdef double p0y = rect.s * dax + rect.c * day
    cdef double p1x = rect.c * dbx - rect.s * dby
    cdef double p1y = rect.s * dbx + rect.c * dby
    cdef double t0 = 0.0
    cdef double t1 = 1.0
    if not _slab(p1x - p0x, p0x, rect.hw, &t0, &t1):
        return False
    return _slab(p1y - p0y, p0y, rect.hl, &t0, &t1)


cdef inline double _orient(double px, double py, double qx, double qy, double rx, double ry) noexcept nogil:
    return (qx - px) * (ry - py) - (qy - py) * (rx - px)


cdef inline bint _on_segment(double px, double py, double qx, double qy, double rx, double ry) noexcept nogil:
    """Given p, q, r collinear: is r within the bounding box of p-q."""
    return (min(px, qx) <= rx <= max(px, qx)) and (min(py, qy) <= ry <= max(py, qy))


cdef inline bint segments_cross(double ax, double ay, double bx, double by,
                                double cx, double cy, double dx, double dy) noexcept nogil:
    """True when segments a-b and c-d share a point (touching included)."""
    cdef double o1 = _orient(ax, ay, bx, by, cx, cy)
    cdef double o2 = _orient(ax, ay, bx, by, dx, dy)
    cdef double o3 = _orient(cx, cy, dx, dy, ax, ay)
    cdef double o4 = _orient(cx, cy, dx, dy, bx, by)
    if ((o1 > 0.0 and o2 < 0.0) or (o1 < 0.0 and o2 > 0.0)) and \
       ((o3 > 0.0 and o4 < 0.0) or (o3 < 0.0 and o4 > 0.0)):
        return True
    if o1 == 0.0 and _on_segment(ax, ay, bx, by, cx, cy): return True
    if o2 == 0.0 and _on_segment(ax, ay, bx, by, dx, dy): return True
    if o3 == 0.0 and _on_segment(cx, cy, dx, dy, ax, ay): return True
    if o4 == 0.0 and _on_segment(cx, cy, dx, dy, bx, by): return True
    return False


cdef inline bint point_in_polygon(const double[:, ::1] poly, double px, double py) noexcept nogil:
    """Even-odd rule; obstacles are simple polygons, so it matches non-zero."""
    cdef Py_ssize_t i, j, n = poly.shape[0]
    cdef bint inside = False
    j = n - 1
    for i in range(n):
        if (poly[i, 1] > py) != (poly[j, 1] > py):
            if px < (poly[j, 0] - poly[i, 0]) * (py - poly[i, 1]) / (poly[j, 1] - poly[i, 1]) + poly[i, 0]:
                inside = not inside
        j = i
    return inside


cdef inline bint segment_hits_polygon(const double[:, ::1] poly,
                                      double ax, double ay, double bx, double by) noexcept nogil:
    """
    True when segment a-b touches the filled polygon: it crosses an edge, or it
    lies wholly inside (an endpoint is inside).  Same contract as
    cache.is_obstruct_obstacle, whose endpoint checks are load-bearing for the
    fully-inside case - a squadron sitting on an obstacle traces line of sight
    from a point inside it.
    """
    cdef Py_ssize_t i, j, n = poly.shape[0]
    j = n - 1
    for i in range(n):
        if segments_cross(ax, ay, bx, by, poly[j, 0], poly[j, 1], poly[i, 0], poly[i, 1]):
            return True
        j = i
    return point_in_polygon(poly, ax, ay)
