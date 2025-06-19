"""Utility functions used across modules."""

import numpy as np


def segment_intersects_rect(p1, p2, rect_min, rect_max):
    """Return True if the line segment from p1 to p2 intersects rectangle."""
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    p = [-dx, dx, -dy, dy]
    q = [p1[0] - rect_min[0], rect_max[0] - p1[0], p1[1] - rect_min[1], rect_max[1] - p1[1]]
    u0, u1 = 0.0, 1.0
    for pi, qi in zip(p, q):
        if abs(pi) < 1e-6:
            if qi < 0:
                return False
        else:
            t = qi / pi
            if pi < 0:
                if t > u1:
                    return False
                if t > u0:
                    u0 = t
            else:
                if t < u0:
                    return False
                if t < u1:
                    u1 = t
    return True
