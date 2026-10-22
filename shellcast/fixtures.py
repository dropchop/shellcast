"""Analytic test fixtures: a sphere (known shell volume) and a head proxy with nose, ears, chin and neck."""

from __future__ import annotations

import numpy as np

import manifold3d as m3

from .geom import Manifold, tm_from_manifold


def sphere(radius=60.0, segments=96):
    return Manifold.sphere(radius, segments)


def head_proxy(scale=1.0, segments=64):
    """Roughly a 160 × 190 × 230 mm head: ellipsoid skull, nose, chin, ears, and a neck cylinder ending flat at the bottom."""
    s = scale
    skull = Manifold.sphere(1.0, segments).scale((80 * s, 95 * s, 110 * s))
    nose = Manifold.sphere(16 * s, 32).translate((0, -88 * s, 0))
    nose_bridge = Manifold.sphere(12 * s, 32).translate((0, -80 * s, 22 * s))
    chin = Manifold.sphere(30 * s, 32).scale((1.1, 0.9, 0.8)).translate((0, -55 * s, -75 * s))
    ear_l = Manifold.sphere(24 * s, 32).scale((0.45, 1, 1.1)).translate((-80 * s, 10 * s, 5 * s))
    ear_r = ear_l.mirror((1, 0, 0))
    brow = Manifold.cube((90 * s, 20 * s, 14 * s), center=True).translate((0, -82 * s, 38 * s))
    neck = Manifold.cylinder(90 * s, 46 * s, 42 * s, 48).translate((0, 8 * s, -170 * s))
    head = Manifold.batch_boolean([skull, nose, nose_bridge, chin, ear_l, ear_r, brow, neck], m3.OpType.Add)
    # eye sockets and nostrils: concavities that must NOT trap the core (opening removes them) but should get vents
    eye_l = Manifold.sphere(14 * s, 32).translate((-30 * s, -86 * s, 28 * s))
    nostril = Manifold.sphere(5 * s, 16).translate((-6 * s, -96 * s, -6 * s))
    head = head - eye_l - eye_l.mirror((1, 0, 0)) - nostril - nostril.mirror((1, 0, 0))
    return head.smooth_out(60)


def bust(scale=1.0):
    """The head proxy standing on wide shoulders and a base, like a scanned or sculpted bust."""
    s = scale
    head = head_proxy(s)
    shoulders = Manifold.cube((300 * s, 180 * s, 45 * s), center=True).translate((0, 8 * s, -180 * s))
    return (head + shoulders).smooth_out(60)


def write_fixture(name: str, path: str):
    m = {"sphere": sphere, "head": head_proxy, "bust": bust}[name]()
    t = tm_from_manifold(m)
    t.export(path)
    return t
