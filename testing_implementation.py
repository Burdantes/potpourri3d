#!/usr/bin/env python3
"""
Demo: symmetric domain with a hole => two symmetric geodesic-like solutions.
We compare:
  - default find_geodesic_path(start, end)
  - custom find_geodesic_path_with_route(route_top)
  - custom find_geodesic_path_with_route(route_bottom)

Run:
  python scripts/compare_geodesics_with_hole.py --show
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
import potpourri3d_bindings as pp3db

def compact_mesh(V, F):
    """
    Remove unreferenced vertices and reindex faces.
    Returns V2, F2, old_to_new (array of length V.shape[0], -1 if dropped).
    """
    used = np.unique(F.reshape(-1))
    used_sorted = np.sort(used)

    old_to_new = -np.ones(V.shape[0], dtype=np.int64)
    old_to_new[used_sorted] = np.arange(used_sorted.size, dtype=np.int64)

    V2 = V[used_sorted]
    F2 = old_to_new[F]

    return V2, F2, old_to_new
def build_square_with_circular_hole(n=60, hole_r=0.35):
    """
    Build a triangulated [-1,1]^2 grid mesh, remove triangles whose centroid is inside hole.
    Returns (V, F, vid(i,j) helper).
    """
    xs = np.linspace(-1.0, 1.0, n)
    ys = np.linspace(-1.0, 1.0, n)

    # vertices
    V = np.array([[x, y, 0.0] for y in ys for x in xs], dtype=np.float64)

    def vid(i, j):
        return i * n + j

    # triangles
    F = []
    for i in range(n - 1):
        for j in range(n - 1):
            v00 = vid(i, j)
            v10 = vid(i, j + 1)
            v01 = vid(i + 1, j)
            v11 = vid(i + 1, j + 1)
            # two triangles
            F.append([v00, v01, v10])
            F.append([v10, v01, v11])

    F = np.array(F, dtype=np.int64)

    # remove triangles with centroid inside hole radius
    tri_pts = V[F][:, :, :2]  # (m,3,2)
    centroids = tri_pts.mean(axis=1)
    r2 = np.sum(centroids**2, axis=1)
    keep = r2 >= hole_r**2
    F = F[keep]
    V, F, _ = compact_mesh(V, F)

    return V, F, vid


def nearest_vertex(V, target_xy):
    d2 = np.sum((V[:, :2] - np.array(target_xy)[None, :]) ** 2, axis=1)
    return int(np.argmin(d2))


def vertex_route_through_waypoints(V, waypoints_xy):
    """
    Build a vertex route by snapping each waypoint to nearest vertex.
    Then connect successive snapped vertices by stepping greedily on the grid-ish mesh:
    - because this is a triangulated grid, we can just interpolate along straight line in xy,
      snap intermediate points, and de-dup.
    This yields a route that is very likely edge-adjacent in the triangulation.
    If your C++ route validator throws, increase samples_per_segment.
    """
    route = []
    samples_per_segment = 200

    for a, b in zip(waypoints_xy[:-1], waypoints_xy[1:]):
        a = np.array(a); b = np.array(b)
        for t in np.linspace(0.0, 1.0, samples_per_segment):
            p = (1 - t) * a + t * b
            v = nearest_vertex(V, p)
            if not route or route[-1] != v:
                route.append(v)

    # de-dup any immediate repeats
    dedup = [route[0]]
    for v in route[1:]:
        if v != dedup[-1]:
            dedup.append(v)
    return dedup


def polyline_length(P):
    if P.shape[0] < 2:
        return 0.0
    return float(np.sum(np.linalg.norm(P[1:] - P[:-1], axis=1)))


def plot_mesh_edges(ax, V, F, max_edges=200000):
    # draw a subset of edges for speed
    edges = set()
    for tri in F:
        a, b, c = tri
        for u, v in [(a, b), (b, c), (c, a)]:
            if u > v:
                u, v = v, u
            edges.add((u, v))
        if len(edges) > max_edges:
            break

    for (u, v) in edges:
        ax.plot([V[u, 0], V[v, 0]], [V[u, 1], V[v, 1]], linewidth=0.3)

    ax.set_aspect("equal", adjustable="box")
def stretch_bottom(V, y0=0.0, factor=1.8):
    """
    Stretch only the part with y < y0 by multiplying its y-coordinate.
    Keeps the top unchanged -> breaks symmetry.
    """
    V2 = V.copy()
    mask = V2[:, 1] < y0
    V2[mask, 1] = y0 + factor * (V2[mask, 1] - y0)
    return V2

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=70, help="grid resolution")
    ap.add_argument("--hole-r", type=float, default=0.35, help="hole radius")
    ap.add_argument("--out", type=str, default="geodesic_hole_compare_stretched.png", help="output PNG")
    ap.add_argument("--show", action="store_true", help="show interactive plot")
    args = ap.parse_args()

    V, F, vid = build_square_with_circular_hole(n=args.n, hole_r=args.hole_r)
    V = stretch_bottom(V, y0=0.0, factor=2.5)
    mgr = pp3db.EdgeFlipGeodesicsManager(V, F)

    if not hasattr(mgr, "find_geodesic_path_with_route"):
        raise RuntimeError(
            "Missing EdgeFlipGeodesicsManager.find_geodesic_path_with_route(). "
            "Make sure you're importing your rebuilt potpourri3d_bindings."
        )

    # Choose symmetric start/end around the hole
    start_xy = (-0.9, 0.0)
    end_xy   = ( 0.9, 0.0)
    v_start = nearest_vertex(V, start_xy)
    v_end   = nearest_vertex(V, end_xy)

    # Two symmetric waypoint routes: go above hole vs below hole
    # The middle waypoint pushes the init to the top/bottom corridor.
    top_waypoints = [start_xy, (0.0,  0.75), end_xy]
    bot_waypoints = [start_xy, (0.0, -0.75), end_xy]

    route_top = vertex_route_through_waypoints(V, top_waypoints)
    route_bot = vertex_route_through_waypoints(V, bot_waypoints)

    # Compute paths
    path_default = mgr.find_geodesic_path(v_start, v_end, 2**63 - 1, 0.0)
    path_top     = mgr.find_geodesic_path_with_route(route_top, 2**63 - 1, 0.0)
    path_bot     = mgr.find_geodesic_path_with_route(route_bot, 2**63 - 1, 0.0)

    print("start:", v_start, "end:", v_end)
    print("default length:", polyline_length(path_default), "points:", path_default.shape[0])
    print("top init length:", polyline_length(path_top), "points:", path_top.shape[0], "route verts:", len(route_top))
    print("bot init length:", polyline_length(path_bot), "points:", path_bot.shape[0], "route verts:", len(route_bot))

    # Plot
    fig, ax = plt.subplots(figsize=(8, 8))
    plot_mesh_edges(ax, V, F)

    # hole circle for clarity
    theta = np.linspace(0, 2*np.pi, 400)
    ax.plot(args.hole_r*np.cos(theta), args.hole_r*np.sin(theta), linewidth=2)

    ax.plot(path_default[:, 0], path_default[:, 1], linewidth=3, label="default (Dijkstra init)")
    ax.plot(path_top[:, 0], path_top[:, 1], linewidth=3, label="custom init: top route")
    ax.plot(path_bot[:, 0], path_bot[:, 1], linewidth=3, label="custom init: bottom route")

    # show the vertex routes (dotted)
    rt = V[np.array(route_top), :2]
    rb = V[np.array(route_bot), :2]
    ax.plot(rt[:, 0], rt[:, 1], linestyle=":", linewidth=2, label="provided top vertex route")
    ax.plot(rb[:, 0], rb[:, 1], linestyle=":", linewidth=2, label="provided bottom vertex route")

    ax.scatter([V[v_start, 0]], [V[v_start, 1]], s=90, marker="o", label="start")
    ax.scatter([V[v_end, 0]], [V[v_end, 1]], s=90, marker="x", label="end")

    ax.set_title("Two symmetric basins around a hole: init route selects which one you get")
    ax.legend()
    fig.tight_layout()
    fig.savefig(args.out, dpi=220)
    print("Saved:", args.out)

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()