"""
Generate a NACA 0012 airfoil STL for LBM simulation.
Chord = 1.0 m, span = 0.05 m (quasi-2D extrusion), AoA = 0 degrees.
Flow direction: +X, span direction: +Y.
"""
import numpy as np
import struct

def naca0012_y(x):
    """NACA 0012 half-thickness at position x/c."""
    t = 0.12  # max thickness ratio
    return 5.0 * t * (
        0.2969 * np.sqrt(x)
        - 0.1260 * x
        - 0.3516 * x**2
        + 0.2843 * x**3
        - 0.1015 * x**4
    )

def generate_naca0012_stl(filename, chord=1.0, span=0.2, n_chord=120, n_span=2):
    """Generate a closed NACA 0012 airfoil STL."""
    # Cosine-clustered points along chord (finer at LE and TE)
    beta = np.linspace(0, np.pi, n_chord + 1)
    x_c = 0.5 * (1.0 - np.cos(beta))  # 0 to 1

    # Upper and lower surface
    y_half = naca0012_y(x_c)
    # Close trailing edge exactly
    y_half[0] = 0.0
    y_half[-1] = 0.0

    # Scale to chord
    x_pts = x_c * chord
    y_upper = y_half * chord
    y_lower = -y_half * chord

    # Span stations
    z_stations = np.linspace(0, span, n_span + 1)

    triangles = []

    def add_tri(p1, p2, p3):
        """Add triangle with computed normal."""
        v1 = np.array(p2) - np.array(p1)
        v2 = np.array(p3) - np.array(p1)
        normal = np.cross(v1, v2)
        norm = np.linalg.norm(normal)
        if norm > 1e-12:
            normal = normal / norm
        triangles.append((normal, p1, p2, p3))

    for j in range(n_span):
        z0 = z_stations[j]
        z1 = z_stations[j + 1]

        for i in range(n_chord):
            # Upper surface: (x[i], z, y_upper[i])
            # Convention: X=streamwise, Y=spanwise, Z=vertical
            # So point = (x, y_span, z_vertical)
            p1 = (x_pts[i],   z0, y_upper[i])
            p2 = (x_pts[i+1], z0, y_upper[i+1])
            p3 = (x_pts[i+1], z1, y_upper[i+1])
            p4 = (x_pts[i],   z1, y_upper[i])
            add_tri(p1, p2, p3)
            add_tri(p1, p3, p4)

            # Lower surface (reversed winding for outward normal)
            p1 = (x_pts[i],   z0, y_lower[i])
            p2 = (x_pts[i],   z1, y_lower[i])
            p3 = (x_pts[i+1], z1, y_lower[i+1])
            p4 = (x_pts[i+1], z0, y_lower[i+1])
            add_tri(p1, p2, p3)
            add_tri(p1, p3, p4)

    # End caps (z=0 and z=span)
    for i in range(n_chord):
        # Cap at z=0 (inward normal -Y)
        pu1 = (x_pts[i],   z_stations[0], y_upper[i])
        pu2 = (x_pts[i+1], z_stations[0], y_upper[i+1])
        pl1 = (x_pts[i],   z_stations[0], y_lower[i])
        pl2 = (x_pts[i+1], z_stations[0], y_lower[i+1])
        add_tri(pu1, pl1, pl2)
        add_tri(pu1, pl2, pu2)

        # Cap at z=span (outward normal +Y)
        pu1 = (x_pts[i],   z_stations[-1], y_upper[i])
        pu2 = (x_pts[i+1], z_stations[-1], y_upper[i+1])
        pl1 = (x_pts[i],   z_stations[-1], y_lower[i])
        pl2 = (x_pts[i+1], z_stations[-1], y_lower[i+1])
        add_tri(pu1, pu2, pl2)
        add_tri(pu1, pl2, pl1)

    # Write binary STL
    with open(filename, 'wb') as f:
        f.write(b'\0' * 80)  # header
        f.write(struct.pack('<I', len(triangles)))
        for normal, p1, p2, p3 in triangles:
            f.write(struct.pack('<fff', *normal))
            f.write(struct.pack('<fff', *p1))
            f.write(struct.pack('<fff', *p2))
            f.write(struct.pack('<fff', *p3))
            f.write(struct.pack('<H', 0))  # attribute byte count

    print(f"Generated {filename}: {len(triangles)} triangles")
    print(f"  Chord: {chord} m, Span: {span} m")
    print(f"  X range: [{x_pts[0]:.4f}, {x_pts[-1]:.4f}]")
    print(f"  Z range: [{y_lower.min():.4f}, {y_upper.max():.4f}]")

if __name__ == "__main__":
    generate_naca0012_stl("model.stl", chord=1.0, span=0.05, n_chord=120, n_span=2)
