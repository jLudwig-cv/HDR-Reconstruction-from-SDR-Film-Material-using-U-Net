import matplotlib.pyplot as plt
import numpy as np
from colour.plotting import plot_chromaticity_diagram_CIE1976UCS, override_style



# Umwandlung von einem XYZ Array in ein uv Array
def xyz_to_uv(xyz):
    X, Y, Z = xyz[..., 0], xyz[..., 1], xyz[..., 2]
    denom = X + 15*Y + 3*Z
    u_prime = np.divide(4*X, denom, out=np.zeros_like(X), where=denom != 0)
    v_prime = np.divide(9*Y, denom, out=np.zeros_like(Y), where=denom != 0)
    return np.stack((u_prime, v_prime), axis=-1)


# Umwandlung eines XY Arrys in ein uv Array
def xy_to_uv(xy):
    x, y = xy[:, 0], xy[:, 1]
    denom = -2 * x + 12 * y + 3
    u_prime = 4 * x / denom
    v_prime = 9 * y / denom
    return np.stack((u_prime, v_prime), axis=-1)


# Abfrage, ob ein Punkt in einem definierten Dreieck ist
def is_inside_polygon(points, polygon):
    from matplotlib.path import Path
    path = Path(polygon)
    return path.contains_points(points, radius=0)


# Funktion, die die Farben eines Bildes in ein CIE1976 Diagramm plottet, mit wichtigen Farbräumen
def plot_cie1976_diagram_with_gamuts(rgb_image, title, sample_rate=0.1):
    RGB_to_XYZ_matrix = np.array([
        [0.4124564, 0.3575761, 0.1804375],
        [0.2126729, 0.7151522, 0.0721750],
        [0.0193339, 0.1191920, 0.9503041]
    ])

    shape = rgb_image.shape
    xyz_image = np.dot(rgb_image.reshape(-1, 3), RGB_to_XYZ_matrix.T).reshape(shape)

    # Umwandlung von XYZ in uv Farbinformationen
    uv_image = xyz_to_uv(xyz_image)

    # Herausziehen der u und v Komponenten
    u_prime = uv_image[..., 0].flatten()
    v_prime = uv_image[..., 1].flatten()

    # Filter out zero points which might be present due to black pixels
    # nonzero_indices = (u_prime > 0) & (v_prime > 0)
    # u_prime = u_prime[nonzero_indices]
    # v_prime = v_prime[nonzero_indices]

    # Subsamplen der Farbpunkte des Bildes
    sample_indices = np.random.choice(len(u_prime), size=int(len(u_prime) * sample_rate), replace=False)
    u_prime = u_prime[sample_indices]
    v_prime = v_prime[sample_indices]

    # Primärvalenzen in xy der drei Farbräume
    rec709_xy_primaries = np.array([[0.64, 0.33], [0.3, 0.6], [0.15, 0.06]])
    dci_p3_xy_primaries = np.array([[0.68, 0.32], [0.265, 0.69], [0.15, 0.06]])
    rec2020_xy_primaries = np.array([[0.708, 0.292], [0.17, 0.797], [0.131, 0.046]])

    # Umrechnen der Primärvalenzen in uv
    rec709_uv_primaries = xy_to_uv(rec709_xy_primaries)
    dci_p3_uv_primaries = xy_to_uv(dci_p3_xy_primaries)
    rec2020_uv_primaries = xy_to_uv(rec2020_xy_primaries)

    # Dreiecke aus den Primärvalenzen erzeugen
    points = np.column_stack((u_prime, v_prime))
    rec709_polygon = rec709_uv_primaries
    dci_p3_polygon = dci_p3_uv_primaries
    rec2020_polygon = rec2020_uv_primaries

    # Bestimmen, wie viele Punkte des Bildes in den Dreiecken liegen
    rec709_mask = is_inside_polygon(points, rec709_polygon)
    dci_p3_mask = is_inside_polygon(points, dci_p3_polygon)
    rec2020_mask = is_inside_polygon(points, rec2020_polygon)

    # Berechnen der Prozentzahlen der Farbraumabdeckung
    rec709_percentage = np.sum(rec709_mask) / len(points) * 100
    dci_p3_percentage = np.sum(dci_p3_mask) / len(points) * 100
    rec2020_percentage = np.sum(rec2020_mask) / len(points) * 100

    # Anzeigen der Farbraumabdeckung
    print(title)
    print(f"Rec. 709: {rec709_percentage:.2f}%")
    print(f"DCI-P3: {dci_p3_percentage - rec709_percentage:.2f}%")
    print(f"Rec. 2020: {rec2020_percentage - dci_p3_percentage:.2f}%")
    print("")

    # Offnen eines neuen Plots
    override_style()
    fig, ax = plot_chromaticity_diagram_CIE1976UCS(standalone=False)

    # Einfügen von verschiedenen Farbräumen als Dreiecke
    ax.plot(*rec709_uv_primaries[[0, 1, 2, 0]].T, color='tab:orange', linestyle='--', label='Rec. 709', linewidth=2)
    ax.plot(*dci_p3_uv_primaries[[0, 1, 2, 0]].T, color='tab:blue', linestyle='-.', label='DCI-P3', linewidth=2)
    ax.plot(*rec2020_uv_primaries[[0, 1, 2, 0]].T, color='tab:green', linestyle=':', label='Rec. 2020', linewidth=2)

    # Plotten der (subgesampleten) Bildpunkte
    ax.plot(u_prime, v_prime, 'o', color='black', markersize=1, alpha=0.3)

    # Definieren des Plots
    ax.set_xlim(-0.1, 0.7)
    ax.set_ylim(-0.1, 0.7)
    ax.set_xlabel("u'")
    ax.set_ylabel("v'")
    ax.set_title(title)
    ax.grid(True)
    ax.legend()
    
    # Anzeigen des Plots
    plt.draw()
    plt.pause(0.001) 


# Plotten eines Histogrammes der Helligkeiten eines Bildes
def plot_histogram_nits(luminance_array, title):
    # Erstellen von logarithmischen Bins
    bins = np.logspace(0, 3, num=256)  # von 1 bis 10^4 (10000)

    plt.figure(figsize=(10, 5))
    plt.hist(luminance_array.flatten(), bins=bins, color='orange', alpha=0.7)
    plt.xscale('log')
    plt.xlim(1, 1000)  # x-Achse von 1 bis 10000 Nits
    plt.title(title)
    plt.xlabel('Helligkeit (Nits)')
    plt.ylabel('Häufigkeit')
    plt.grid(True)
    plt.draw()
    plt.pause(0.001) 