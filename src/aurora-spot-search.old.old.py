import os
import glob
import json
import numpy as np
import rasterio
import rasterio.merge
from rasterio.warp import reproject, Resampling
from rasterio.transform import xy
from skimage.feature import peak_local_max
from pyproj import Transformer

import matplotlib.pyplot as plt
import matplotlib.patches as patches

# -----------------------
# Paramètres utilisateur (tes valeurs)
# -----------------------
asc_dir = "../data/BDALTI_ASC"
dem_files = sorted(glob.glob(os.path.join(asc_dir, "*.asc")))
if not dem_files:
    raise FileNotFoundError(f"Aucun fichier .asc trouvé dans {asc_dir}")

viirs_file = "../data/viirs_2024.tif"   # VIIRS (EPSG:4326)

# Observation
observation_angle_deg = 180      # orientation (0=nord, 90=est, 180=sud, etc.)
observation_width_deg = 30     # ouverture du cône (°)

# Distances max
plain_distance_m = 3000        # vérif relief (plaine) dans un rayon de 3 km
city_distance_m  = 30000       # vérif villes lumineuses dans un rayon de 30 km

# Seuils
light_threshold = 1            # radiance VIIRS locale max
city_threshold = 20            # seuil ville (max brut dans la bande nord)
altitude_threshold = 40        # altitude mini (m) — utilisé dans peak_local_max
contrast_threshold = 15        # delta_h minimal (m)
contrast_radius_m = 500       # fenêtre carrée de 500 m
peak_distance_m = 500          # distance minimale entre pics


geojson_output = "../extracts/spots.geojson"

# -----------------------
# Fonctions utilitaires
# -----------------------
def load_and_merge_asc(files):
    """Charge et fusionne les fichiers ASC BD ALTI (Lambert-93)."""
    datasets = [rasterio.open(f) for f in files]
    mosaic, dem_transform = rasterio.merge.merge(datasets)
    if mosaic.ndim == 3:
        mosaic = mosaic[0]
    dem_crs = rasterio.crs.CRS.from_epsg(2154)  # Lambert-93
    for ds in datasets:
        ds.close()
    return mosaic, dem_transform, dem_crs


def resample_viirs_to_dem(viirs_path, ref_shape, ref_transform, ref_crs):
    """Rééchantillonne VIIRS (EPSG:4326) sur la grille DEM (Lambert-93)."""
    with rasterio.open(viirs_path) as src:
        dst = np.empty(ref_shape, dtype=np.float32)
        reproject(
            source=rasterio.band(src, 1),
            destination=dst,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=ref_transform,
            dst_crs=ref_crs,
            resampling=Resampling.bilinear
        )
    return dst


def meters_to_pixels(meters, transform):
    """Convertit une distance en mètres en (px_y, px_x) selon le transform du DEM."""
    px_x = meters / abs(transform.a)  # transform.a = pixel width (easting)
    px_y = meters / abs(transform.e)  # transform.e = negative pixel height (northing)
    return int(round(px_y)), int(round(px_x))


def extract_cone_precise(arr, r, c, distance_px, angle_deg, width_deg):
    """Extrait les valeurs dans un cône (rayon=distance_px, orientation=angle_deg, ouverture=width_deg)."""
    rows, cols = arr.shape
    r0 = max(0, r - distance_px)
    r1 = min(rows, r + distance_px)
    c0 = max(0, c - distance_px)
    c1 = min(cols, c + distance_px)

    sub = arr[r0:r1, c0:c1]
    if sub.size == 0:
        return np.array([])

    # Coordonnées relatives (par rapport au point central)
    rr, cc = np.indices(sub.shape)
    rr = rr + r0 - r
    cc = cc + c0 - c

    dist = np.sqrt(rr**2 + cc**2)
    angle = (np.degrees(np.arctan2(cc, -rr)) + 360) % 360  # 0=nord, 90=est

    mask_dist = dist <= distance_px
    dtheta = ((angle - angle_deg + 180) % 360) - 180
    mask_angle = np.abs(dtheta) <= width_deg / 2

    return sub[mask_dist & mask_angle]

def extract_cone_old(arr, r, c, distance_px, angle_deg, width_deg, n_rect=3):
    """
    Approximation rapide d'un cône en le décomposant en n_rect rectangles.
    
    arr        : matrice (DEM ou VIIRS)
    r, c       : position centrale (pixel)
    distance_px: rayon max en pixels
    angle_deg  : orientation (0=nord, 90=est, etc.)
    width_deg  : ouverture totale du cône en degrés
    n_rect     : nombre de rectangles utilisés pour approximer le cône
    """
    rows, cols = arr.shape
    results = []

    # Convertir l'angle en radians (0 = nord = -y)
    theta = np.deg2rad(angle_deg)
    # Vecteur direction
    dx = np.sin(theta)
    dy = -np.cos(theta)

    # Demi-largeur en radians
    half_width = np.deg2rad(width_deg / 2)

    for i in range(1, n_rect + 1):
        # Distance à laquelle placer ce rectangle (fraction du rayon total)
        frac = i / n_rect
        d = frac * distance_px

        # Largeur de ce rectangle à cette distance
        half_span = d * np.tan(half_width)

        # Rectangle englobant en coordonnées locales
        min_r = int(round(r - d*dy - half_span*np.cos(theta)))
        max_r = int(round(r - d*dy + half_span*np.cos(theta)))
        min_c = int(round(c + d*dx - half_span*np.sin(theta)))
        max_c = int(round(c + d*dx + half_span*np.sin(theta)))

        # Clipper aux bords
        min_r = max(0, min_r)
        max_r = min(rows, max_r)
        min_c = max(0, min_c)
        max_c = min(cols, max_c)

        if max_r > min_r and max_c > min_c:
            results.append(arr[min_r:max_r, min_c:max_c])

    if not results:
        return np.array([])
    return np.concatenate([r.flatten() for r in results])
    
def extract_cone(arr, r, c, distance_px, angle_deg, width_deg, n_rect=3):
    """
    Approximation rapide d'un cône par n_rect bandes rectangulaires.
    """
    rows, cols = arr.shape
    results = []

    # Convertir angle en radians
    theta = np.deg2rad(angle_deg)
    dx = np.sin(theta)
    dy = -np.cos(theta)

    # Demi-largeur en radians
    half_width = np.deg2rad(width_deg / 2)

    # Découpage du rayon en segments
    for i in range(n_rect):
        d0 = (i / n_rect) * distance_px
        d1 = ((i + 1) / n_rect) * distance_px
        d_mid = 0.5 * (d0 + d1)

        # largeur de la bande à cette distance
        half_span = d_mid * np.tan(half_width)

        # centre du rectangle à cette distance
        r_mid = int(round(r - d_mid * dy))
        c_mid = int(round(c + d_mid * dx))

        # rectangle aligné avec l'image (pas tourné)
        min_r = max(0, int(round(r_mid - (d1 - d0)/2)))
        max_r = min(rows, int(round(r_mid + (d1 - d0)/2)))
        min_c = max(0, int(round(c_mid - half_span)))
        max_c = min(cols, int(round(c_mid + half_span)))

        if max_r > min_r and max_c > min_c:
            results.append(arr[min_r:max_r, min_c:max_c])

    if not results:
        return np.array([])
    return np.concatenate([r.flatten() for r in results])


def extract_north_band(arr, r, c, north_px, band_px):
    """Extrait la bande rectangulaire au nord d'un point (r, c)."""
    r0 = max(0, r - north_px)
    r1 = r
    c0 = max(0, c - band_px)
    c1 = min(arr.shape[1], c + band_px)
    return arr[r0:r1, c0:c1]


def check_plain_north(dem, r, c, north_px, band_px):
    """Renvoie True si la bande nord est une 'plaine' (max ≤ hauteur du point)."""
    north_band = extract_north_band(dem, r, c, north_px, band_px)
    return north_band.size > 0 and np.nanmax(north_band) <= dem[r, c]


def debug_plot_overlay(dem, viirs, r, c, distance_px, angle_deg, width_deg, n_rect=3, filename="debug_overlay.png"):
    """
    Génère une image montrant DEM+VIIRS en overlay, le point central et les rectangles approximant le cône.
    """
    fig, ax = plt.subplots(figsize=(8, 8))

    # Échelle DEM : 0 à max altitude
    ax.imshow(dem, cmap="terrain", origin="lower",
              vmin=np.nanmin(dem), vmax=np.nanmax(dem), alpha=0.5)

    # Échelle VIIRS : 0 à 30 nW/cm²/sr (ou ajuste selon ta carte)
    ax.imshow(viirs, cmap="inferno", origin="lower",
              vmin=0, vmax=30, alpha=0.5)

    ax.plot(c, r, "ro", label="Point central")

    theta = np.deg2rad(angle_deg)
    dx = np.sin(theta)
    dy = -np.cos(theta)
    half_width = np.deg2rad(width_deg / 2)

    for i in range(n_rect):
        d0 = (i / n_rect) * distance_px
        d1 = ((i + 1) / n_rect) * distance_px
        d_mid = 0.5 * (d0 + d1)
        half_span = d_mid * np.tan(half_width)

        r_mid = r - d_mid * dy
        c_mid = c + d_mid * dx

        min_r = r_mid - (d1 - d0)/2
        max_r = r_mid + (d1 - d0)/2
        min_c = c_mid - half_span
        max_c = c_mid + half_span

        rect = patches.Rectangle(
            (min_c, min_r),
            max_c - min_c,
            max_r - min_r,
            linewidth=1,
            edgecolor="cyan",
            facecolor="none"
        )
        ax.add_patch(rect)

    ax.legend()
    plt.savefig(filename, dpi=150)
    plt.close(fig)
    print(f"Image overlay sauvegardée : {filename}")

# Transformer Lambert-93 -> WGS84 (créé une fois)
transformer = Transformer.from_crs("EPSG:2154", "EPSG:4326", always_xy=True)

# -----------------------
# Pipeline
# -----------------------
print("Chargement DEM (fusion des .asc en Lambert-93)...")
dem, dem_transform, dem_crs = load_and_merge_asc(dem_files)
rows, cols = dem.shape
print(f"DEM chargé : shape = {dem.shape}")

# Conversion distances → pixels (utiliser composante Y pour distances nord-sud)
plain_distance_px_y, _ = meters_to_pixels(plain_distance_m, dem_transform)
city_distance_px_y, _  = meters_to_pixels(city_distance_m, dem_transform)
peak_distance_px_y, peak_distance_px_x = meters_to_pixels(peak_distance_m, dem_transform)
peak_distance_px = max(peak_distance_px_y, peak_distance_px_x)
contrast_radius_px_y, contrast_radius_px_x = meters_to_pixels(contrast_radius_m, dem_transform)

print("Rééchantillonnage VIIRS sur la grille DEM (Lambert-93)...")
viirs = resample_viirs_to_dem(viirs_file, dem.shape, dem_transform, dem_crs)
print("VIIRS resampled shape:", viirs.shape)

print("Détection des pics locaux (peak_local_max)...")
coordinates = peak_local_max(
    dem,
    min_distance=peak_distance_px,
    threshold_abs=altitude_threshold,   # altitude absolue minimale
    exclude_border=False
)
print("Nombre de pics détectés (brut):", len(coordinates))

print("Filtrage des pics selon contraste/local light/plaine/ville...")
valid_points = []
for (r, c) in coordinates:
    # vérifs basiques
    if r < 0 or r >= rows or c < 0 or c >= cols:
        continue
    if np.isnan(dem[r, c]):
        continue

    # Contraste relatif (∆h) autour du pic
    win = dem[max(0, r-contrast_radius_px_y):r+contrast_radius_px_y+1,
              max(0, c-contrast_radius_px_x):c+contrast_radius_px_x+1]
    if win.size == 0:
        continue
    local_median = np.nanmedian(win)
    delta_h = dem[r, c] - local_median
    if delta_h < contrast_threshold:
        # rejet : pas assez de relief relatif
        continue

    # Lumière locale (pixel du site)
    viirs_local = float(viirs[r, c]) if np.isfinite(viirs[r, c]) else None
    if viirs_local is None or viirs_local > light_threshold:
        # rejet : trop lumineux localement ou pas de donnée VIIRS
        continue

    ### 
    ### # Vérif plaine au nord (fenêtre courte)
    ### if not check_plain_north(dem, r, c, north_window_plain_px_y, band_width_plain_px_x):
    ###     # rejet : pas de plaine au nord immédiat
    ###     continue
    ### 
    ### # Max brut dans la bande nord (fenêtre longue)
    ### north_band = extract_north_band(viirs, r, c, north_window_city_px_y, band_width_city_px_x)
    ### viirs_max_north = float(np.nanmax(north_band)) if north_band.size else None
    ### if viirs_max_north is not None and viirs_max_north > city_threshold:
    ###     # rejet : grosse source lumineuse dans la bande nord
    ###     continue

    # Relief dans le cône
    plain_band = extract_cone(dem, r, c,
                              distance_px=plain_distance_px_y,
                              angle_deg=observation_angle_deg,
                              width_deg=observation_width_deg)
                              
    if plain_band.size == 0 or np.nanmax(plain_band) > dem[r, c]:
        continue

    # Lumière dans le cône
    city_band = extract_cone(viirs, r, c,
                             distance_px=city_distance_px_y,
                             angle_deg=observation_angle_deg,
                             width_deg=observation_width_deg)
                             
    viirs_max_city = float(np.nanmax(city_band)) if city_band.size else None
    if viirs_max_city is not None and viirs_max_city > city_threshold:
        continue

    # si tout passe, on stocke les métadonnées (pas de recalcul plus tard)
    valid_points.append((r, c, float(viirs_local), viirs_max_city, float(delta_h)))

print("Points retenus après filtrage :", len(valid_points))


# -----------------------
# Tri des points
# -----------------------
sorted_points = sorted(
    valid_points,
    key=lambda p: (
        p[2],                                     # viirs_local (ascendant)
        -p[4],                                    # delta_h (descendant)
        p[3] if p[3] is not None else float("inf"),  # viirs_max_city (ascendant)
        -dem[p[0], p[1]]                          # elevation (descendant)
    )
)

#top3 = sorted_points[:3]
#for i, (r, c, viirs_local, viirs_max_city, delta_h) in enumerate(top3, 1):
#    debug_plot_overlay(
#        dem, viirs, r, c,
#        distance_px=city_distance_px_y,
#        angle_deg=observation_angle_deg,
#        width_deg=observation_width_deg,
#        n_rect=3,
#        filename=f"top{i}_debug.png"
#    )

# -----------------------
# Écriture GeoJSON (on convertit les coordonnées point par point)
# -----------------------
os.makedirs(os.path.dirname(geojson_output), exist_ok=True)
features = []
for i, (r, c, viirs_local, viirs_max_city, delta_h) in enumerate(sorted_points):
    # coordonnées projetées (Lambert-93)
    x_proj, y_proj = xy(dem_transform, r, c)
    # conversion en WGS84
    lon, lat = transformer.transform(x_proj, y_proj)
    feat = {
        "type": "Feature",
        "properties": {
            "id": i,
            "elevation": float(dem[r, c]),
            "delta_h": delta_h,
            "viirs_local": viirs_local,
            "viirs_max_city": viirs_max_city
        },
        "geometry": {"type": "Point", "coordinates": [lon, lat]}
    }
    features.append(feat)

geojson_fc = {"type": "FeatureCollection", "features": features}
with open(geojson_output, "w", encoding="utf-8") as f:
    json.dump(geojson_fc, f, ensure_ascii=False, indent=2)

print(f"GeoJSON généré : {geojson_output}")
