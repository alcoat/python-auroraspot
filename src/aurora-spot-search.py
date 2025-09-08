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

# -----------------------
# Paramètres utilisateur (tes valeurs)
# -----------------------
asc_dir = "../data/BDALTI_ASC"
dem_files = sorted(glob.glob(os.path.join(asc_dir, "*.asc")))
if not dem_files:
    raise FileNotFoundError(f"Aucun fichier .asc trouvé dans {asc_dir}")

viirs_file = "../data/viirs_2024.tif"   # VIIRS (EPSG:4326)

# Fenêtres en mètres
north_window_plain_m = 3000    # 3 km au nord
band_width_plain_m  = 1000     # ±1 km
north_window_city_m = 30000    # 30 km au nord
band_width_city_m   = 10000    # ±10 km

# Seuils
light_threshold = 1            # radiance VIIRS locale max
city_threshold = 20            # seuil ville (max brut dans la bande nord)
altitude_threshold = 40        # altitude mini (m) — utilisé dans peak_local_max
contrast_threshold = 15        # delta_h minimal (m)
contrast_window_m = 1000       # fenêtre carrée de 1 km
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
north_window_plain_px_y, north_window_plain_px_x = meters_to_pixels(north_window_plain_m, dem_transform)
north_window_city_px_y, north_window_city_px_x  = meters_to_pixels(north_window_city_m, dem_transform)
band_width_plain_px_y, band_width_plain_px_x    = meters_to_pixels(band_width_plain_m, dem_transform)
band_width_city_px_y, band_width_city_px_x      = meters_to_pixels(band_width_city_m, dem_transform)
peak_distance_px_y, peak_distance_px_x          = meters_to_pixels(peak_distance_m, dem_transform)
peak_distance_px = max(peak_distance_px_y, peak_distance_px_x)
contrast_win_px_y, contrast_win_px_x = meters_to_pixels(contrast_window_m, dem_transform)
half_win_y = contrast_win_px_y // 2
half_win_x = contrast_win_px_x // 2

print("Fenêtres (pixels): plain=", north_window_plain_px_y,
      " city=", north_window_city_px_y,
      " band_plain=", band_width_plain_px_x,
      " band_city=", band_width_city_px_x,
      " peak_dist=", peak_distance_px)

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
    win = dem[max(0, r-half_win_y):r+half_win_y+1, max(0, c-half_win_x):c+half_win_x+1]
    if win.size == 0:
        continue
    local_mean = np.nanmean(win)
    delta_h = dem[r, c] - local_mean
    if delta_h < contrast_threshold:
        # rejet : pas assez de relief relatif
        continue

    # Lumière locale (pixel du site)
    viirs_local = float(viirs[r, c]) if np.isfinite(viirs[r, c]) else None
    if viirs_local is None or viirs_local > light_threshold:
        # rejet : trop lumineux localement ou pas de donnée VIIRS
        continue

    # Vérif plaine au nord (fenêtre courte)
    if not check_plain_north(dem, r, c, north_window_plain_px_y, band_width_plain_px_x):
        # rejet : pas de plaine au nord immédiat
        continue

    # Max brut dans la bande nord (fenêtre longue)
    north_band = extract_north_band(viirs, r, c, north_window_city_px_y, band_width_city_px_x)
    viirs_max_north = float(np.nanmax(north_band)) if north_band.size else None
    if viirs_max_north is not None and viirs_max_north > city_threshold:
        # rejet : grosse source lumineuse dans la bande nord
        continue

    # si tout passe, on stocke les métadonnées (pas de recalcul plus tard)
    valid_points.append((r, c, float(viirs_local), viirs_max_north, float(delta_h)))

print("Points retenus après filtrage :", len(valid_points))

# -----------------------
# Écriture GeoJSON (on convertit les coordonnées point par point)
# -----------------------
os.makedirs(os.path.dirname(geojson_output), exist_ok=True)
features = []
for i, (r, c, viirs_local, viirs_max_north, delta_h) in enumerate(valid_points):
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
            "viirs_max_north": viirs_max_north
        },
        "geometry": {"type": "Point", "coordinates": [lon, lat]}
    }
    features.append(feat)

geojson_fc = {"type": "FeatureCollection", "features": features}
with open(geojson_output, "w", encoding="utf-8") as f:
    json.dump(geojson_fc, f, ensure_ascii=False, indent=2)

print(f"GeoJSON généré : {geojson_output}")
