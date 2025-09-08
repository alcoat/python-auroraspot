import os
import glob
import json
import numpy as np
import rasterio
import rasterio.merge
from rasterio.warp import reproject, Resampling, transform
from rasterio.transform import xy
from skimage.feature import peak_local_max

# -----------------------
# Paramètres utilisateur
# -----------------------
asc_dir = "../data/BDALTI_ASC"            # dossier contenant les .asc
dem_files = sorted(glob.glob(os.path.join(asc_dir, "*.asc")))
if not dem_files:
    raise FileNotFoundError(f"Aucun fichier .asc trouvé dans {asc_dir}")

viirs_file = "../data/viirs_2024.tif"   # ton VIIRS (EPSG:4326)

# fenêtres en mètres
north_window_plain_m = 3000
band_width_plain_m  = 1000

north_window_city_m = 30000
band_width_city_m   = 10000

altitude_threshold = 40
contrast_threshold = 5
peak_distance_m = 500

geojson_debug  = "spots_debug.geojson"  # fichier brut avec toutes les infos


# -----------------------
# Fonctions utilitaires
# -----------------------
def load_and_merge_asc(files):
    datasets = [rasterio.open(f) for f in files]
    mosaic, transform = rasterio.merge.merge(datasets)
    if mosaic.ndim == 3:
        mosaic = mosaic[0]
    crs = rasterio.crs.CRS.from_epsg(2154)  # Lambert-93
    for ds in datasets:
        ds.close()
    return mosaic, transform, crs

def resample_viirs_to_dem(viirs_path, ref_shape, ref_transform, ref_crs):
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
    px_x = meters / abs(transform.a)
    px_y = meters / abs(transform.e)
    return int(round(px_y)), int(round(px_x))

def lambert93_to_wgs84(x, y):
    lon, lat = transform("EPSG:2154", "EPSG:4326", [x], [y])
    return lon[0], lat[0]


# -----------------------
# Pipeline
# -----------------------
print("Chargement DEM...")
dem, dem_transform, dem_crs = load_and_merge_asc(dem_files)

north_window_city_px_y, north_window_city_px_x  = meters_to_pixels(north_window_city_m, dem_transform)
band_width_city_px_y, band_width_city_px_x      = meters_to_pixels(band_width_city_m, dem_transform)
peak_distance_px_y, peak_distance_px_x          = meters_to_pixels(peak_distance_m, dem_transform)
peak_distance_px = max(peak_distance_px_y, peak_distance_px_x)

print("Rééchantillonnage VIIRS...")
viirs = resample_viirs_to_dem(viirs_file, dem.shape, dem_transform, dem_crs)

print("Détection des pics...")
coordinates = peak_local_max(
    dem,
    min_distance=peak_distance_px,
    threshold_abs=altitude_threshold,
    exclude_border=False
)
print("Pics détectés:", len(coordinates))

# -----------------------
# Extraction des infos
# -----------------------
features_all = []
for i, (r, c) in enumerate(coordinates):
    if np.isnan(dem[r, c]):
        continue

    # Contraste relatif
    window = dem[max(0, r-5):r+6, max(0, c-5):c+6]
    if window.size == 0:
        continue
    local_mean = np.nanmean(window)
    delta_h = dem[r, c] - local_mean
    if delta_h < contrast_threshold:
        continue

    # Coordonnées
    x, y = xy(dem_transform, r, c)
    lon, lat = lambert93_to_wgs84(x, y)

    # Valeurs VIIRS
    viirs_local = float(viirs[r, c]) if np.isfinite(viirs[r, c]) else None
    # Bande nord brute
    r0 = max(0, r - north_window_city_px_y)
    r1 = r
    c0 = max(0, c - band_width_city_px_x)
    c1 = min(viirs.shape[1], c + band_width_city_px_x)
    north_band = viirs[r0:r1, c0:c1]
    viirs_max_north = float(np.nanmax(north_band)) if north_band.size else None
    # Bande nord pondérée
    if north_band.size > 0:
        rows_idx, cols_idx = np.indices(north_band.shape)
        dist = np.sqrt((rows_idx - (r - r0))**2 + (cols_idx - (c - c0))**2)
        dist[dist == 0] = 1.0
        weights = 1.0 / np.sqrt(dist)
        weighted = north_band * weights
        viirs_weighted_north = float(np.nanmax(weighted))
    else:
        viirs_weighted_north = None

    features_all.append({
        "type": "Feature",
        "properties": {
            "id": i,
            "elevation": float(dem[r, c]),
            "delta_h": float(delta_h),
            "viirs_local": viirs_local,
            "viirs_max_north": viirs_max_north,
            "viirs_weighted_north": viirs_weighted_north
        },
        "geometry": {"type": "Point", "coordinates": [lon, lat]}
    })

geojson_fc = {"type": "FeatureCollection", "features": features_all}
with open(geojson_debug, "w", encoding="utf-8") as f:
    json.dump(geojson_fc, f, ensure_ascii=False, indent=2)

print(f"Fichier debug généré : {geojson_debug}")
