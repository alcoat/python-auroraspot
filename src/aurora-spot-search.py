import rasterio
from rasterio.merge import merge
from rasterio.warp import reproject, Resampling
import glob
import numpy as np
from skimage.feature import peak_local_max
import geojson
from pyproj import Transformer

# --- paramètres ---
asc_dir = "../data/BDALTI_ASC/"         # dossier contenant les .asc IGN
viirs_file = "../data/viirs_2024.tif"   # GeoTIFF VIIRS téléchargé (WGS84)
output_geojson = "../extracts/points_hauts_viirs_refines.geojson"

threshold_alt = 60          # altitude minimale en m
min_distance = 20           # distance min entre sommets (en pixels)

# Fenêtres différentes pour plaine et ville (BD ALTI 25m → 1 px = 25 m)
north_window_plain = 120    # ~3 km au nord pour vérifier la plaine
north_window_city = 400     # ~10 km au nord pour chercher des villes
band_width_city = 200       # ±5 km de large pour la bande de recherche de villes

light_threshold = 5         # seuil max de luminosité locale (zone sombre)
city_threshold = 10         # seuil max toléré au nord (ville trop proche)
use_distance_weighting = False  # activer la pondération par distance

# --- ouverture et mosaïque BD ALTI ---
asc_files = glob.glob(f"{asc_dir}/*.asc")
print(f"Nombre de tuiles trouvées : {len(asc_files)}")

src_files = [rasterio.open(f) for f in asc_files]
mosaic, out_transform = merge(src_files)
profile = src_files[0].profile
for src in src_files:
    src.close()

dem = mosaic[0].astype(float)

# gérer nodata
nodata = profile.get("nodata", -99999)
dem[dem == nodata] = np.nan
print("MNT fusionné :", dem.shape)

# --- lecture VIIRS ---
with rasterio.open(viirs_file) as src:
    viirs_data = src.read(1).astype(float)
    viirs_transform = src.transform
    viirs_crs = src.crs

# --- reprojeter VIIRS sur la grille du MNT ---
viirs_resampled = np.empty_like(dem, dtype=float)
reproject(
    source=viirs_data,
    destination=viirs_resampled,
    src_transform=viirs_transform,
    src_crs=viirs_crs,
    dst_transform=out_transform,
    dst_crs="EPSG:2154",  # MNT IGN est en Lambert-93
    resampling=Resampling.bilinear
)

# --- détection des max locaux ---
mask = (dem >= threshold_alt)
coordinates = peak_local_max(
    dem,
    min_distance=min_distance,
    threshold_abs=threshold_alt,
    labels=mask
)
print("Sommets détectés :", len(coordinates))

# --- filtrage ---
selected_points = []
for r, c in coordinates:
    if not np.isfinite(dem[r, c]):
        continue

    # Vérif lumière locale (point doit être sombre)
    if not np.isfinite(viirs_resampled[r, c]) or viirs_resampled[r, c] > light_threshold:
        continue

    # --- 1) Vérif plaine au nord (fenêtre courte ~3 km) ---
    north_alt = dem[max(0, r-north_window_plain):r, c]
    if len(north_alt) == 0 or not np.isfinite(north_alt).any():
        continue
    mean_north = np.nanmean(north_alt)
    if mean_north >= dem[r, c] - 10:  # il faut que ça descende d’au moins 10 m
        continue

    # --- 2) Vérif lumière au nord (fenêtre longue ~20 km) ---
    r0 = max(0, r-north_window_city)
    r1 = r
    c0 = max(0, c-band_width_city)
    c1 = min(dem.shape[1], c+band_width_city)
    north_band = viirs_resampled[r0:r1, c0:c1]

    if north_band.size > 0:
        if use_distance_weighting:
            # pondération par distance (réduit l’impact des villes très lointaines)
            distances = np.arange(r1-r0)[:, None] + 1
            weighted_band = north_band / distances
            if np.nanmax(weighted_band) >= city_threshold:
                continue
        else:
            if np.nanmax(north_band) >= city_threshold:
                continue

    # si tout est bon → on garde
    selected_points.append((r, c))

print("Points retenus :", len(selected_points))

# --- conversion Lambert-93 -> WGS84 ---
transformer = Transformer.from_crs("EPSG:2154", "EPSG:4326", always_xy=True)

features = []
for r, c in selected_points:
    x, y = rasterio.transform.xy(out_transform, r, c)
    lon, lat = transformer.transform(x, y)  # Lambert-93 -> WGS84
    point = geojson.Point((lon, lat))
    features.append(geojson.Feature(
        geometry=point,
        properties={
            "elevation": float(dem[r, c]),
            "light": float(viirs_resampled[r, c]) if np.isfinite(viirs_resampled[r, c]) else None
        }
    ))

feature_collection = geojson.FeatureCollection(features)

with open(output_geojson, "w", encoding="utf-8") as f:
    geojson.dump(feature_collection, f, ensure_ascii=False, indent=2)

print(f"GeoJSON exporté (WGS84) : {output_geojson}")
