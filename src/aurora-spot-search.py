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
import simplekml

import matplotlib.pyplot as plt
import matplotlib.patches as patches

import geopandas as gpd
from shapely.geometry import Point

# -----------------------
# Paramètres utilisateur
# -----------------------

# -----------------------
# Zone de recherche (centre GPS + demi-côté en m)
# -----------------------

# Bordeaux
center_lat = 44.8359
center_lon = -0.5304
half_width_m  = 50000
half_height_m = 50000

# Entrées
asc_dir = "../data/BDALTI_ASC"                # dossier contenant les .asc IGN
viirs_file = "../data/viirs_2024.tif"         # Carte VIIRS (EPSG:4326)
road_shp = "../data/BDTOPO/TRONCON_DE_ROUTE.shp"     # chemin vers ton shapef

# -----------------------
# Paramètres accessibilité
# -----------------------
max_distance_road_m = 100                     # distance maximale en m

# Observation (orientation et angle du cône)
observation_angle_deg = 0      # orientation (0=nord, 90=est, 180=sud, etc.) 
observation_width_deg = 45     # ouverture du cône en degrés
obs_small_base_m = 3000

# Distances max (en mètres)
plain_distance_m = 3000        # vérif relief (plaine) dans un rayon de 3 km
city_distance_m  = 30000       # vérif villes lumineuses dans un rayon de 30 km
city_halfwidth_m = 5000        # demi-largeur de 5 km

# Seuils
light_threshold = 2.5          # radiance VIIRS locale max
city_threshold = 18            # seuil ville
altitude_threshold = 40        # altitude mini (m)
contrast_threshold = 15        # delta_h minimal (m)
peak_distance_m = 500          # distance minimale entre pics
contrast_radius_m = 500        # rayon pour calcul du contraste (m)
viirs_radius_m = 250           # ton rayon en m

# Poids
win_weight = [1.0, 0.666, 0.333]

# Sorties
geojson_output = "../extracts/spots.geojson"  # sortie GeoJSON
kml_output = "../extracts/spots.kml"          # sortie KML
debug_output = "../extracts/debug.png"        # sortie Debug

# -----------------------
# Fonctions utilitaires
# -----------------------

def asc_bbox(asc_file):
    """Retourne xmin, ymin, xmax, ymax en Lambert-93 du fichier ASC"""
    with open(asc_file, 'r', encoding='utf-8', errors='ignore') as f:
        header = {}
        for _ in range(6):
            line = f.readline()
            if not line:
                continue
            parts = line.split()
            if len(parts) >= 2:
                header[parts[0].lower()] = float(parts[1])
    ncols = int(header.get('ncols'))
    nrows = int(header.get('nrows'))
    xll = header.get('xllcorner')
    yll = header.get('yllcorner')
    cell = header.get('cellsize')
    xmin = xll
    ymin = yll
    xmax = xll + ncols * cell
    ymax = yll + nrows * cell
    return xmin, ymin, xmax, ymax

def intersects(b1, b2):
    """Retourne True si deux bounding boxes s’intersectent"""
    xmin1, ymin1, xmax1, ymax1 = b1
    xmin2, ymin2, xmax2, ymax2 = b2
    return not (xmax1 < xmin2 or xmax2 < xmin1 or ymax1 < ymin2 or ymax2 < ymin1)

def load_and_merge_asc(files, clip_bbox=None):
    """
    Charge et fusionne plusieurs fichiers .asc IGN (BD ALTI).
    Si clip_bbox est fourni (xmin,ymin,xmax,ymax en L93),
    clippe la mosaïque à cette zone.
    """
    datasets = [rasterio.open(f) for f in files]
    mosaic, dem_transform = rasterio.merge.merge(datasets)
    if mosaic.ndim == 3:
        mosaic = mosaic[0]
    dem_crs = rasterio.crs.CRS.from_epsg(2154)  # Lambert-93
    for ds in datasets:
        ds.close()

    if clip_bbox is not None:
        xmin_zone, ymin_zone, xmax_zone, ymax_zone = clip_bbox
        inv = ~dem_transform
        col_min_f, row_max_f = inv * (xmin_zone, ymin_zone)
        col_max_f, row_min_f = inv * (xmax_zone, ymax_zone)
        row_min = int(max(0, np.floor(min(row_min_f, row_max_f))))
        row_max = int(max(0, np.ceil(max(row_min_f, row_max_f))))
        col_min = int(max(0, np.floor(min(col_min_f, col_max_f))))
        col_max = int(max(0, np.ceil(max(col_min_f, col_max_f))))
        mosaic = mosaic[row_min:row_max, col_min:col_max]
        from rasterio.transform import Affine
        dem_transform = Affine(dem_transform.a, dem_transform.b, dem_transform.c + col_min * dem_transform.a,
                               dem_transform.d, dem_transform.e, dem_transform.f + row_min * dem_transform.e)

    return mosaic, dem_transform, dem_crs

def resample_viirs_to_dem(viirs_path, ref_shape, ref_transform, ref_crs):
    """
    Rééchantillonne le raster VIIRS (EPSG:4326) sur la grille du DEM.
    Retourne un tableau numpy aligné sur le DEM.
    """
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
    """
    Convertit une distance en mètres en nombre de pixels
    selon l'échelle du raster (transform).
    """
    px_x = meters / abs(transform.a)
    px_y = meters / abs(transform.e)
    return int(round(px_y)), int(round(px_x))

def extract_cone_bbox(arr, r, c, distance_px, angle_deg, width_deg, use_weight):
    """
    Approximation rapide d'un cône orienté
    par n_rect rectangles couvrants (bounding boxes).
    Peut renvoyer aussi la liste des rectangles.
    """
    rows, cols = arr.shape
    theta = np.deg2rad(180-angle_deg)
    dx = np.sin(theta)
    dy = -np.cos(theta)
    half_span_max = distance_px * np.tan(np.deg2rad(observation_width_deg / 2))
    n_rect = len(win_weight)
    
    results = []
    boxes = []

    for i in range(n_rect):
        d0 = (i / n_rect) * distance_px
        d1 = ((i + 1) / n_rect) * distance_px

        t0 = d0 / distance_px
        t1 = d1 / distance_px
        half0 = t0 * half_span_max
        half1 = t1 * half_span_max

        r0 = r - d0 * dy
        c0 = c + d0 * dx
        r1 = r - d1 * dy
        c1 = c + d1 * dx

        c0_left  = c0 - half0 * np.cos(theta)
        r0_left  = r0 - half0 * np.sin(theta)
        c0_right = c0 + half0 * np.cos(theta)
        r0_right = r0 + half0 * np.sin(theta)

        c1_left  = c1 - half1 * np.cos(theta)
        r1_left  = r1 - half1 * np.sin(theta)
        c1_right = c1 + half1 * np.cos(theta)
        r1_right = r1 + half1 * np.sin(theta)

        min_r = int(max(0, np.floor(min(r0_left, r0_right, r1_left, r1_right))))
        max_r = int(min(rows, np.ceil(max(r0_left, r0_right, r1_left, r1_right))))
        min_c = int(max(0, np.floor(min(c0_left, c0_right, c1_left, c1_right))))
        max_c = int(min(cols, np.ceil(max(c0_left, c0_right, c1_left, c1_right))))

        if max_r > min_r and max_c > min_c:
            boxes.append((min_r, max_r, min_c, max_c))
            if use_weight: 
                vals_rect = arr[min_r:max_r, min_c:max_c].ravel()
                results.append(vals_rect * win_weight[i])
            else:
                results.append(arr[min_r:max_r, min_c:max_c])

    vals = np.array([]) if not results else np.concatenate([r.flatten() for r in results])
    return vals, boxes


def extract_trapezoid_bbox(arr, r, c, distance_px, half_span0_px, half_span1_px, angle_deg, use_weight):
    """
    Approximation rapide d'un trapèze isocèle orienté par
    n_rect rectangles couvrants (bounding boxes).
    """
    rows, cols = arr.shape
    theta = np.deg2rad(180-angle_deg)
    dx = np.sin(theta)
    dy = -np.cos(theta)
    n_rect = len(win_weight)
    
    results = []
    boxes = []

    for i in range(n_rect):
        d0 = (i / n_rect) * distance_px
        d1 = ((i + 1) / n_rect) * distance_px

        # demi-largeurs aux deux extrémités
        t0 = d0 / distance_px
        t1 = d1 / distance_px
        half0 = (1 - t0) * half_span0_px + t0 * half_span1_px
        half1 = (1 - t1) * half_span0_px + t1 * half_span1_px

        # positions centrales aux deux extrémités
        r0 = r - d0 * dy
        c0 = c + d0 * dx
        r1 = r - d1 * dy
        c1 = c + d1 * dx

        # on calcule les 4 coins du trapèze local
        # extrémité proche
        c0_left  = c0 - half0 * np.cos(theta)
        r0_left  = r0 - half0 * np.sin(theta)
        c0_right = c0 + half0 * np.cos(theta)
        r0_right = r0 + half0 * np.sin(theta)
        # extrémité éloignée
        c1_left  = c1 - half1 * np.cos(theta)
        r1_left  = r1 - half1 * np.sin(theta)
        c1_right = c1 + half1 * np.cos(theta)
        r1_right = r1 + half1 * np.sin(theta)

        # bounding box alignée à la grille
        min_r = int(max(0, np.floor(min(r0_left, r0_right, r1_left, r1_right))))
        max_r = int(min(rows, np.ceil(max(r0_left, r0_right, r1_left, r1_right))))
        min_c = int(max(0, np.floor(min(c0_left, c0_right, c1_left, c1_right))))
        max_c = int(min(cols, np.ceil(max(c0_left, c0_right, c1_left, c1_right))))

        if max_r > min_r and max_c > min_c:
            boxes.append((min_r, max_r, min_c, max_c))
            if use_weight: 
                vals_rect = arr[min_r:max_r, min_c:max_c].ravel()
                results.append(vals_rect * win_weight[i])
            else:
                results.append(arr[min_r:max_r, min_c:max_c])

    vals = np.array([]) if not results else np.concatenate([r.flatten() for r in results])
    return vals, boxes

def init_plot_overlay(dem, viirs):
    """
    Initialise une image montrant DEM+VIIRS en overlay, le point central et les rectangles approximant le cône.
    """
    fig, ax = plt.subplots(figsize=(8, 8))

    # Échelle DEM : 0 à max altitude
    ax.imshow(dem, cmap="terrain", origin="upper",
              vmin=0, vmax=80, alpha=0.5)

    # Échelle VIIRS : 0 à 30 nW/cm²/sr (ou ajuste selon ta carte)
    ax.imshow(viirs, cmap="inferno", origin="upper",
              vmin=0, vmax=20, alpha=0.5)
       
    return fig, ax

def add_plot_overlay(ax, r, c, boxes, color):
    """
    Génère une image montrant DEM+VIIRS en overlay, le point central et les rectangles approximant le cône.
    """
    ax.plot(c, r, "ro", label="Point central")

    for (min_r, max_r, min_c, max_c) in boxes:
        rect = plt.Rectangle((min_c, min_r),
                             max_c - min_c,
                             max_r - min_r,
                             edgecolor=color,
                             facecolor='none',
                             lw=1.5)
        ax.add_patch(rect)

# Transformer Lambert-93 -> WGS84
transformer = Transformer.from_crs("EPSG:2154", "EPSG:4326", always_xy=True)
transformer_to_l93 = Transformer.from_crs("EPSG:4326", "EPSG:2154", always_xy=True)

# -----------------------
# Pipeline
# -----------------------
center_x, center_y = transformer_to_l93.transform(center_lon, center_lat)

# Bbox DEM avec marge DEM          
bbox_dem = (center_x - half_width_m - plain_distance_m,
            center_y - half_height_m - plain_distance_m,
            center_x + half_width_m + plain_distance_m,
            center_y + half_height_m + plain_distance_m)            

# Bbox VIIRS avec marge plus large
bbox_viirs = (center_x - half_width_m - city_distance_m,
              center_y - half_height_m - city_distance_m,
              center_x + half_width_m + city_distance_m,
              center_y + half_height_m + city_distance_m)              

asc_all = glob.glob(os.path.join(asc_dir, "*.asc"))
dem_files = [f for f in asc_all if intersects(asc_bbox(f), bbox_viirs)]
if not dem_files:
    raise FileNotFoundError(f"Aucun fichier .asc trouvé dans la zone {bbox_viirs}")

#
print("Chargement DEM...")
dem, dem_transform, dem_crs = load_and_merge_asc(dem_files, clip_bbox=bbox_viirs)
rows, cols = dem.shape
print("DEM chargé :", dem.shape)

# convert wrong DEM value to NaN
dem[dem < -10000] = np.nan

print("Chargement des routes...")
roads = gpd.read_file(road_shp)

# S'assurer que les routes sont en Lambert-93 (EPSG:2154)
if roads.crs is None or roads.crs.to_epsg() != 2154:
    roads = roads.to_crs(epsg=2154)

# Un seul gros MultiLineString pour accélérer la distance
road_union = roads.unary_union
print("Routes chargées")

# Conversion mètres → pixels
plain_distance_px_y, _ = meters_to_pixels(plain_distance_m, dem_transform)
city_distance_px_y, _  = meters_to_pixels(city_distance_m, dem_transform)
city_halfwidth_px, _ = meters_to_pixels(city_halfwidth_m, dem_transform)
peak_distance_px_y, peak_distance_px_x = meters_to_pixels(peak_distance_m, dem_transform)
peak_distance_px = max(peak_distance_px_y, peak_distance_px_x)
contrast_radius_px_y, contrast_radius_px_x = meters_to_pixels(contrast_radius_m, dem_transform)
viirs_radius_px_y, viirs_radius_px_x = meters_to_pixels(viirs_radius_m, dem_transform)

half_span0_m = obs_small_base_m / 2
half_span1_m = city_distance_m * np.tan(np.deg2rad(observation_width_deg/2))
half_span0_px, _ = meters_to_pixels(half_span0_m, dem_transform)
half_span1_px, _ = meters_to_pixels(half_span1_m, dem_transform)

print("Rééchantillonnage VIIRS...")
viirs = resample_viirs_to_dem(viirs_file, dem.shape, dem_transform, dem_crs)

print("Calcul masque bbox_dem...")
xmin_dem, ymin_dem, xmax_dem, ymax_dem = bbox_dem

# Convertir bbox_dem en indices ligne/colonne
inv = ~dem_transform
col_min_f, row_max_f = inv * (xmin_dem, ymin_dem)
col_max_f, row_min_f = inv * (xmax_dem, ymax_dem)

row_min = int(max(0, np.floor(min(row_min_f, row_max_f))))
row_max = int(min(dem.shape[0], np.ceil(max(row_min_f, row_max_f))))
col_min = int(max(0, np.floor(min(col_min_f, col_max_f))))
col_max = int(min(dem.shape[1], np.ceil(max(col_min_f, col_max_f))))

# Masque True uniquement dans bbox_dem
mask = np.zeros_like(dem, dtype=bool)
mask[row_min:row_max, col_min:col_max] = True


print("Détection des pics (peak_local_max)...")
coordinates = peak_local_max(
    dem,
    min_distance=peak_distance_px,
    threshold_abs=altitude_threshold,
    exclude_border=False,
    labels=mask
)
print("Pics bruts:", len(coordinates))

print("Filtrage et calcul des indicateurs...")
valid_points = []
for (r, c) in coordinates:
    if np.isnan(dem[r, c]):
        continue

    # Contraste relatif dans un voisinage
    win = dem[max(0, r-contrast_radius_px_y):r+contrast_radius_px_y+1,
              max(0, c-contrast_radius_px_x):c+contrast_radius_px_x+1]
    if win.size == 0:
        continue
    local_median = np.nanmedian(win)
    delta_h = dem[r, c] - local_median

    # Lumière locale
    win_viirs = viirs[max(0, r-viirs_radius_px_y):r+viirs_radius_px_y+1,
                  max(0, c-viirs_radius_px_x):c+viirs_radius_px_x+1]
    if win_viirs.size == 0:
       continue
    viirs_local = float(np.nanmax(win_viirs))

    # Relief dans le cône approx
    plain_band, cboxes = extract_cone_bbox(dem, r, c,
                                           distance_px=plain_distance_px_y,
                                           angle_deg=observation_angle_deg,
                                           width_deg=observation_width_deg,
                                           use_weight=False)
    if plain_band.size == 0:
        continue    
    plain_max = float(np.nanmedian(plain_band))
    #plain_max = float(np.nanmax(plain_band))

    # Lumière dans le cône approx
    city_band, tboxes = extract_trapezoid_bbox(viirs, r, c,
                                               distance_px=city_distance_px_y,
                                               half_span0_px=half_span0_px,
                                               half_span1_px=half_span1_px,
                                               angle_deg=observation_angle_deg,
                                               use_weight=True)                                
                                    
    if city_band.size == 0:
        continue
    viirs_max_city = float(np.nanmax(city_band))
    
    valid_points.append((r, c, viirs_local, viirs_max_city, delta_h, plain_max, cboxes, tboxes))

print("Points retenus avant tri:", len(valid_points))

# -----------------------
# Tri multi-critères
# -----------------------
sorted_points = sorted(
    valid_points,
    key=lambda p: (
        p[2] if p[2] is not None else float("inf"),    # viirs_local asc
        -p[4],                                         # delta_h desc
        p[3] if p[3] is not None else float("inf"),    # viirs_max_city asc
        -dem[p[0], p[1]]                               # elevation desc
    )
)

# -----------------------
# Génération GeoJSON + KML avec groupes
# -----------------------
features = []
kml = simplekml.Kml()
folders = {}

fig, ax = init_plot_overlay(dem, viirs)

for i, (r, c, viirs_local, viirs_max_city, delta_h, plain_max, cboxes, tboxes) in enumerate(sorted_points):
    x, y = xy(dem_transform, r, c)
    lon, lat = transformer.transform(x, y)

    elevation = float(dem[r, c])
    d_route = road_union.distance(Point(x, y))

    # Déterminer le groupe selon tes critères
    if delta_h < contrast_threshold:
        group = "rejected_delta"
    elif plain_max > elevation:
        group = "rejected_plain"        
    elif viirs_local is None or viirs_local > light_threshold:
        group = "rejected_light"
    elif viirs_max_city is not None and viirs_max_city > city_threshold:
        group = "rejected_city"
    elif d_route > max_distance_road_m:
        group = "rejected_access"
    else:
        group = "valid"

    feature = {
        "type": "Feature",
        "properties": {
        "id": int(i),
        "group": group,
        "elevation": float(elevation),
        "plain_max": float(plain_max),
        "delta_h": float(delta_h),
        "viirs_local": float(viirs_local) if viirs_local is not None else None,
        "viirs_max_city": float(viirs_max_city) if viirs_max_city is not None else None
    },
    "geometry": {"type": "Point", "coordinates": [float(lon), float(lat)]}
    }
    features.append(feature)

    # Créer ou récupérer le Folder KML
    if group not in folders:
        folders[group] = kml.newfolder(name=group)

    # Ajouter un Placemark au bon Folder
    pnt = folders[group].newpoint(name=f"{i}", coords=[(lon, lat)])
    pnt.description = (f"Elevation: {elevation:.1f} m\n"
                       f"plain_max: {plain_max:.1f} m\n"
                       f"delta_h: {delta_h:.1f} m\n"                       
                       f"viirs_local: {viirs_local}\n"
                       f"viirs_max_city: {viirs_max_city}")
    
    if group == "rejected_plain":
        add_plot_overlay(ax, r, c, cboxes, "cyan")
    if group == "rejected_city" and viirs_local < 1:
        add_plot_overlay(ax, r, c, tboxes, "blue")

geojson_fc = {"type": "FeatureCollection", "features": features}

# Sauvegarde GeoJSON
os.makedirs(os.path.dirname(geojson_output), exist_ok=True)
with open(geojson_output, "w", encoding="utf-8") as f:
    json.dump(geojson_fc, f, ensure_ascii=False, indent=2)
print(f"GeoJSON généré : {geojson_output}")

# Sauvegarde KML
os.makedirs(os.path.dirname(kml_output), exist_ok=True)
kml.save(kml_output)
print(f"KML généré : {kml_output}")

# Sauvegarde debug image
plt.savefig(debug_output, dpi=150)
plt.close(fig)
print(f"Debug généré : {debug_output}")
