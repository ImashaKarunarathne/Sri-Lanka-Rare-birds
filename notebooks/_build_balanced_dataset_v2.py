"""
Builds data/processed/bird_habitat_guild_balanced_with_count.csv -- the v8
dataset. Same 6 guilds as v7 (bird_habitat_guild_balanced_classification.csv),
same balancing approach (undersample every class to the smallest, 1,275 rows),
but adds `individualCount` (flock/group size at the sighting) to the feature
set.

Why this matters: investigating the duplicate-row accuracy ceiling further
showed lat/long are actually high-precision (6 decimal places) -- the ~55-66%
duplicate-feature-row rate seen throughout v3-v7 isn't a raster-resolution
artifact, it's that 82% of rows share their *exact* coordinates with another
row, because citizen-science observers repeatedly report from a small set of
known birding hotspots (2,135 unique coordinate pairs across 7,650 rows).
`individualCount` is a genuine behavioural trait (how large a group the
species was seen in) that's independent of location/climate, so it can break
ties that no amount of feature engineering from the existing spatial/climate
columns ever could (deterministic transforms of tied inputs are still tied).

Ceiling impact, measured directly from the data:
  - v7 (balanced, no individualCount): 93.3%
  - + individualCount, unbalanced:      95.7%
  - + individualCount, balanced:        97.9%  <- this dataset
"""
import time
from pathlib import Path
import pandas as pd

FULL_DATASET = Path(r"D:\Msc\Sem_2\Applied ML\Assignment and project\srilanka_bird_diversity\SriLanka-Bird-Diversity-Dataset.csv")
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DST = PROJECT_ROOT / "data" / "processed" / "bird_habitat_guild_balanced_with_count.csv"
RANDOM_STATE = 42

GUILD_MAP = {
    "Harpactes fasciatus": "Harpactes fasciatus (rainforest)",
    "Hirundo domicola": "Hirundo domicola (highland)",
    "Motacilla alba": "Motacilla alba + citreola (urban/wetland-edge)",
    "Motacilla citreola": "Motacilla alba + citreola (urban/wetland-edge)",
    "Thalasseus bergii": "Coastal (tern + curlew)",
    "Numenius arquata": "Coastal (tern + curlew)",
    "Gallinago stenura": "Lowland open-country/wetland-edge",
    "Sturnia pagodarum": "Lowland open-country/wetland-edge",
    "Alauda gulgula": "Lowland open-country/wetland-edge",
    "Prinia sylvatica": "Upland scrub/woodland-edge",
    "Falco peregrinus": "Upland scrub/woodland-edge",
}
KEEP_COLS = [
    "verbatimScientificName", "stateProvince", "decimalLatitude", "decimalLongitude",
    "eventDate", "avg_rad", "NDVI", "LandCover_Class", "elevation_meters", "individualCount",
    "Carbon_Mass", "Dust_Mass", "SO2_Mass", "Sulfate_Mass", "Sea_Salt_Mass",
    "Total_Aerosol_Extinction", "temp_mean", "rainfall", "wind_mean",
    "humid_mean", "shortwave_radiation", "lka_general_2020",
]

parts = []
for chunk in pd.read_csv(FULL_DATASET, usecols=KEEP_COLS, chunksize=200_000):
    parts.append(chunk[chunk["verbatimScientificName"].isin(GUILD_MAP.keys())])
df = pd.concat(parts, ignore_index=True)
df["guild"] = df["verbatimScientificName"].map(GUILD_MAP)

min_n = df["guild"].value_counts().min()
balanced_parts = [g.sample(n=min_n, random_state=RANDOM_STATE) for _, g in df.groupby("guild")]
balanced = pd.concat(balanced_parts, ignore_index=True)

DST.parent.mkdir(parents=True, exist_ok=True)
balanced.to_csv(DST, index=False)

print("Wrote", DST)
print(balanced.shape)
print(balanced["guild"].value_counts())
