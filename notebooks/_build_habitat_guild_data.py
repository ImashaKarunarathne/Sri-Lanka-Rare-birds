"""
Builds data/processed/bird_habitat_guild_classification.csv from
bird_uncommon_viable_classification.csv by collapsing the 5 species whose
environmental niches overlap heavily (pairwise standardized-feature centroid
distance < ~1.5, vs. 2.5-4.7 for the other 5) into two data-driven habitat
guilds. The two guilds were found by average-linkage hierarchical clustering
(k=2) on the 5 overlapping species' standardized feature centroids:

  - "Upland scrub/woodland-edge": Prinia sylvatica + Falco peregrinus
    (higher elevation ~190-360m, higher NDVI ~0.71, lower LandCover_Class ~9-10)
  - "Lowland open-country/wetland-edge": Gallinago stenura + Sturnia pagodarum
    + Alauda gulgula (lower elevation ~12-98m, lower NDVI ~0.56-0.63,
    higher LandCover_Class ~11-12)

The 5 remaining species (Harpactes fasciatus, Hirundo domicola,
Thalasseus bergii, Numenius arquata, Motacilla alba) are environmentally
distinct from everything else (centroid distance > 1.9 to their nearest
neighbour) and are kept as individual classes.
"""
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC = PROJECT_ROOT / "data" / "processed" / "bird_uncommon_viable_classification.csv"
DST = PROJECT_ROOT / "data" / "processed" / "bird_habitat_guild_classification.csv"

GUILD_MAP = {
    "Prinia sylvatica": "Upland scrub/woodland-edge",
    "Falco peregrinus": "Upland scrub/woodland-edge",
    "Gallinago stenura": "Lowland open-country/wetland-edge",
    "Sturnia pagodarum": "Lowland open-country/wetland-edge",
    "Alauda gulgula": "Lowland open-country/wetland-edge",
    "Harpactes fasciatus": "Harpactes fasciatus (rainforest)",
    "Hirundo domicola": "Hirundo domicola (highland)",
    "Thalasseus bergii": "Thalasseus bergii (marine coastal)",
    "Numenius arquata": "Numenius arquata (coastal wader)",
    "Motacilla alba": "Motacilla alba (urban/wetland-edge)",
}

df = pd.read_csv(SRC)
df["habitat_guild"] = df["verbatimScientificName"].map(GUILD_MAP)
assert df["habitat_guild"].isna().sum() == 0

DST.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(DST, index=False)

print("Wrote", DST)
print(df.shape)
print(df["habitat_guild"].value_counts())
print(df["habitat_guild"].value_counts(normalize=True).round(4))
