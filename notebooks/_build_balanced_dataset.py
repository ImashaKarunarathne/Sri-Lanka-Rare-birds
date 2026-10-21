"""
Builds data/processed/bird_habitat_guild_balanced_classification.csv from the
FULL raw Sri Lanka Bird Diversity Dataset (1,552,048 rows, 429 species) --
not the 12,010-row v3 subsample.

Discovery process (see conversation / notebook 08 for the full exploration):
1. Computed per-species occurrence counts + environmental centroids for all
   429 species via chunked aggregation over the full CSV.
2. Clustered the 230 species with 300-10,000 occurrences (K-Means, k=6, on
   standardized environmental features) -- this independently rediscovered
   almost exactly the v4/v6 guild structure (highland endemics, wet-zone
   forest, coastal/wetland, etc.), validating the earlier hand-built scheme.
3. Tried using WHOLE clusters (many species pooled per class) to get more
   balanced, bigger classes -- this badly HURT the accuracy ceiling (down to
   64.3%), because many unrelated species share identical environmental grid
   cells once enough of them are pooled together. Rejected.
4. Instead, searched the full species list for the species environmentally
   closest to our existing smallest class (Motacilla alba, 640 occurrences)
   and found Motacilla citreola (Citrine Wagtail, 698 occurrences, a genuine
   sibling species) sitting right next to it -- pairing them raises that
   guild to 1,338 records without hurting separability (ceiling unchanged
   at 88.1% for the full, still-imbalanced data).
5. Undersampled every one of the 6 classes to the smallest class's size
   (1,275, Hirundo domicola) for a perfectly balanced dataset. This alone
   raised the ceiling further, to 93.3% -- fewer rows per class means fewer
   cross-class duplicate-feature collisions, so balancing is not just fairer
   but genuinely more learnable.
"""
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FULL_ROWS = PROJECT_ROOT / "data" / "processed" / "full_candidate_species_rows.csv"
DST = PROJECT_ROOT / "data" / "processed" / "bird_habitat_guild_balanced_classification.csv"
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

df_all = pd.read_csv(FULL_ROWS)
df = df_all[df_all["verbatimScientificName"].isin(GUILD_MAP.keys())].copy()
df["guild"] = df["verbatimScientificName"].map(GUILD_MAP)

min_n = df["guild"].value_counts().min()
parts = [g.sample(n=min_n, random_state=RANDOM_STATE) for _, g in df.groupby("guild")]
balanced = pd.concat(parts, ignore_index=True)

DST.parent.mkdir(parents=True, exist_ok=True)
balanced.to_csv(DST, index=False)

print("Wrote", DST)
print(balanced.shape)
print(balanced["guild"].value_counts())
print(balanced["verbatimScientificName"].value_counts())
