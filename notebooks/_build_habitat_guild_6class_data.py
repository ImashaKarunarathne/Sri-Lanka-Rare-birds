"""
Builds data/processed/bird_habitat_guild_6class_classification.csv from
bird_habitat_guild_classification.csv (the 7-class v4 target) by additionally
merging the 2 coastal species, which v3/v4's own centroid-distance analysis
already flagged as the closest remaining pair outside the big open-country
cluster (~1.7 standardized-feature distance):

  - Thalasseus bergii (marine coastal) + Numenius arquata (coastal wader)
    -> "Coastal (tern + curlew)"

Motivation: a duplicate-feature-row ceiling analysis showed 28.3% of rows
share identical environmental covariates with a different-labelled row, which
caps achievable accuracy at 86.9% for the 7-class target no matter how good
the model is. Merging the coastal pair raises that ceiling to 88.1% (still
not 90% -- 4 classes only reaches 89.1%, a near-binary split is needed for
95.9% -- but it's the next incremental step before the problem would have to
be redefined into something much less interesting than species/guild
classification).
"""
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC = PROJECT_ROOT / "data" / "processed" / "bird_habitat_guild_classification.csv"
DST = PROJECT_ROOT / "data" / "processed" / "bird_habitat_guild_6class_classification.csv"

MERGE_MAP = {
    "Thalasseus bergii (marine coastal)": "Coastal (tern + curlew)",
    "Numenius arquata (coastal wader)": "Coastal (tern + curlew)",
}

df = pd.read_csv(SRC)
df["habitat_guild_6"] = df["habitat_guild"].replace(MERGE_MAP)

DST.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(DST, index=False)

print("Wrote", DST)
print(df.shape)
print(df["habitat_guild_6"].value_counts())
print(df["habitat_guild_6"].value_counts(normalize=True).round(4))
