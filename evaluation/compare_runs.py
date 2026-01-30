import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

# ---------------- FILES ----------------

BASELINE = Path("evaluation/rag_results.csv")
V2 = Path("evaluation/rag_results_v2.csv")

OUT_CSV = Path("evaluation/rag_comparison.csv")
OUT_PNG = Path("evaluation/rag_comparison_chart.png")

# ---------------- LOAD ----------------

df1 = pd.read_csv(BASELINE)
df2 = pd.read_csv(V2)

m1 = df1.mean(numeric_only=True)
m2 = df2.mean(numeric_only=True)

delta = m2 - m1
pct = (delta / m1) * 100

# ---------------- COMBINED DF ----------------

compare_df = pd.DataFrame({
    "run_1": m1,
    "run_2": m2,
    "delta": delta,
    "percent_change": pct,
})

compare_df = compare_df.round(4)

OUT_CSV.parent.mkdir(exist_ok=True)
compare_df.to_csv(OUT_CSV)

print("\n===== COMPARISON SAVED =====")
print(compare_df)
print("\nCSV:", OUT_CSV)

# ---------------- CHART ----------------

plt.figure()
compare_df[["run_1", "run_2"]].plot(kind="bar")
plt.title("RAG Evaluation Comparison")
plt.ylabel("Score")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()

plt.savefig(OUT_PNG)
plt.close()

print("\nChart saved to:", OUT_PNG)
