import json
import glob
import os

# List of j values (clusters)
cluster_js = [2, 8, 14, 15]  # <-- Replace with your actual list of j values
max_i = 2  # <-- Replace with the maximum possible i value (or set high, script will skip missing files)

all_data = []

for j in cluster_js:
    for i in range(max_i):
        filename = os.path.join("data", f"cluster_{j}_{i}.json")
        if os.path.exists(filename):
            with open(filename, 'r', encoding='utf-8') as f:
                try:
                    for line in f:
                        line = line.strip()
                        if line:
                            data = json.loads(line)
                            all_data.append(data)
                except Exception as e:
                    print(f"Error reading {filename}: {e}")

# Save combined data to a single file
with open("combined_clusters.json", "w", encoding="utf-8") as out_f:
    json.dump(all_data, out_f, ensure_ascii=False, indent=2)

print(f"Combined {len(all_data)} items into combined_clusters.json")
