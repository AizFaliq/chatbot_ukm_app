import json

# Load your existing JSON
with open("knowledge_base.json", "r", encoding="utf-8") as f:
    docs = json.load(f)

flattened = []
counter = 1

for entry in docs:
    subjek = entry.get("Subjek", "Unknown")
    tajuk = entry.get("Tajuk", "Unknown")
    units = entry.get("Unit", [])
    subtajuks = entry.get("Subtajuk", [])

    soalan_list = entry.get("Soalan", [])
    jawapan_list = entry.get("Jawapan", [])

    # Handle pairings
    for i, (q, a) in enumerate(zip(soalan_list, jawapan_list), start=1):
        flattened.append({
            "id": f"{subjek}_{tajuk}_Q{i}",   # e.g. Science_Electricity_Q1
            "document": f"Soalan: {q}\nJawapan: {a}",
            "metadata": {
                "Subjek": subjek,
                "Tajuk": tajuk,
                "Unit": units[i-1] if i-1 < len(units) else None,
                "Subtajuk": subtajuks[i-1] if i-1 < len(subtajuks) else None
            }
        })
        counter += 1

# Save adjusted JSON
with open("fst_data.json", "w", encoding="utf-8") as f:
    json.dump(flattened, f, ensure_ascii=False, indent=2)

print(f"Flattened knowledge base saved with {len(flattened)} entries")