import os
import json
import random

input_dir = "/fast/nchandak/forecasting/newsdata/testset/qgen/done"
output_file = "/fast/nchandak/forecasting/newsdata/testset/qgen/combined_sampled_shuffled.jsonl"
sample_size = 200

all_entries = []

# List all .jsonl files in the input directory
for fname in os.listdir(input_dir):
    if fname.endswith(".jsonl"):
        file_path = os.path.join(input_dir, fname)
        with open(file_path, "r", encoding="utf-8") as f:
            lines = [json.loads(line) for line in f if line.strip()]
            if len(lines) > sample_size:
                sampled = random.sample(lines, sample_size)
            else:
                sampled = lines  # If less than sample_size, take all
            all_entries.extend(sampled)

# Shuffle all combined entries
random.shuffle(all_entries)

# Write to output file
with open(output_file, "w", encoding="utf-8") as out_f:
    for entry in all_entries:
        out_f.write(json.dumps(entry, ensure_ascii=False) + "\n")

print(f"Combined {len(all_entries)} entries from {input_dir} into {output_file}")
