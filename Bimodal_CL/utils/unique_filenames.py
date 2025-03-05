import json

filenames = []

for i in range(0, 111):
    if i in [97, 98, 99]:
        continue

    print(f"Processing file: {i}_captions.json")
    with open(
       f"/BS/dduka/work/projects/TempNet/Bimodal_CL/cc3m_extracted/{i}/{i}_captions.json"
    ) as f:
        captions = json.load(f)

        for filename, caption in captions.items():
            filename = filename.split("_", 1)[1]
            filenames.append(filename)

print(f"Length of all filenames: {len(filenames)}")
print(f"Lenght of all set(filenames): {len(set(filenames))}")
