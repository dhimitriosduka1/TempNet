import json

all_captions = {}
nr_filenames = 0

for i in range(0, 111):
    if i in [97, 98, 99]:
        continue

    print(f"Processing file: {i}_captions.json")
    with open(
        f"/BS/dduka/work/projects/TempNet/Bimodal_CL/cc3m_extracted/{i}/{i}_captions.json"
    ) as f:
        data = json.load(f)
        nr_filenames += len(data)
        all_captions = all_captions | data


print(f"Length of all_captions: {len(all_captions)}")
print(f"Length of all_filenames: {nr_filenames}")

with open("/BS/dduka/work/projects/TempNet/Bimodal_CL/cc3m_extracted/merged.json", "w") as f:
    json.dump(all_captions, f)