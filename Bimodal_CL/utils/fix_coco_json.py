import json

PREFIX = "COCO_val2014_"

with open("/BS/dduka/work/projects/TempNet/Bimodal_CL/clip_train/coco_test_new_.json", "r") as f:
    coco = json.load(f)

    for item in coco:
        image_path = item["image"]
        tokens = image_path.split("/")
        item["image"] = tokens[0] + "/" + PREFIX + tokens[1]

with open("/BS/dduka/work/projects/TempNet/Bimodal_CL/clip_train/coco_test_new.json", "w") as f:
    json.dump(coco, f)