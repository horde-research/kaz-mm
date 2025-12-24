import json

SUBJECT_MAPPING = {
    "People": "Person/People",
    "Crowd (>5)": "Person/People",
    "Small Group (2-5)": "Person/People",
    "Single Person": "Person/People",
    "Human(s)": "Person/People",
    "Group of People": "Person/People",
    "Human/People": "Person/People",
    "Clothing/People": "Person/People",

    "Plant": "Plant(s)",
    "Plant/Other": "Plant(s)",
    "Plant(s)/Flower(s)": "Plant(s)",
    "Plant(s)/Trees": "Plant(s)",

    "Text": "Other",
    "Text/Other": "Other",
    "Text Document": "Other",
    "Text/Document": "Other",
    "Text Overlay": "Other",
    "Text/Signage": "Other",

    "Map": "Other",
    "Character(s)": "Other",
    "Product(s)": "Other",
    "Building": "Other",
    "Human Presence": "Other",
    "Architecture/Scenery": "Scenery/Architecture",
}

def normalize_primary_subject(data):
    if isinstance(data, dict):
        for key, value in data.items():
            if key == "primary_subject" and isinstance(value, str):
                new_value = SUBJECT_MAPPING.get(value, value)
                data[key] = new_value
            else:
                normalize_primary_subject(value)
    elif isinstance(data, list):
        for item in data:
            normalize_primary_subject(item)
    return data


with open('/home/vitalymorozov/PycharmProjects/horde-common/user/vitaly_m/KazDataset_Analysis/full_dataset_with_captions_0208.json', "r", encoding="utf-8") as f:
    data = json.load(f)

normalized_data = normalize_primary_subject(data)

with open("/home/vitalymorozov/PycharmProjects/horde-common/user/vitaly_m/KazDataset_Analysis/full_dataset_with_captions_0308.json", "w", encoding="utf-8") as f:
    json.dump(normalized_data, f, ensure_ascii=False, indent=2)