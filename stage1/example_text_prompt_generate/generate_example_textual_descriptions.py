import os
import csv
import glob
import random

# -----------------------------------------
# Templates for generating 2–3 sentence descriptions
# -----------------------------------------
TEMPLATES = [
    [
        "This cropped Ktrans MRI slice shows a subtle hypointense region within the pancreatic parenchyma.",
        "The lesion exhibits reduced contrast uptake compared to the surrounding tissue.",
        "Its margins appear slightly irregular, consistent with PDAC physiology."
    ],
    [
        "In this Ktrans map, a darker, poorly enhancing region is noted within the pancreas.",
        "The low-perfusion area stands out against the normally enhancing background.",
        "This appearance is typical of a hypovascular PDAC lesion."
    ],
    [
        "This slice demonstrates a faintly demarcated hypointense focus in the pancreatic tissue.",
        "Contrast wash-in seems reduced, resulting in lower signal relative to the background.",
        "Texture heterogeneity suggests possible malignant infiltration."
    ],
    [
        "A localized low-signal region appears within the pancreatic head/body in this cropped Ktrans image.",
        "The lesion shows reduced permeability compared to adjacent normal structures.",
        "Edges are ill-defined, matching the infiltrative behavior of PDAC."
    ],
    [
        "The image contains a darker-than-normal region consistent with reduced Ktrans perfusion.",
        "Signal asymmetry hints at a possible underlying malignant mass.",
        "Border irregularity reinforces the suspicion of PDAC."
    ],
    [
        "This cropped MRI slice shows an area of hypo-enhancement relative to the surrounding pancreas.",
        "The contrast deficit gives the tumor a distinct darker appearance.",
        "Such perfusion loss aligns with typical PDAC contrast response patterns."
    ]
]


def make_description():
    """Return a 2–3 sentence description from templates."""
    block = random.choice(TEMPLATES)      # choose template group
    num_sent = random.choice([2, 3])      # use 2 or 3 sentences
    selected = random.sample(block, num_sent)
    return " ".join(selected)


# -----------------------------------------
# Main generator
# -----------------------------------------
def generate_description_csv(
    root_dir="/users/PAS3110/sephora20/workspace/PDAC/data/pdac_osu/cropped/ktrans",
    output_csv="pdac_ktrans_descriptions.csv"
):
    train_imgs = glob.glob(os.path.join(root_dir, "train", "images", "*.tif"))
    test_imgs  = glob.glob(os.path.join(root_dir, "test",  "images", "*.tif"))

    all_imgs = train_imgs + test_imgs
    all_imgs = sorted(all_imgs)

    rows = []

    for img_path in all_imgs:
        fname = os.path.basename(img_path)
        stem, _ = os.path.splitext(fname)       # image_id

        description = make_description()

        rows.append({
            "image_id": stem,
            "description": description
        })

    # Write CSV
    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["image_id", "description"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"✓ Description CSV saved at: {output_csv}")
    print(f"Total images processed: {len(rows)}")


if __name__ == "__main__":
    generate_description_csv()
