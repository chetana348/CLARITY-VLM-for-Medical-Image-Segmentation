import pandas as pd

# Define PDAC class descriptions
data = [
    {
        "id": 0,
        "Title": "Background",
        "Shape": "No discrete lesion",
        "Location": "N/A",
        "Appearance/Density": "N/A",
        "Contour/Symmetry": "Smooth",
        "Internal Texture": "Homogeneous"
    },
    {
        "id": 1,
        "Title": "Pancreatic ductal adenocarcinoma",
        "Shape": "Irregular or lobulated mass",
        "Location": "Typically in pancreatic head or body",
        "Appearance/Density": "Hypoattenuating mass with poorly defined margins",
        "Contour/Symmetry": "Infiltrative and ill-defined",
        "Internal Texture": "Heterogeneous with necrotic or low-density components"
    }
]

df = pd.DataFrame(data)

output_path = "classes_definitions_pdac.csv"
df.to_csv(output_path, index=False)

print(f"CSV saved to: {output_path}")
print(df)
