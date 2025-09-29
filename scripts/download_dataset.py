#Importing Path from pathlib library - to work easier with file paths.
from pathlib import Path

#Setting Paths
ZIP_URL   = "https://prod-dcd-datasets-cache-zipfiles.s3.eu-west-1.amazonaws.com/r6h24d2d3y-1.zip" #Do NOT change this path. It is the direct link to download the dataset.
DEST_DIR  = Path("/content/drive/MyDrive/AI/Project_Summer_Module")   #CHANGE your chosen DESTINATION folder where you want to save the dataset.
LOCAL_ZIP = Path("/content") / "UIdataGB_dataset.zip"                  #Temporary download path for the zip.

#Create destination (if it doesn’t exist)
DEST_DIR.mkdir(parents=True, exist_ok=True)

print(f"Zip will be downloaded to: {LOCAL_ZIP}")
print(f"Dataset will be extracted to: {DEST_DIR}")
