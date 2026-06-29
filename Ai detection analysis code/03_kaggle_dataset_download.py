# Step 1: install/upgrade kaggle CLI (you got a version warning last time)
!pip install -q --upgrade kaggle

# Step 2: upload kaggle.json — make sure you select the ACTUAL credentials file,
# not a dataset CSV (that's what happened last time)
from google.colab import files
print("Select your kaggle.json file (from Kaggle > Account > Create New API Token)")
files.upload()

# Step 3: move it to the expected location + lock permissions
!mkdir -p ~/.kaggle
!cp kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json

# Step 4: download
!kaggle datasets download -d barkataliarbab/ai-vs-human-scientific-text-dataset

# Step 5: unzip
!unzip -o ai-vs-human-scientific-text-dataset.zip -d aigtxt_dataset

# Step 6: inspect — same diagnostic as before
import pandas as pd, os
print(os.listdir("aigtxt_dataset"))
