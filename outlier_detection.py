import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from PIL import Image
from scipy.stats import zscore

# Define the path to your dataset
data_dir = 'D:/archive/Test_Images'

# Function to get image sizes
def get_image_sizes(subtypes):
    image_sizes = []
    for subtype in subtypes:
        folder_path = os.path.join(data_dir, subtype)
        for image_file in os.listdir(folder_path):
            image_path = os.path.join(folder_path, image_file)
            image = Image.open(image_path)
            width, height = image.size
            image_sizes.append((width, height))
    return image_sizes

# Define the cancer subtypes
subtypes = ['CC', 'EC', 'HGSC', 'LGSC', 'MC']

# Get image sizes
image_sizes = get_image_sizes(subtypes)

# Compute Z-scores for widths and heights
widths, heights = zip(*image_sizes)
width_zscores = zscore(widths)
height_zscores = zscore(heights)

# Set Z-score threshold for outlier detection
zscore_threshold = 3  # Adjust as needed

# Identify outliers
width_outliers = np.where(np.abs(width_zscores) > zscore_threshold)[0]
height_outliers = np.where(np.abs(height_zscores) > zscore_threshold)[0]
all_outliers = set(width_outliers).union(set(height_outliers))

# Visualize outliers
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
sns.scatterplot(x=widths, y=heights, color='skyblue', label='Normal')
sns.scatterplot(x=np.array(widths)[list(all_outliers)], y=np.array(heights)[list(all_outliers)], color='red', label='Outliers')
plt.title('Outlier Detection: Image Sizes')
plt.xlabel('Image Width')
plt.ylabel('Image Height')
plt.legend()

plt.subplot(1, 2, 2)
sns.histplot(width_zscores, bins=30, kde=True, color='skyblue', label='Width Z-scores')
sns.histplot(height_zscores, bins=30, kde=True, color='salmon', label='Height Z-scores')
plt.title('Distribution of Z-scores')
plt.xlabel('Z-score')
plt.legend()

plt.tight_layout()
plt.show()
