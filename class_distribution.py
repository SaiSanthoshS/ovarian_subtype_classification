#Class Distribution

import os
import matplotlib.pyplot as plt
import seaborn as sns

# Get the counts of images in each subtype
subtypes = ['CC', 'EC', 'HGSC', 'LGSC', 'MC']
subtype_counts = [len(os.listdir(os.path.join('D:/archive/Test_Images', subtype))) for subtype in subtypes]

# Plot the distribution
plt.figure(figsize=(10, 6))
sns.barplot(x=subtypes, y=subtype_counts, palette='viridis')
plt.title('Distribution of Images Across Subtypes')
plt.xlabel('Cancer Subtypes')
plt.ylabel('Number of Images')
plt.show()

print(subtype_counts)