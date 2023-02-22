from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pydicom import dicomio

from skimage.filters import threshold_otsu
from skimage.segmentation import clear_border
from skimage.measure import label, regionprops
from skimage.morphology import closing, square
from skimage.color import label2rgb
from skimage import util

import seaborn as sns

data_dir = Path('./data')

# glob recursice all .IMA files
image_files = list(data_dir.glob('**/*.IMA'))
image_files = [Path(f) for f in image_files]
image_files = sorted(image_files)

# FLAIR images

flair_images = [f for f in image_files if 'TFISEG-TI' in str(f)]

img_stack_list = []
for f in flair_images:
    img = dicomio.read_file(f)
    img_stack_list.append(img.pixel_array)

img_stack = np.array(img_stack_list).astype(float)
img_stack /= np.max(img_stack)
img_stack_sum = np.sum(img_stack, axis=0)

thresh = threshold_otsu(img_stack_sum)
bw = closing(img_stack_sum > thresh, square(3))
cleared = clear_border(bw)

label_image = label(cleared)
regions = regionprops(label_image)
regions = sorted(regions, key=lambda x: x.centroid[0] + x.centroid[1])

# fig, ax = plt.subplots(figsize=(10, 6))
# ax.imshow(img_stack_sum, cmap=plt.cm.gray)
# for idx, r in enumerate(regions):
#     ax.plot(r.centroid[1], r.centroid[0], 'ro')
#     ax.text(r.centroid[1] + 10 , r.centroid[0] - 10 , str(idx), color='w')
# plt.show()

image_data_df = pd.DataFrame(columns=['TI', "region", "avg"])
for flair_image_fp in flair_images:
    img = dicomio.read_file(flair_image_fp)
    img_data = img.pixel_array    
    TI = float(img["InversionTime"].value)

    for idx, r in enumerate(regions):
        pixel_coords = np.array(r.coords)
        pixel_values = img_data[pixel_coords[:, 0], pixel_coords[:, 1]]
        avg = np.median(pixel_values)

        image_data_df =  pd.concat([image_data_df, pd.DataFrame({
            'TI': TI,
            'region': idx,
            'avg': avg
        }, index=[0])])

sns.lineplot(data=image_data_df, x="TI", y="avg", hue="region")
plt.show()