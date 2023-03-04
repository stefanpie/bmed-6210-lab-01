from pathlib import Path
from pprint import pp

import matplotlib.pyplot as plt
import matplotlib.transforms as transforms
import seaborn as sns
import numpy as np
import pandas as pd

from pydicom import dicomio

from skimage.filters import threshold_otsu
from skimage.segmentation import clear_border
from skimage.measure import label, regionprops
from skimage.morphology import closing, square

data_dir = Path('./data')

image_files = list(data_dir.glob('**/*.IMA'))
image_files = [Path(f) for f in image_files]
image_files = sorted(image_files)

slice_images = [f for f in image_files if 'GRE' in str(f)]

img_name = []
img_stack_list = []
slices_thick = []
for f in slice_images:
    img_name.append(f.stem)
    img = dicomio.read_file(f)
    img_stack_list.append(img.pixel_array)
    z_thickness = float(img["SliceThickness"].value) * 1e-2
    slices_thick.append(z_thickness)
    print(img["PixelSpacing"].value)

#df of only img name and slice thickness
data_df = pd.DataFrame({'img_name': img_name, 'slice_thickness': slices_thick})
data_df = data_df.sort_values(by=['slice_thickness', 'img_name'])
# only keep two values for each slice thickness
# if there are more than two slices with the same thickness, keep the first two
data_df = data_df.groupby('slice_thickness').head(2)
data_df = data_df.reset_index(drop=True)
data_df = data_df.groupby('slice_thickness').filter(lambda x: len(x) == 2)
data_df = data_df.reset_index(drop=True)

img_stack = np.array(img_stack_list).astype(float)
img_stack /= np.max(img_stack)
img_stack_sum = np.sum(img_stack, axis=0)

thresh = threshold_otsu(img_stack_sum)
bw = closing(img_stack_sum > thresh, square(3))
cleared = clear_border(bw)

label_image = label(cleared)
regions = regionprops(label_image)
regions = sorted(regions, key=lambda x: x.centroid[0] + x.centroid[1])

# get bottom right region
last_region = None
for r in regions:
    if last_region is None:
        last_region = r
    else:
        if r.centroid[0] + r.centroid[1] > last_region.centroid[0] + last_region.centroid[1]:
            last_region = r

fig, ax = plt.subplots(1, 2, figsize=(10, 4))
ax[0].imshow(img_stack_sum, cmap=plt.cm.gray)
for idx, r in enumerate(regions):
    ax[0].plot(r.centroid[1], r.centroid[0], color='w', marker='x')
    ax[0].text(r.centroid[1] + 10 , r.centroid[0] - 10 , str(idx), color='w')
ax[1].imshow(label_image, cmap=plt.cm.nipy_spectral)
for idx, r in enumerate(regions):
    ax[1].plot(last_region.centroid[1], last_region.centroid[0], color='w', marker='x')
    ax[1].text(last_region.centroid[1] + 10 , last_region.centroid[0] - 10 , "last_region", color='w')
plt.suptitle('Sum of Image Stack and Extracted Regions')
plt.tight_layout()
plt.savefig('./regions_snr.png', dpi=300)

new_data_df = pd.DataFrame(columns=['slice_thickness', 'SNR'])

# for each slice thickness, get the two images
for name, group in data_df.groupby('slice_thickness'):
    name_0 = group.iloc[0]['img_name']
    thickness_0 = group.iloc[0]['slice_thickness']
    slice_0 = img_stack_list[img_name.index(name_0)]

    name_1 = group.iloc[1]['img_name']
    thickness_1 = group.iloc[1]['slice_thickness']
    slice_1 = img_stack_list[img_name.index(name_1)]

    roi_mean_0 = np.mean(slice_0[last_region.coords[:, 0], last_region.coords[:, 1]])
    roi_mean_1 = np.mean(slice_1[last_region.coords[:, 0], last_region.coords[:, 1]])

    diff_img = slice_1 - slice_0
    diff_std = np.std(diff_img[last_region.coords[:, 0], last_region.coords[:, 1]])

    SNR = np.sqrt(2) * roi_mean_0 / diff_std

    new_data_df = pd.concat([new_data_df, pd.DataFrame({'slice_thickness': [thickness_0], 'SNR': [SNR]})])

new_data_df = new_data_df.reset_index(drop=True)
print(new_data_df)

# plot data
fig, ax = plt.subplots(1, 1, figsize=(6, 6))
ax.plot(new_data_df['slice_thickness']*1e2, new_data_df['SNR'], '-o')
ax.set_xlabel('Slice Thickness (cm)')
ax.set_ylabel('SNR')
ax.set_title('SNR vs Slice Thickness')
plt.tight_layout()
plt.savefig('./snr.png', dpi=300)





