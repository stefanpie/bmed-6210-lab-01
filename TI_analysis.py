from pathlib import Path

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

from IR_curve_fit import curve_fit_FLAIR, IR_func


data_dir = Path('./data')

image_files = list(data_dir.glob('**/*.IMA'))
image_files = [Path(f) for f in image_files]
image_files = sorted(image_files)

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

fig, ax = plt.subplots(1, 2, figsize=(10, 4))
ax[0].imshow(img_stack_sum, cmap=plt.cm.gray)
for idx, r in enumerate(regions):
    ax[0].plot(r.centroid[1], r.centroid[0], color='w', marker='x')
    ax[0].text(r.centroid[1] + 10 , r.centroid[0] - 10 , str(idx), color='w')
ax[1].imshow(label_image, cmap=plt.cm.nipy_spectral)
for idx, r in enumerate(regions):
    ax[1].plot(r.centroid[1], r.centroid[0], color='w', marker='x')
    ax[1].text(r.centroid[1] + 10 , r.centroid[0] - 10 , str(idx), color='w')
plt.suptitle('Sum of Image Stack and Extracted Regions')
plt.tight_layout()
plt.savefig('./regions.png', dpi=300)

image_data_df = pd.DataFrame(columns=['region', "TI", "signal"])
for flair_image_fp in flair_images:
    img = dicomio.read_file(flair_image_fp)
    img_data = img.pixel_array    
    TI = float(img["InversionTime"].value) * 1e-3

    for idx, r in enumerate(regions):
        pixel_coords = np.array(r.coords)
        pixel_values = img_data[pixel_coords[:, 0], pixel_coords[:, 1]]
        avg = np.median(pixel_values)

        image_data_df =  pd.concat([image_data_df, pd.DataFrame({
            'region': idx,
            'TI': TI,
            'signal': avg
        }, index=[0])])

# line plot with dotted line and x for each point
# sns.lineplot(data=image_data_df, x="TI", y="signal", hue="region", markers=True, dashes=True)
# plt.show()

region_fit_df = pd.DataFrame(columns=["region", "T1_fit", "M0_fit"])
for region in image_data_df.region.unique():
    region_df = image_data_df[image_data_df.region == region]
    region_df = region_df.sort_values(by="TI")
    TI = region_df.TI.values
    signal = region_df.signal.values

    res_x = curve_fit_FLAIR(TI, signal)
    region_fit_df = pd.concat([region_fit_df, pd.DataFrame({
        'region': region,
        'T1_fit': res_x[0],
        'M0_fit': res_x[1]
    }, index=[0])])

region_fit_df_for_table = region_fit_df.copy()
region_fit_df_for_table['region'] = region_fit_df_for_table['region'].astype(int)
region_fit_df_for_table['T1_fit'] = region_fit_df_for_table['T1_fit'] * 1e3
region_fit_df_for_table['M0_fit'] = region_fit_df_for_table['M0_fit']
region_fit_latex_table = region_fit_df_for_table.to_latex(index=False, formatters={'T1_fit': '{:.2f}'.format, 'M0_fit': '{:.0f}'.format})
print(region_fit_latex_table)

region_fit_computed_df = pd.DataFrame(columns=["region", "TI_fit", "signal_fit"])
for region in region_fit_df.region.unique():
    T1 = region_fit_df[region_fit_df.region == region].T1_fit.values[0]
    M0 = region_fit_df[region_fit_df.region == region].M0_fit.values[0]

    TI = np.logspace(-3, 1, 1000)
    signal = IR_func(TI, T1, M0)

    region_fit_computed_df = pd.concat([region_fit_computed_df, pd.DataFrame({
        'region': region,
        'TI_fit': TI,
        'signal_fit': signal
    }, index=None)])



df_combined_0 = image_data_df.copy()
df_combined_0['fit'] = "real"
df_combined_1 = region_fit_computed_df.copy()
df_combined_1['fit'] = "fit"
df_combined_1.rename(columns={'TI_fit': 'TI', 'signal_fit': 'signal'}, inplace=True)
df_combined = pd.concat([df_combined_0, df_combined_1])

fig, ax = plt.subplots(figsize=(10, 6))

lin_plt = sns.lineplot(data=df_combined, x="TI", y="signal", hue="region", style="fit", ax=ax)
sns.move_legend(lin_plt, "lower left")

ax.set_xscale('log')
ax.set_xlim(1e-3, 1e1)
ax.set_ylim(0, 2000)

trans = ax.get_xaxis_transform()
for region in region_fit_df.region.unique():
    T1 = region_fit_df[region_fit_df.region == region].T1_fit.values[0]
    TI = T1 * np.log(2)
    ax.axvline(TI, color='red', linestyle='-', linewidth=0.5)
    ax.text(TI, 1.01, f"T1 = {T1 * 1e3:.2f} ms\nTI = {TI * 1e3:.2f} ms", rotation=45, color='red', fontsize=8, transform=trans)

ax.set_xlabel("Inversion Time (s)")
ax.set_ylabel("Signal")
ax.set_title("FLAIR Signal vs. Inversion Time\nUsing Recorded and Fitted Data", loc='left')


plt.tight_layout()
plt.savefig("./FLAIR_signal_vs_TI.png", dpi=300)
