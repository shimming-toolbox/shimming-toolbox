import matplotlib.pyplot as plt
import numpy as np
import nibabel as nib
import tempfile
import os
from matplotlib.colors import LinearSegmentedColormap

from shimmingtoolbox.masking.mask_utils import (
    basic_softmask,
    linear_softmask,
    gaussian_filter_softmask,
    gaussian_sct_softmask
)

def plot_softmasks(path_output="."):
    """
    Plot softmasks for a poster with French labels and color theme.
    """

    # Create a binary mask (sphere)
    shape = (64, 64, 64)
    center = np.array(shape) // 2
    radius = 14

    x = np.arange(shape[0])[:, None, None]
    y = np.arange(shape[1])[None, :, None]
    z = np.arange(shape[2])[None, None, :]

    distance = np.sqrt((x - center[0])**2 + (y - center[1])**2 + (z - center[2])**2)
    sphere = distance <= radius
    binmask_array = sphere.astype(np.uint8)

    # Create a gaussian softmask (2D Gaussian repeated across z)
    sigma = radius
    center_2d = np.array(shape[:2]) // 2
    x = np.arange(shape[0])[:, None]
    y = np.arange(shape[1])[None, :]
    gaussian_2d = np.exp(-((x - center_2d[0])**2 + (y - center_2d[1])**2) / (2 * sigma**2))
    gaussian_2d /= gaussian_2d.max()
    gauss = np.repeat(gaussian_2d[:, :, None], shape[2], axis=2)
    gauss_array = gauss.astype(np.float32)

    # Save temporary NIfTI files
    with tempfile.NamedTemporaryFile(suffix=".nii.gz", delete=False) as tmp_bin, \
         tempfile.NamedTemporaryFile(suffix=".nii.gz", delete=False) as tmp_gss:

        nib.save(nib.Nifti1Image(binmask_array, affine=np.eye(4)), tmp_bin.name)
        nib.save(nib.Nifti1Image(gauss_array, affine=np.eye(4)), tmp_gss.name)

        # Generate softmasks
        cst_softmask = basic_softmask(tmp_bin.name, 6, 0.5)
        lin_softmask = linear_softmask(tmp_bin.name, 6)
        gss_softmask = gaussian_filter_softmask(tmp_bin.name, 6)
        sum_softmask = gaussian_sct_softmask(tmp_bin.name, tmp_gss.name)

    # Clean up
    os.remove(tmp_bin.name)
    os.remove(tmp_gss.name)

    # Prepare colormap and styling
    gold_cmap = LinearSegmentedColormap.from_list("black-to-gold", ['#0D1B2A', '#D4AF37'], N=256)
    fig, axes = plt.subplots(1, 5, figsize=(20, 5), facecolor='#E3E3E3')

    titles = [
        '0) Masque binaire',
        '1) Masque continu \nconstant',
        '2) Masque continu \nlinéaire',
        '3) Masque continu par \nfiltrage gaussien',
        "4) Masque continu par \naddition d'une gaussienne",
    ]
    images = [
        (binmask_array, gold_cmap),
        (cst_softmask, gold_cmap),
        (lin_softmask, gold_cmap),
        (gss_softmask, gold_cmap),
        (sum_softmask, gold_cmap),
    ]

    for ax, (img, cmap), title in zip(axes.flat, images, titles):
        ax.imshow(img[:, :, 32], cmap=cmap, vmin=0, vmax=1)
        ax.set_title(title, fontsize=20, color='#0D1B2A')
        ax.axis('off')

    plt.tight_layout()

    # Save the figure as SVG
    fname_svg = os.path.join(path_output, 'figure_softmasks.svg')
    fig.savefig(fname_svg, format='svg', bbox_inches='tight', facecolor=fig.get_facecolor())

    # Optional PNG version
    fname_png = os.path.join(path_output, 'figure_softmasks.png')
    fig.savefig(fname_png, dpi=300, bbox_inches='tight', facecolor=fig.get_facecolor())

    plt.show()

if __name__ == "__main__":
    plot_softmasks("/Users/antoineguenette/Desktop/figures_affiche")
