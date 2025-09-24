import os
import numpy as np
import rasterio
from rasterio.windows import from_bounds
from rasterio.warp import calculate_default_transform, reproject, Resampling
from scipy.fft import fft2, ifft2, fftshift, ifftshift
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.filters import sobel, meijering
from skimage.measure import label, regionprops
import pandas as pd

# ---------------------------------------------------------------------
# Setup paths
# ---------------------------------------------------------------------
workdir = r"...\image_segmentation"
infile_ls = os.path.join(workdir, "Landsat9_SR_RGB_MexicoCity.tif")  # RGB only

cropped_file = os.path.join(workdir, "Landsat_cropped.tif")
reproj_file = os.path.join(workdir, "Landsat_reproj.tif")
outfile_lp = os.path.join(workdir, "Landsat_butter_lowpass.tif")
outfile_hp = os.path.join(workdir, "Landsat_butter_highpass.tif")
fig_compare = os.path.join(workdir, "Landsat_comparison.png")
metrics_csv = os.path.join(workdir, "Landsat_metrics.csv")

# ---------------------------------------------------------------------
# Crop bounding box (lat/lon)
# ---------------------------------------------------------------------
min_lon, min_lat = -99.158, 19.33 # bounding box coordinates
max_lon, max_lat = -98.923, 19.52 # bounding box coordinates

with rasterio.open(infile_ls) as src: # opens input raster file
    bbox = rasterio.warp.transform_bounds(
        {"init": "EPSG:4326"}, src.crs, min_lon, min_lat, max_lon, max_lat
    )                                  # converts bounding box crs to image crs
    window = from_bounds(*bbox, transform=src.transform) # creates window for bounding box
    cropped = src.read(window=window).astype(np.float32) # reads image data that is inside bounding box
    cropped_transform = src.window_transform(window) # creates new affine transform for the cropped image.
    cropped_profile = src.profile # copies original image metadata
    cropped_profile.update({    # updates cropped image metadata
        "height": cropped.shape[1],
        "width": cropped.shape[2],
        "transform": cropped_transform
    })

with rasterio.open(cropped_file, "w", **cropped_profile) as dst:
    dst.write(cropped) # save file


# ---------------------------------------------------------------------
# Reprojection / Resampling
# ---------------------------------------------------------------------
dst_crs = 'EPSG:32614'  # UTM zone 14N for Mexico City, target CRS for reprojection
dst_resolution = 15  # meters per pixel, resampling resolution

with rasterio.open(cropped_file) as src:
    transform, width, height = calculate_default_transform(
        src.crs, dst_crs, src.width, src.height, *src.bounds, resolution=dst_resolution
    )  # opens cropped image and calculates the transformation parameters needed for reprojection

    profile_reproj = src.profile.copy() #gets metadata from cropped image for reprojected image
    profile_reproj.update({
        'crs': dst_crs,
        'transform': transform,
        'width': width,
        'height': height
    })

    reprojected_band = np.zeros((src.count, height, width), dtype=np.float32)
    for i in range(src.count):
        reproject(
            source=src.read(i+1),
            destination=reprojected_band[i],
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=transform,
            dst_crs=dst_crs,
            resampling=Resampling.bilinear
        )  # - Reprojects and resamples the cropped image using bilinear resampling

with rasterio.open(reproj_file, 'w', **profile_reproj) as dst:
    dst.write(reprojected_band) # save file


# ---------------------------------------------------------------------
# Butterworth filter functions
# ---------------------------------------------------------------------
def butterworth_lowpass(shape, cutoff, order=8): # defines low pass Butterworth filter function
    rows, cols = shape # Extracts the number of rows and columns from the input
    crow, ccol = rows // 2, cols // 2 # Determines the **center point** of the filter matrix
    y, x = np.ogrid[:rows, :cols] # creates two grids for row and col. idnices of filter
    dist = np.sqrt((x-ccol)**2 + (y-crow)**2) # Euclidean distance of pixels
    return 1 / (1 + (dist / cutoff)**(2*order)) #Implements Butterworth filter

def butterworth_highpass(shape, cutoff, order=3):  # defines high-pass pass Butterworth filter function
    return 1 - butterworth_lowpass(shape, cutoff, order) #Creates the high-pass filter by subtracting a **low-pass Butterworth filter** from `1`


def apply_filter(img, mask): #Defines FFT function
    f = fft2(img) #Computes the FFT
    fshift = fftshift(f) #shifts 0-frequency components from corners to center of frequncy spectrum
    fshift_filtered = fshift * mask #applies mask (i.e., the filter) to shifted frequency data
    f_ishift = ifftshift(fshift_filtered) #performs inverse FFT shift
    return np.real(ifft2(f_ishift)) #performs inverse FFT

def normalize_band(band): #Defines normalization function
    band = band - np.min(band) #finds minum value in band
    return band / np.max(band) if np.max(band) > 0 else band #calculates max value, if > 0, divides by max

def edge_density(img): #Defines edge density function
    edges = sobel(img) #applies Sobel filter
    return np.sum(edges > 0.01) / edges.size #only consider Sobel filter values > 0.01

# ---------------------------------------------------------------------
# Object count
# ---------------------------------------------------------------------
def object_count_filtered(img, threshold=0.3, min_size=10): #defines function that counts objects in image
    bw = img > threshold #takes pixels > threshold
    labeled = label(bw) #assigns label to each object
    props = regionprops(labeled) #gets properties of each labeled object
    count = sum(p.area >= min_size for p in props) #counts objects > min size
    return count

def entropy(img, bins=256): #function that calculates entropy
    img_scaled = np.clip(img*255, 0, 255).astype(np.uint8) #scales pixel values between 0 and 1
    hist, _ = np.histogram(img_scaled.ravel(), bins=bins, range=(0,256)) #calcuates histogram
    prob = hist/np.sum(hist) #gets probability distribution
    prob = prob[prob>0] #removes bins with prob = 0
    return -np.sum(prob*np.log2(prob)) #calulates entropy

# ---------------------------------------------------------------------
# Apply LP & HP Butterworth  to Red band
# ---------------------------------------------------------------------
red_band = reprojected_band[0]  # Red band after reprojection/resampling

cutoff_lp = 100 #cutoff frequency for low pass filter
order_lp = 8 #order of low pass filter
cutoff_hp = 20 #cutoff frequency for high pass filter
order_hp = 3 #order of high pass filter

img_lp = apply_filter(normalize_band(red_band), # applies low pass filter to normalized image
                      butterworth_lowpass(red_band.shape, cutoff_lp, order_lp))
img_hp = apply_filter(normalize_band(red_band), # applies high pass filter to normalized image
                      butterworth_highpass(red_band.shape, cutoff_hp, order_hp))

# ---------------------------------------------------------------------
# Meijering edge detection
# ---------------------------------------------------------------------
edges_meijering = meijering(normalize_band(red_band), sigmas=[1,2,3]) #detects edges in the cropped image
#normalizes pixels of output image between 0 and 1.
edges_meijering = (edges_meijering - edges_meijering.min()) / (edges_meijering.max() - edges_meijering.min() + 1e-8)

# ---------------------------------------------------------------------
# Compute metrics
# ---------------------------------------------------------------------
metrics = {}
red_norm = normalize_band(red_band) #cropped image is normalized

metrics['SSIM_LP'] = ssim(red_norm, normalize_band(img_lp), data_range=1.0) #SSIM butterworth low-pass
metrics['SSIM_HP'] = ssim(red_norm, normalize_band(img_hp), data_range=1.0) #SSIM butterworth high-pass
metrics['PSNR_LP'] = psnr(red_norm, normalize_band(img_lp), data_range=1.0) #PSNR butterworth low-pass
metrics['PSNR_HP'] = psnr(red_norm, normalize_band(img_hp), data_range=1.0) #PSNR butterworth high-pass

metrics['EdgeDensity_LP'] = edge_density(normalize_band(img_lp)) #edge density butterworth low-pass
metrics['EdgeDensity_HP'] = edge_density(normalize_band(img_hp)) #edge density butterworth high-pass
#edge density meijering
metrics['EdgeDensity_Meijering'] = np.sum(edges_meijering > 0.01) / edges_meijering.size


#object count butterworth low-pass
metrics['ObjectCount_LP'] = object_count_filtered(normalize_band(img_lp), threshold=0.3, min_size=10)
#object count butterworth high-pass
metrics['ObjectCount_HP'] = object_count_filtered(normalize_band(img_hp), threshold=0.3, min_size=10)
#object count meijering
metrics['ObjectCount_Meijering'] = object_count_filtered(edges_meijering, threshold=0.1, min_size=10)


metrics['Entropy_LP'] = entropy(normalize_band(img_lp))  #entropy butterworth low-pass
metrics['Entropy_HP'] = entropy(normalize_band(img_hp))  #entropy butterworth high-pass
metrics['Entropy_Meijering'] = entropy(edges_meijering) #entropy meijering

pd.DataFrame([metrics]).to_csv(metrics_csv, index=False)
print("Metrics saved to CSV:", metrics_csv)

# ---------------------------------------------------------------------
# Visualization: Red band + Meijering edges
# ---------------------------------------------------------------------
fig, axs = plt.subplots(1, 4, figsize=(24, 6)) #plots comparison file

axs[0].imshow(red_norm, cmap='gray')
axs[0].set_title("A. Original Red Band")
axs[0].axis('off')
axs[1].imshow(normalize_band(img_lp), cmap='gray')
axs[1].set_title("B. Red Butterworth LP")
axs[1].axis('off')
axs[2].imshow(normalize_band(img_hp), cmap='gray')
axs[2].set_title("C. Red Butterworth HP")
axs[2].axis('off')
axs[3].imshow(edges_meijering, cmap='gray')
axs[3].set_title("D. Meijering Edges")
axs[3].axis('off')

plt.tight_layout()
plt.savefig(fig_compare, dpi=300)
plt.close()


# ---------------------------------------------------------------------
# Save PNGs
# ---------------------------------------------------------------------
plt.imsave(os.path.join(workdir, "Landsat_reproj_red.png"), red_norm, cmap='gray', dpi=300)
plt.imsave(os.path.join(workdir, "Landsat_butter_lowpass.png"), normalize_band(img_lp), cmap='gray', dpi=300)
plt.imsave(os.path.join(workdir, "Landsat_butter_highpass.png"), normalize_band(img_hp), cmap='gray', dpi=300)
plt.imsave(os.path.join(workdir, "Landsat_edges_meijering.png"), edges_meijering, cmap='gray', dpi=300)


# ---------------------------------------------------------------------
# Save comparison PNG: Original vs Cropped vs Reprojected/Resampled Red band
# ---------------------------------------------------------------------
with rasterio.open(infile_ls) as src_full:
    try:
        full_rgb = src_full.read().astype(np.float32)
        full_red = full_rgb[0]
    except Exception:
        full_red = src_full.read(1).astype(np.float32)

full_red_norm = normalize_band(full_red)
cropped_red_norm = normalize_band(cropped[0])         # cropped red
reproj_red_norm = normalize_band(red_band)            # reprojected/resampled red

fig, axs = plt.subplots(1, 3, figsize=(18, 6))

axs[0].imshow(full_red_norm, cmap='gray')
axs[0].set_title("A. Original Red Band", fontsize=22)
axs[0].axis('off')

axs[1].imshow(cropped_red_norm, cmap='gray')
axs[1].set_title("B. Cropped Red Band", fontsize=22)
axs[1].axis('off')

axs[2].imshow(reproj_red_norm, cmap='gray')
axs[2].set_title("C. Reprojected/Resampled Red Band", fontsize=22)
axs[2].axis('off')

plt.tight_layout()
compare_red_png = os.path.join(workdir, "Landsat_red_comparison.png")
plt.savefig(compare_red_png, dpi=300)
plt.close()


# ---------------------------------------------------------------------
# Create combined figure: Original, FFT magnitude, LP, HP, Meijering edges
# ---------------------------------------------------------------------
# Compute FFT magnitude for visualization
f = fft2(normalize_band(red_band))
fshift = fftshift(f)
fft_magnitude = np.log1p(np.abs(fshift))  # log scale for better visibility
fft_magnitude = normalize_band(fft_magnitude)

fig, axs = plt.subplots(3, 2, figsize=(12, 18))  # 3 rows x 2 cols

# Row 1
axs[0, 0].imshow(red_norm, cmap='gray')
axs[0, 0].set_title("A. Original Red Band", fontsize=22)
axs[0, 0].axis('off')

axs[0, 1].imshow(fft_magnitude, cmap='gray')
axs[0, 1].set_title("B. FFT Magnitude", fontsize=22)
axs[0, 1].axis('off')

# Row 2
axs[1, 0].imshow(normalize_band(img_lp), cmap='gray')
axs[1, 0].set_title("C. Butterworth Low-Pass", fontsize=22)
axs[1, 0].axis('off')

axs[1, 1].imshow(normalize_band(img_hp), cmap='gray')
axs[1, 1].set_title("D. Butterworth High-Pass", fontsize=22)
axs[1, 1].axis('off')

# Row 3
axs[2, 0].imshow(edges_meijering, cmap='gray')
axs[2, 0].set_title("E. Meijering Edges", fontsize=22)
axs[2, 0].axis('off')

# Empty subplot for symmetry
axs[2, 1].axis('off')

plt.tight_layout()
fig_combined_png = os.path.join(workdir, "Landsat_red_all_steps.png")
plt.savefig(fig_combined_png, dpi=300)
plt.close()


# ---------------------------------------------------------------------
# Save filtered outputs as GeoTIFFs
# ---------------------------------------------------------------------
# Ensure they share the same georeference as the reprojected raster
profile_filtered = profile_reproj.copy()
profile_filtered.update(dtype=rasterio.float32, count=1)

# Low-pass TIFF
with rasterio.open(outfile_lp, "w", **profile_filtered) as dst:
    dst.write(normalize_band(img_lp).astype(np.float32), 1)


# High-pass TIFF
with rasterio.open(outfile_hp, "w", **profile_filtered) as dst:
    dst.write(normalize_band(img_hp).astype(np.float32), 1)


# Meijering edge TIFF
outfile_meijering = os.path.join(workdir, "Landsat_meijering.tif")
with rasterio.open(outfile_meijering, "w", **profile_filtered) as dst:
    dst.write(edges_meijering.astype(np.float32), 1)


# ---------------------------------------------------------------------
# Save reprojected red band as GeoTIFF
# ---------------------------------------------------------------------
outfile_red = os.path.join(workdir, "Landsat_reproj_red.tif")

profile_red = profile_reproj.copy()
profile_red.update(dtype=rasterio.float32, count=1)

with rasterio.open(outfile_red, "w", **profile_red) as dst:
    dst.write(red_band.astype(np.float32), 1)



# ---------------------------------------------------------------------
# Crop two bounding boxes from outputs and save PNGs-used to see details of filters
# ---------------------------------------------------------------------
# Define bounding boxes in EPSG:4326 (lon/lat)
bboxes = {
    "ROI1": (-99.081, 19.398, -98.937, 19.518),
    "ROI2": (-99.119, 19.365, -99.054, 19.423)
}

# Files to crop
tif_files = {
    "A. Reproj Red": outfile_red,
    "B. Butterworth LP": outfile_lp,
    "C. Butterworth HP": outfile_hp,
    "D. Meijering": outfile_meijering,
}

for roi_name, (min_lon, min_lat, max_lon, max_lat) in bboxes.items():
    cropped_imgs = []

    for label, path in tif_files.items():
        with rasterio.open(path) as src:
            # convert bbox to the image CRS (UTM in this case)
            bbox_proj = rasterio.warp.transform_bounds(
                "EPSG:4326", src.crs, min_lon, min_lat, max_lon, max_lat
            )
            window = from_bounds(*bbox_proj, transform=src.transform)
            data = src.read(1, window=window).astype(np.float32)
            cropped_imgs.append((label, normalize_band(data)))

    # Create figure in 2x2 grid
    n_imgs = len(cropped_imgs)
    nrows, ncols = 2, 2
    fig, axs = plt.subplots(nrows, ncols, figsize=(12, 12))
    axs = axs.ravel()  # flatten 2D axes array

    for ax, (label, img) in zip(axs, cropped_imgs):
        ax.imshow(img, cmap="gray")
        ax.set_title(label, fontsize=18)
        ax.axis("off")

    # Hide unused subplots if < 4 images
    for ax in axs[len(cropped_imgs):]:
        ax.axis("off")

    plt.tight_layout()
    out_png = os.path.join(workdir, f"Landsat_{roi_name}_comparison.png")
    plt.savefig(out_png, dpi=300)
    plt.close()


