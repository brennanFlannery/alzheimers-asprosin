import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2
from skimage.filters import apply_hysteresis_threshold
from skimage.morphology import remove_small_objects, remove_small_holes, binary_closing
from scipy.signal import medfilt, wiener
from tqdm import tqdm
import os

experiment_num = "1s"
# path = "C:/Users/flanm/OneDrive - case.edu/Documents/School/Asprosin Project/Asprosin AAV NLGF" # Experiment 2
# path = "C:/Users/flanm/OneDrive - case.edu/Documents/School/Asprosin Project/Calbindin PSTAT3/All Images"
# path = "C:/Users/flanm/OneDrive - case.edu/Documents/School/Asprosin Project/Experiment 3 AAV 5xFAD"
# path = "C:/Users/flanm/OneDrive - case.edu/Documents/School/Asprosin Project/Experiment 1"
path = "C:/Users/flanm/Downloads/Crus2-20241125T004331Z-001"
# files = glob.glob(path + "/*.tif")
# file = "ASP1 L10 M.tif"
def make_correlation_graph(c, name):
    dplot = []
    vplot = []
    for i in range(c.shape[0]):
        for j in range(c.shape[1]):
            value = np.average((c[i,j], c[j,i]))
            distance = np.sqrt(i**2 + j**2)
            dplot.append(distance)
            vplot.append(value)
    vplot = [x for _, x in sorted(zip(dplot, vplot))]
    vplot = medfilt(vplot, 1001)
    dplot = sorted(dplot)
    plt.figure()
    plt.plot(dplot, np.abs(vplot) , "-k")
    plt.title(name.split("\\")[-1])
    plt.savefig(name.split("\\")[0] + "/" + name.split("\\")[-1].split(".")[0] + ".jpg")

    return(dplot, vplot)

def calculate_overlap(files):
    cor_graphs = {"file":[], "correlation_values":[], "distance_values":[]}
    for file in tqdm(files):
        img = plt.imread(file)
        img_copy = img.copy()
        # plt.imshow(img)
        # plt.show()
        r = img[:, :, 0]
        g = img[:, :, 1]
        b = img[:, :, 2]

        # Correlation graph and value
        c = np.corrcoef(r,g)
        c_mean = np.mean(c)
        # distances, values = make_correlation_graph(c, file)
        # cor_graphs["file"].append(file)
        # cor_graphs["correlation_values"].append(values)
        # cor_graphs["distance_values"].append(distances)

        # Adaptive normalization
        clahe = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(10, 10))
        g_eq = clahe.apply(g)

        # First pass filter the green channel
        ret, thresh1 = cv2.threshold(g, 120, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        g_thresh = g.copy()
        g_thresh[g_thresh < ret] = 0

        # First pass filter the red channel
        ret, thresh2 = cv2.threshold(r, 120, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        r_thresh = r.copy()
        r_thresh[r_thresh < ret] = 0
        data = r_thresh[r_thresh > 0]
        low = np.percentile(data, 60)
        high = np.percentile(data, 80)
        r_filt = apply_hysteresis_threshold(r_thresh, low, high)

        # Hysteresis threshold
        data = g_thresh[g_thresh > 0]
        low = np.percentile(data, 60)
        high = np.percentile(data, 80)

        g_filt = apply_hysteresis_threshold(g_thresh, low, high)
        g_filt = remove_small_holes(g_filt)
        g_filt = binary_closing(g_filt)
        g_filt = remove_small_objects(g_filt, min_size=100)

        g_test = (g_filt * g_thresh)
        g_test = (g_test / np.max(g_test) * 255).astype(np.uint8())

        # plt.imshow(g_filt*1)
        # plt.show()
        g_filt = (g_thresh * g_filt)
        r_filt = (r_thresh * r_filt)
        img_copy[:, :, 1] = (g_filt / np.max(g_filt) * 255).astype(np.uint8())
        img_copy[:, :, 0] = (r_filt / np.max(r_filt) * 255).astype(np.uint8())

        overlap_img = (g_filt > 0) * (r_filt > 0) * 1
        overlap = np.sum(overlap_img)

        overlap_fixed = overlap / np.size(overlap_img)
        overlap_relative = overlap / np.sum((g_filt > 0) * 1)
        overlap_pixels = overlap

        data_final["overlap_fixed"].append(overlap_fixed)
        data_final["overlap_relative"].append(overlap_relative)
        data_final["overlap_pixels"].append(overlap_pixels)
        data_final["correlation"].append(c_mean)
        if experiment_num == 1:
            data_final["subject"].append(file.split("\\")[-1].split("crop")[0])
            data_final["region"].append(file.split("\\")[-1].split("crop")[1].split(".")[0])
            data_final["subregion"].append("N/A")
        elif experiment_num == 3:
            data_final["subject"].append(file.split("\\")[-2].split("-")[0])
            data_final["region"].append(file.split("\\")[-1].split("_")[2])
            data_final["subregion"].append("N/A")
        elif experiment_num == "1x2":
            data_final["subject"].append(file.split("\\")[-2])
            data_final["region"].append(file.split("\\")[-1].split("_")[1].split(".")[0])
            data_final["subregion"].append("N/A")
        elif experiment_num == "1s":
            data_final["subject"].append(file.split("\\")[-1].split("_")[0])
            data_final["region"].append(file.split("\\")[-1].split("_")[1].split(".")[0])
            data_final["subregion"].append("N/A")
        else:
            data_final["subject"].append(file.split("\\")[-4])
            data_final["region"].append(file.split("\\")[-3])
            data_final["subregion"].append(file.split("\\")[-2])
        data_final["file"].append(file.split("\\")[-1])
        # plt.imshow(overlap_img)
        # plt.show()
        # print(overlap/np.size(overlap_img))
        # plt.imshow(img_copy)
        # plt.show()
        # plt.hist(g.ravel(), bins=256, range=(0.0, 256), fc='k', ec='k')
        # plt.show()
        # plt.hist(g_thresh.ravel(), bins=256, range=(0.0, 70), fc='k', ec='k')
        # plt.show()
    return data_final, cor_graphs


data_final = {"subject": [], "region": [], "subregion": [], "file": [], "overlap_relative": [], "overlap_fixed": [],
              "overlap_pixels":[], "correlation":[]}

if experiment_num == 2:
    folders = glob.glob(path + "/*/")
    for subject in folders:
        regions = glob.glob(subject + "/*/")
        for region in regions:
            subregions = glob.glob(region + "/*/")
            for subregion in subregions:
                files = glob.glob(subregion + "/*.tif")
                data_final, cor_graphs = calculate_overlap(files)
elif experiment_num == 1:
    files = glob.glob(path + "/*.tif")
    data_final, cor_graphs = calculate_overlap(files)
elif experiment_num == 3 or experiment_num in ["1x2", "1s"]:
    folders = glob.glob(path + "/*/")
    files = []
    for folder in folders:
        tfiles = glob.glob(folder + "*.tif")
        files.extend(tfiles)
    # files = glob.glob(path + "/*/")
    data_final, cor_graphs = calculate_overlap(files)

df = pd.DataFrame(data_final)
df.to_excel(path + "/AsprosinImageAnalysis_Experiment1s.xlsx")
# df = pd.DataFrame(cor_graphs)
# df.to_csv(path + "/AsprosinImageAnalysis_Experiment1_CorGraphs.csv")
