
import nibabel as nib
import numpy as np
import numpy.linalg as npl
from tqdm import tqdm
import os

def convert(basel_filename, ns_mask_filename):
    # load image in get data
    basel_img = nib.load(basel_filename)
    ns_mask_img = nib.load(ns_mask_filename)
    basel_data = basel_img.get_data()
    mask_data = ns_mask_img.get_data()
    # bin of zeros of NS shape (bin for basel data)
    basel_data_in_ns = np.zeros(ns_mask_img.shape)
    # create a list of indices according to NS shape (NS indices for zip)
    x,y,z = ns_mask_img.shape
    points_ns = []
    for a in tqdm(range(x)):
        for b in range(y):
            for c in range(z):
                points_ns.append([a,b,c])
    # create a list of basel points in NS length (Basel indices for zip)
    neuro_vox2basel_vox = npl.inv(basel_img.affine).dot(ns_mask_img.affine)
    points_basel = np.rint([list(nib.affines.apply_affine(neuro_vox2basel_vox, point)) for point in tqdm(points_ns)]).astype(int)
    # for zipped basel and NS points, basel data in NS [xt,yt,zt] = basel data [x,y,z]
    for point_basel,point_ns in zip(tqdm(points_basel),points_ns):
        basel_data_in_ns[point_ns[0],point_ns[1],point_ns[2]] = basel_data[point_basel[0],point_basel[1],point_basel[2]]
    # adapt ns mask (replace zeros by nan, and nonzeros by zeros)
    ns_mask = []
    for x in tqdm(range(len(mask_data))):
        ns_mask.append([])
        for y in range(len(mask_data[x])):
            ns_mask[x].append([])
            for z in range(len(mask_data[x][y])):
                if mask_data[x][y][z] == 0:
                    ns_mask[x][y].append(np.nan)
                else:
                    ns_mask[x][y].append(0)
    ns_mask = np.array(ns_mask)
    # apply mask: remove non brain areas using NS mask
    basel_masked_data_in_ns = ns_mask + basel_data_in_ns
    # create .nii image and save
    basel_ns_masked_nii = nib.Nifti1Image(basel_masked_data_in_ns, ns_mask_img.affine)
    work_dir = "working_data/basel_data_in_ns"
    filename = basel_filename.split("/")[-1]
    save_path = os.path.join(work_dir, filename)
    if not os.path.exists(work_dir):
       os.makedirs(work_dir)
    nib.save(basel_ns_masked_nii, save_path)
    print("done: {}".format(filename))

# convert("working_data/basel_data/encoding negative vs neutral pictures.nii", "working_data/mask.nii.gz")
# convert("working_data/basel_data/encoding positive vs neutral pictures.nii", "working_data/mask.nii.gz")
# convert("working_data/basel_data/encoding subsequently remembered vs not remembered in free recall.nii", "working_data/mask.nii.gz")
# convert("working_data/basel_data/recognition previously seen vs new pictures.nii", "working_data/mask.nii.gz")
# convert("working_data/basel_data/working memory 2 back vs 0 back.nii", "working_data/mask.nii.gz")
