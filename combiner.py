
import numpy as np
import nibabel as nib
import os

def combine(features, fts=[0.01, 0.05, 0.1, 0.15]):

    if len(features) == 0:
        print("no features selected: nothing done")
        return
    work_path = "working_data/ns_data"
    for ft in fts:
        images = []
        mixed_features = ""
        for feature in features:
            nsf_file_path = os.path.join(work_path, feature, feature+"_ft="+str(ft), feature+"_ft="+str(ft)+"_consistency_z.nii.gz")
            nsr_file_path = os.path.join(work_path, feature, feature+"_ft="+str(ft), feature+"_ft="+str(ft)+"_specificity_z.nii.gz")
            nsf = nib.load(nsf_file_path)
            nsr = nib.load(nsr_file_path)
            mixed_features += feature+"_"
            images.append([nsf, nsr])

        mixed_features = mixed_features[:-1]
        mixed_features_ft = mixed_features+"_ft="+str(ft)
        nsf_data = images[0][0].get_data()*0
        nsr_data = images[0][1].get_data()*0
        for image in images:
            nsf_data += image[0].get_data()
            nsr_data += image[1].get_data()
        nsf_data = nsf_data/len(images)
        nsr_data = nsr_data/len(images)

        nsf_mixed = nib.Nifti1Image(nsf_data, images[0][0].affine)
        nsr_mixed = nib.Nifti1Image(nsr_data, images[0][1].affine)

        save_path = os.path.join("working_data", "mixed_ns_data", mixed_features, mixed_features_ft)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        nsf_save_path = os.path.join(save_path, mixed_features_ft+"_consistency_z.nii.gz")
        nsr_save_path = os.path.join(save_path, mixed_features_ft+"_specificity_z.nii.gz")

        nib.save(nsf_mixed, nsf_save_path)
        nib.save(nsr_mixed, nsr_save_path)

        print("{} saved to {}".format(mixed_features, save_path))
    print("done")

# # recognition memory
# combine(features=["recognition memory", "recognition", "seen", "pictures"])
# combine(features=["recognition memory", "seen", "pictures"])
# combine(features=["recognition", "seen", "pictures"])
# # pleasant
# combine(features=["encoding", "pleasant", "pictures"])
# # unplesant
# combine(features=["encoding", "unpleasant", "pictures"])
# combine(features=["encoding", "negative emotional", "pictures"])
# # episodic memory
# combine(features=["encoding", "subsequently", "remembered"])
# combine(features=["memory encoding", "subsequently", "remembered"])
# combine(features=["encoding", "subsequent memory"])
# combine(features=["memory encoding", "subsequent memory"])
# combine(features=["encoding", "episodic memory"])
# combine(features=["memory encoding", "episodic memory"])
combine(features="")
combine(features="")
combine(features="")
