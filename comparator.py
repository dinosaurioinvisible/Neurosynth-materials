
import matplotlib.pyplot as plt
import nibabel as nib
import os
import numpy as np
import numpy.linalg as npl
import seaborn as sns
import random
from tqdm import tqdm

# version for previously masked data
class ImageComparison():

    def __init__(self, images=None\
    , ft="0.01"\
    , preplots=False, plots=False\
    , whole=False\
    , top=False, ratio_mm=5\
    , mse_rank=False\
    , manual=False\
    , txt=False, csv=False\
    , new_folder="comparisons"):

        # load images and get data
        self.get_images(images, ft, ratio_mm)
        # choose/create output folder and initialize reports (name, value):
        self.output(new_folder)
        # core preprocessing
        self.preprocess()
        # plots
        if preplots == True:
            self.plot_hist()
        self.plots = plots

        # compare whole
        if whole == True:
            self.compare_whole()
        # compare by top regions
        if top == True:
            self.compare_top_activations()
        # compare by rankings instead of z-scores
        if mse_rank == True:
            self.mse_ranking()
        # compare manual
        if manual == True:
            self.compare_manual()

        # visualization
        #self.show_1()
        #self.show_2()

        # reports
        if txt == True:
            self.write_report(mode="txt")
        if csv == True:
            self.write_report(mode="csv")

#################################################################
            # look for and check images
#################################################################

    def images_check(self, images):
        for img in images[:3]:
            while not os.path.isfile(img):
                print("there's no "+str(img)+" file in "+os.getcwd())
                img = raw_input("filename?: ")
        return images

    def manual_images(self):
        print("you need to select some images to load")
        basel_img = raw_input("basel image filename?: ")
        fwd_img = raw_input("forward inference (consistency) filename?: ")
        rev_img = raw_input("reverse inference (specificity) filename: ")
        topic = raw_input("topic: ")
        images = [basel_img, fwd_img, rev_img, topic]
        return images

    def get_images(self, images, ft, ratio_mm):
        if images == None:
            images = self.manual_images()
        self.images_check(images)
        self.basel_img = nib.load(images[0])
        self.nsf_img = nib.load(images[1])
        self.nsr_img = nib.load(images[2])
        self.topic = images[3]+"_ft="+str(ft)+"_r="+str(ratio_mm)
        # data
        self.basel_data = self.basel_img.get_data()
        self.nsf_data = self.nsf_img.get_data()
        self.nsr_data = self.nsr_img.get_data()
        # define ratio in voxels
        self.ratio_vx = int(ratio_mm/2)

#################################################################
                # write basic info and reports
#################################################################

    def output(self, path):
        # choose/create folder for saving data
        self.directory = os.path.join(path, self.topic)
        if not os.path.exists(self.directory):
           os.makedirs(self.directory)
        # create report (name, value) and tables objects:
        self.report = []
        self.report.append(("feature", "_".join(self.topic.split("_")[0:-2])))
        self.report.append(("freq threshold", self.topic.split("_")[-2].split("=")[1]))
        self.report.append(("ratio in milimeters", self.topic.split("_")[-1].split("=")[1]))
        self.tables = []

    def write_report(self, mode=None):
        if mode == "txt":
            f = open(self.directory+"/"+self.topic+" comparison report.txt", "w")
            for i in self.report:
                f.write("\n{}: {}".format(i[0],i[1]))
            f.write("\n")
            f.close()
        elif mode == "csv":
            e = open(self.directory+"/"+self.topic+" comparison report.csv", "w")
            for i in self.report:
                e.write("\n{};{};".format(i[0],i[1]))
            for i in self.tables:
                #e.write("\n\ntopic;rank;x;y;z;z-score")
                e.write("\n{};{};{};{};{};".format(i[0],i[1],i[2],i[3],i[4]))
            e.close()

######################################################################
                    # normalization
# standard normalization by (X-mean)/SD assuming normal distributions
######################################################################

    ''' reshape and remove nonnan, count zeros '''
    def dim(self, datas):
        nonnan_data = []
        for data in datas:
            reshape = data.reshape(-1, )
            nonnan = [i for i in reshape if not np.isnan(i)]
            brain_voxels = len(nonnan)
            nonzeros = np.count_nonzero(nonnan)
            nz_percentage = round((nonzeros*100/brain_voxels), 2)
            print("brain voxels: {}, nonzeros: {}, percentage: {}".format(brain_voxels, nonzeros, nz_percentage))
            nonnan_data.append(np.array(nonnan))
        return nonnan_data

    ''' normalize using the whole brain voxels '''
    def normalize(self, data):
        reshape = data.reshape(-1,)
        trunc = [i for i in reshape if not np.isnan(i)]
        mean = np.mean(trunc)
        std = np.std(trunc)
        return (data-mean)/std

    def preprocess(self):
        print("preprocessing...")
        # masking using basel data (previously masked)
        self.nsf_data = self.basel_data*0 + self.nsf_data
        self.nsr_data = self.basel_data*0 + self.nsr_data
        # print("before normalizing")
        self.basel_nonnan, self.nsf_nonnan, self.nsr_nonnan = self.dim([self.basel_data, self.nsf_data, self.nsr_data])
        self.basel_norm = self.normalize(self.basel_data)
        self.nsf_norm = self.normalize(self.nsf_data)
        self.nsr_norm = self.normalize(self.nsr_data)
        print("after normalizing")
        self.basel_norm_nonnan, self.nsf_norm_nonnan, self.nsr_norm_nonnan = self.dim([self.basel_norm, self.nsf_norm, self.nsr_norm])
        print('-> done')

#################################################################
            # visualization: previous plots
#################################################################

    ''' density and joint plots for non processed data '''
    def distpreplots(self):
        # overlapped non normalized
        plt.figure()
        sns.distplot(self.basel_nonnan, color="blue", label="basel")
        sns.distplot(self.nsf_nonnan, color="green", label="forward inference")
        sns.distplot(self.nsr_nonnan, color="red", label="reverse inference")
        plt.legend()
        plt.suptitle(self.topic+" distplots")
        plt.savefig(self.directory+"/"+self.topic+"_distplots_.png")
        # overlapped normalized
        plt.figure()
        sns.distplot(self.basel_norm_nonnan, color="blue", label="basel")
        sns.distplot(self.nsf_norm_nonnan, color="green", label="forward inference")
        sns.distplot(self.nsr_norm_nonnan, color="red", label="reverse inference")
        plt.legend()
        plt.suptitle(self.topic+" normalized distplots")
        plt.savefig(self.directory+"/"+self.topic+"_normalized distplots_.png")
        # subplots
        # fig, axes = plt.subplots(2,3, figsize=(12,12))
        # sns.distplot(self.basel_nonnan, color="blue", label="basel", ax=axes[0,0])
        # axes[0,0].legend()
        # sns.distplot(self.nsf_nonnan, color="green", label="forward inference", ax=axes[0,1])
        # axes[0,1].legend()
        # sns.distplot(self.nsr_nonnan, color="red", label="reverse inference", ax=axes[0,2])
        # axes[0,2].legend()
        # sns.distplot(self.basel_norm_nonnan, color="blue", label="basel normalized", ax=axes[1,0])
        # axes[1,0].legend()
        # sns.distplot(self.nsf_norm_nonnan, color="green", label="forward normalized", ax=axes[1,1])
        # axes[1,1].legend()
        # sns.distplot(self.nsr_norm_nonnan, color="red", label="reverse normalized", ax=axes[1,2])
        # axes[1,2].legend()
        # plt.suptitle(self.topic+" seaborn_distplots")
        # plt.savefig(self.directory+"/"+self.topic+"_distplots_.png")
    def jointpreplots(self):
        sns.jointplot(self.basel_norm_nonnan, self.nsf_norm_nonnan, kind='reg')
        plt.suptitle(self.topic+" basel fwd normalized correlation")
        plt.savefig(self.directory+"/"+self.topic+"_basel forward normalized correlation.png")

        sns.jointplot(self.basel_norm_nonnan, self.nsr_norm_nonnan, kind='reg')
        plt.suptitle(self.topic+" basel rev normalized correlation")
        plt.savefig(self.directory+"/"+self.topic+"_basel reverse normalized correlation.png")

    def plot_hist(self):
        print("\ncreating distribution plots...")
        self.distpreplots()
        print("creating correlation plots...")
        self.jointpreplots()
        print("-> done")

#################################################################
                    # Whole comparison
#################################################################

    ''' compute MSE between data'''
    def compare_whole_data(self, flat_basel_data, flat_ns_data, title=""):
        # whole MSE error
        mse = round(np.mean((flat_basel_data-flat_ns_data)**2), 2)
        # whole correlation
        corr = round(np.corrcoef(flat_basel_data, flat_ns_data)[0][1], 2)
        # report
        print("{} {} MSE {}".format(self.topic, title, mse))
        print("{} {} correlation {}".format(self.topic, title, corr))
        self.report.append((title+" whole brain MSE", mse))
        self.report.append((title+" whole brain correlation", corr))

    def compare_whole(self):
        print("\ncomparing whole brain data...")
        self.report.append(("\nWhole brain comparison", ""))
        self.compare_whole_data(self.basel_norm_nonnan, self.nsf_norm_nonnan, title="NS normalized forward")
        self.compare_whole_data(self.basel_norm_nonnan, self.nsr_norm_nonnan, title="NS normalized reverse")
        print("-> done")

#################################################################
                    # top activations
#################################################################

    ''' creates a sorted by z-score list of coordinates (and z-score absolute values) '''
    def get_sorted_coordinates(self, data):
        coordinates = []
        for x in range(len(data)):
            for y in range(len(data[x])):
                for z in range(len(data[x][y])):
                    if not np.isnan(data[x][y][z]):
                        # absolute value to identify the more significant z-scores
                        coordinates.append((abs(data[x][y][z]), [x,y,z], data[x][y][z]))
        coordinates = sorted(coordinates, reverse=True)
        # return the real (non absolut) z-score value
        sorted_coordinates = [(i[2], i[1]) for i in coordinates]
        return sorted_coordinates
    ''' find the peak non-adjacent (defined by ratio) activation voxels '''
    def get_top_locations(self, coordinates, ratio, title="", limit=100):
        top = []
        noc = []
        for i in tqdm(coordinates):
            if i[1] not in noc:
                top.append(i)
                for rx in range(i[1][0]-ratio, i[1][0]+ratio+1):
                    for ry in range(i[1][1]-ratio, i[1][1]+ratio+1):
                        for rz in range(i[1][2]-ratio, i[1][2]+ratio+1):
                            noc.append([rx,ry,rz])
            if len(top) == limit:
                break
        print(str(len(top))+" regions found")
        # tables from coordinates: (z-score, [x,y,z])
        self.tables.append(("\n"+title,"x","y","z","z-score"))
        self.tables += [(str(i+1),top[i][1][0],top[i][1][1],top[i][1][2],round(top[i][0],2)) for i in range(len(top))]
        return top
    ''' weighted average of distances and cosine similarities of the 3d coordinates '''
    def compute_distance(self, top1, top2, title):
        len_top = len(top2) if len(top2) < len(top1) else len(top1)
        weights = np.arange(1, 0, -0.01)[0:len_top]
        dist1 = np.array([i[1] for i in top1])
        dist2 = np.array([i[1] for i in top2])
        distances = np.array([np.linalg.norm(dist1[i] - dist2[i]) for i in range(len_top)])
        weighted_distance = round(sum(np.array(distances) * weights)/len(distances),2)
        cosims = []
        for loc1,loc2 in zip(dist1, dist2):
            ab = 0; ax = 0; bx = 0
            for ai,bi in zip(loc1,loc2):
                ab += ai*bi
                ax += ai*ai
                bx += bi*bi
            cosim = ab/(np.sqrt(ax)*np.sqrt(bx))
            cosims.append(cosim)
        weighted_cosim = round(sum(np.array(cosims) * weights)/len(cosims),2)
        print("{} {} weighted distance: {}".format(self.topic, title, weighted_distance))
        print("{} {} weighted cosine similarity: {}".format(self.topic, title, weighted_cosim))
        self.report.append((title+" weighted distance", weighted_distance))
        self.report.append((title+" weighted cosine similarity", weighted_cosim))
    ''' MSE and correlation for highest z-scores '''
    def compare_zscores(self, top1, data2, title):
        zs1 = np.array([i[0] for i in top1])
        zs2 = np.array([data2[i[1][0]][i[1][1]][i[1][2]] for i in top1])
        mse = round(((zs1-zs2)**2).mean(), 2)
        corr = round(np.corrcoef(zs1, zs2)[0,1], 2)
        if self.plots == True:
            sns.jointplot(zs1, zs2, kind="reg")
            plt.suptitle(self.topic+" "+title+" correlation")
            plt.savefig(self.directory+"/"+self.topic+"_"+title+" correlation.png")
        print("{} {} MSE: {}".format(self.topic, title, mse))
        print("{} {} correlation: {}".format(self.topic, title, corr))
        self.report.append((title+" top voxels MSE", mse))
        self.report.append((title+" top voxels correlation", corr))
    def compare_top_activations(self):
        print("\ncomparing by top active regions...")
        self.report.append(("\nPeak z-scores comparison", ""))
        # sort coordinates
        self.sorted_basel = self.get_sorted_coordinates(self.basel_norm)
        self.sorted_fwd = self.get_sorted_coordinates(self.nsf_norm)
        self.sorted_rev = self.get_sorted_coordinates(self.nsr_norm)
        # get top locations by z-scores
        self.top_basel = self.get_top_locations(self.sorted_basel, self.ratio_vx, "Top Basel z-scores")
        top_fwd = self.get_top_locations(self.sorted_fwd, self.ratio_vx, "Top NS forward z-scores")
        top_rev = self.get_top_locations(self.sorted_rev, self.ratio_vx, "Top NS reverse z-scores")
        # get weighted average distance
        self.compute_distance(self.top_basel, top_fwd, title="Top voxels Basel/FWD")
        self.compute_distance(self.top_basel, top_rev, title="Top voxels Basel/REV")
        # compare z-scores
        self.compare_zscores(self.top_basel, self.nsf_norm, title="basel_to_fwd")
        self.compare_zscores(self.top_basel, self.nsr_norm, title="basel_to_rev")
        self.compare_zscores(top_fwd, self.basel_norm, title="fwd_to_basel")
        self.compare_zscores(top_rev, self.basel_norm, title="rev_to_basel")
        print("-> done")

#################################################################
                    # MSE ranking
# en realidad mas que las ubicaciones no se que provecho sacarle a esto
#################################################################

    ''' rank and then apply mse '''
    def mse_ranking(self):
        print("\ncomparing by MSE rank")
        self.report.append(("\nMSE rank comparison", ""))
        # compute mean:
        mean_basel_nsf = np.mean(self.basel_nnn - self.nsr_nnn)
        mean_basel_nsr = np.mean(self.basel_nnn - self.nsr_nnn)
        # compute mse for each voxel
        mse_nsf = (self.basel_norm - self.nsf_norm)/mean_basel_nsf
        mse_nsr = (self.basel_norm - self.nsr_norm)/mean_basel_nsr
        # sort coordinates
        sorted_mse_nsf = self.get_sorted_coordinates(mse_nsf)
        sorted_mse_nsr = self.get_sorted_coordinates(mse_nsr)
        # inverse sorting (so less significant errors are first)
        sorted_mse_nsf = sorted(sorted_mse_nsf, reverse=False)
        sorted_mse_nsr = sorted(sorted_mse_nsr, reverse=False)
        # get top locations by rank
        top_rank_nsf = self.get_top_locations(sorted_mse_nsf, self.ratio_vx, "NS forward MSE ranking")
        top_rank_nsr = self.get_top_locations(sorted_mse_nsr, self.ratio_vx, "NS reverse MSE ranking")
        # # get MSE rank correlation
        # top_nsf = np.array([i[0] for i in top_rank_nsf])
        # top_nsr = np.array([i[0] for i in top_rank_nsr])
        # basel_nsf = np.array([data2[i[1][0]][i[1][1]][i[1][2]] for i in top_nsf])
        # basel_nsr = np.array([data2[i[1][0]][i[1][1]][i[1][2]] for i in top_nsr])
        # corr = round(np.corrcoef(top_nsf, basel_nsf)[0,1], 2)
        # corr = round(np.corrcoef(top_nsr, basel_nsf)[0,1], 2)
        print("-> done")

#################################################################
            # manual comparison
#################################################################

    def compare_coordinates(self, x, y, z):
        if x == "":
            y = int(y); z = int(z)
            basel = self.basel_norm[:,y,z]
            fwd = self.nsf_norm[:,y,z]
            rev = self.nsr_norm[:,y,z]
        elif y == "":
            x = int(x); z = int(z)
            basel = self.basel_norm[x,:,z]
            fwd = self.nsf_norm[x,:,z]
            rev = self.nsr_norm[x,:,z]
        elif z == "":
            x = int(x); y = int(y)
            basel = self.basel_norm[x,y,:]
            fwd = self.nsf_norm[x,y,:]
            rev = self.nsr_norm[x,y,:]
        else:
            basel = self.basel_norm[x,y,z]
            fwd = self.nsf_norm[x,y,z]
            rev = self.nsr_norm[x,y,z]
        print("\nbasel values:")
        print(basel)
        print("\nneurosynth forward values:")
        print(fwd)
        print("\nneurosynth reverse values:")
        print(rev)
        print("\n\n Basel - forward inference resulting slice:")
        print(basel-fwd)
        print("\n\n Basel - reverse inference resulting slice:")
        print(basel-rev)
    def compare_manual(self):
        manual = 0
        while manual != str(0):
            print("insert neurosynth coordinates ("+str(self.ns_shape)+")")
            x = raw_input("x: ")
            y = raw_input("y: ")
            z = raw_input("z: ")
            self.compare_coordinates(x,y,z)
            manual = raw_input("\ninsert [0] to exit, any other to do it again: ")

#################################################################
                    # auto run comparisons
#################################################################

def run_comparisons(ns_features=[], basel_filenames=[], override=False\
, new_folder="comparisons", title=""):
    print("\ngetting files")
    # define features and fts
    if len(ns_features) == 0 or len(basel_filenames) == 0:
        print("no file selected: nothing done")
        return
    ns_features = ns_features
    basel_filenames = basel_filenames
    fts = [0.01, 0.05, 0.1, 0.15]
    ratios = [4, 8, 12, 16, 20]
    # get images
    for ns_feature,basel_filename in zip(ns_features, basel_filenames):
        excel_data = []
        for ft in fts:
            # define urls
            basel_url = os.path.join("working_data/basel_data_in_ns", basel_filename)
            ns_work_path = "working_data/ns_data"
            ns_folder = ns_feature+"_ft="+str(ft)
            nsf_name = ns_feature+"_ft="+str(ft)+"_consistency_z.nii.gz"
            nsr_name = ns_feature+"_ft="+str(ft)+"_specificity_z.nii.gz"
            nsf_url = os.path.join(ns_work_path, ns_feature, ns_folder, nsf_name)
            nsr_url = os.path.join(ns_work_path, ns_feature, ns_folder, nsr_name)
            # basic check
            new_path = os.path.join(new_folder, ns_feature)
            check_path = os.path.join(new_path, ns_folder)
            if os.path.exists(check_path):
                print("directory already exists")
                if override == False:
                    print("override deactivated, skipping to the next one...")
                    break
            if not os.path.exists(basel_url):
                print("incorrect url for basel file")
                break
            if not os.path.exists(nsf_url):
                print("not nsf file in basel_data, trying mixed_ns_data")
                ns_work_path = "working_data/mixed_ns_data"
                nsf_url = os.path.join(ns_work_path, ns_feature, ns_folder, nsf_name)
                if not os.path.exists(nsf_url):
                    print("incorrect url for nsf file")
                    break
                print("ok")
            if not os.path.exists(nsr_url):
                print("not nsr file in basel_data, trying mixed_ns_data")
                ns_work_path = "working_data/mixed_ns_data"
                nsr_url = os.path.join(ns_work_path, ns_feature, ns_folder, nsr_name)
                if not os.path.exists(nsf_url):
                    print("incorrect url for nsr file")
                    break
                print("ok")
            images = [basel_url, nsf_url, nsr_url, ns_feature]
            for ratio in ratios:
                # run comparison
                print("\n\n############## start #################")
                print("\nfeature: {}, ft={}, ratio={}".format(ns_feature, ft, ratio))

                x = ImageComparison(images=images\
                , ft=ft\
                , preplots=True, plots=False\
                , whole=True\
                , top=True, ratio_mm=ratio\
                , mse_rank=False\
                , manual=False\
                , txt=True, csv=True\
                , new_folder=new_path)
                # tables
                e = open(new_path+"/"+str(ns_feature)+".csv", "a")
                for i in x.tables:
                    e.write("\n{};{};{};{};{};".format(i[0],i[1],i[2],i[3],i[4]))
                e.close()
                # excel report
                if len(excel_data) == 0:
                    for i in x.report:
                        excel_data.append([i[0], i[1]])
                else:
                    for ix,ir in zip(excel_data,x.report):
                        ix.append(ir[1])
            f = open(new_folder+"/general_report "+title+".csv", "a")
        for row in excel_data:
            for element in row:
                f.write(str(element)+";")
            f.write("\n")
        f.write("\n")
    f.close()


#################################################################
                        # main
#################################################################

# run_comparisons(ns_features = ["working memory"\
# , "episodic memory"\
# , "recognition memory"\
# , "pleasant"\
# , "unpleasant"]\
# , basel_filenames = ["working memory 2 back vs 0 back.nii"\
# , "recognition previously seen vs new pictures.nii"\
# , "encoding subsequently remembered vs not remembered in free recall.nii"\
# , "encoding positive vs neutral pictures.nii"\
# , "encoding negative vs neutral pictures.nii"])

# recognition memory
# run_comparisons(ns_features = ["recognition memory_seen_pictures"\
# , "recognition memory_recognition_seen_pictures"\
# , "recognition_seen_pictures"]\
# , basel_filenames = ["recognition previously seen vs new pictures.nii"\
# , "recognition previously seen vs new pictures.nii"\
# , "recognition previously seen vs new pictures.nii"]\
# , title = "mixed_recognition memory")

# pleasant
# run_comparisons(ns_features = ["encoding_pleasant_pictures"]\
# , basel_filenames = ["encoding positive vs neutral pictures.nii"])

# unpleasant
# run_comparisons(ns_features = ["encoding_unpleasant_pictures"\
# , "encoding_negative emotional_pictures"]\
# , basel_filenames = ["encoding negative vs neutral pictures.nii"\
# , "encoding negative vs neutral pictures.nii"]\
# , title = "mixed_unplesant")

# episodic memory
# run_comparisons(ns_features = ["encoding_episodic memory"\
# , "memory encoding_episodic memory"]\
# , basel_filenames = ["encoding subsequently remembered vs not remembered in free recall.nii"\
# , "encoding subsequently remembered vs not remembered in free recall.nii"]\
# , title="mixed_episodic memory")

# subsequent memory
# run_comparisons(ns_features = ["subsequent memory"\
# , "encoding_subsequent memory"\
# , "memory encoding_subsequent memory"]\
# , basel_filenames = ["encoding subsequently remembered vs not remembered in free recall.nii"\
# , "encoding subsequently remembered vs not remembered in free recall.nii"\
# , "encoding subsequently remembered vs not remembered in free recall.nii"]\
# , title = "mixed_subsequent memory")

# encoding - subsequently remembered
# run_comparisons(ns_features = ["memory encoding"\
# , "encoding_subsequently_remembered"\
# , "memory encoding_subsequently_remembered"]\
# , basel_filenames = ["encoding subsequently remembered vs not remembered in free recall.nii"\
# , "encoding subsequently remembered vs not remembered in free recall.nii"\
# , "encoding subsequently remembered vs not remembered in free recall.nii"]\
# , title = "mixed_memory encoding")


if __name__ == "__main__":
    main()
