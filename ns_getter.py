
# assuming existing dataset.pkl file
from neurosynth import Dataset
from neurosynth import meta, decode, network
import numpy as np
import os
import tqdm

def get_images(term="", dataset=None, fts=0.01, many_fts=True, fdr=1, updir="working_data/ns_data"):
    print("\ndownloading images for "+term)
    # basi check
    if term == "":
        print("no term selected: nothing done")
        return
    if not dataset:
        print("loading dataset")
        dataset = Dataset.load("working_data/dataset.pkl")
    fts = [0.01, 0.03, 0.05, 0.08, 0.1, 0.15, 0.2] if many_fts == True else [fts]
    # find/create directory
    directory = os.path.join(updir, term)
    if not os.path.exists(directory):
        os.makedirs(directory)
    # report
    f = open(directory+"/"+term+" download report.txt", "a")
    f.write("\n\nterm: "+term)
    # get images
    for ft in fts:
        ids = dataset.get_studies(features=term, frequency_threshold=ft)
        print(str(len(ids))+" studies found for "+term+" with "+str(ft)+" freq threshold")
        f.write("\nft = "+str(ft)+" : studies = "+str(len(ids)))
        # save
        name = term+"_ft="+str(ft)
        img_dir = os.path.join(directory, name)
        ma = meta.MetaAnalysis(dataset, ids, q=fdr)
        ma.save_results(img_dir, name)
        print("downloaded "+name+" to "+os.path.join(os.getcwd(), img_dir))
    f.close()

def get_more_images():
    print("loading dataset")
    dataset = Dataset.load("working_data/dataset.pkl")
    # no borrar para acordarme cuales ya he bajado
    # get_images(term="working memory", dataset=dataset)
    # get_images(term="episodic memory", dataset=dataset)
    # get_images(term="recognition memory", dataset=dataset)
    # get_images(term="pleasant", dataset=dataset)
    # get_images(term="unpleasant", dataset=dataset)
    # get_images(term="seen", dataset=dataset)
    # get_images(term="vision", dataset=dataset)
    # get_images(term="pictures", dataset=dataset)
    # get_images(term="recognition", dataset=dataset)
    # get_images(term="encoding", dataset=dataset)
    # get_images(term="remembered", dataset=dataset)
    # get_images(term="subsequently", dataset=dataset)
    # get_images(term="subsequent memory", dataset=dataset)
    # get_images(term="negative emotional", dataset=dataset)
    # get_images(term="memory", dataset=dataset)
    # get_images(term="memory encoding", dataset=dataset)
    get_images(term="", dataset=dataset)


get_more_images()
