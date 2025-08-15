import pickle
import numpy as np
#from hdbscan import HDBSCAN
from scipy.optimize import linear_sum_assignment
import numba
import time
#from fp_dist_calc import fp_dist_calc
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from ase.build.tools import sort
from ase.neighborlist import NeighborList, natural_cutoffs
from ase.visualize import view
import libfp
#import fplib3_pure
from tqdm import tqdm
from ase.io import write
from sklearn_extra.cluster import KMedoids


def create_fp_dist_matrix_cpp_with_types(fps, types):
    N = len(fps)
    dist_matrix = np.zeros((N,N))
    shift = np.sqrt(fps.shape[2])
    for i in tqdm(range(N), desc="Building distance matrix"):
        for j in range(i+1):
            dist_matrix[i,j] = libfp.get_fp_dist(fps[i], fps[j], types)*shift
            dist_matrix[j,i] = dist_matrix[i,j]
    return dist_matrix

### Create the distance matrix needed for the clustering algorithm and save it to a file###
def create_and_save_fp_dist_matrix(atoms, fingerprints):
    all_symbols = atoms[0].get_chemical_symbols()
    unique_symbols = {}
    for a in all_symbols:
        unique_symbols[a] = 0
    unique_symbols = [a for a in unique_symbols]
    type_nums = [all_symbols.count(x) for x in unique_symbols]
    typt = type_nums
    nat = sum(typt)
    #pos = positions
    types = []
    for i in range(len(typt)):
        types += [i+1]*typt[i]
    types = np.array(types, int)


    ################ Actually create the fingerprint distance matrix with types considered #########
    st = time.time()
    dist_matrix = create_fp_dist_matrix_cpp_with_types(fingerprints, types)
    print("Distance matrix computation took {} seconds".format(np.around(time.time()-st, 5)))
    
    with open("fp_dist_matrix.pkl", "wb") as f:
       pickle.dump(dist_matrix, f)
    #########################################################################


### Read distance matrix and cluster according to KMedoids (any clustering algorithm can be used) ###
def read_and_cluster_atoms(fname="fp_dist_matrix.pkl"):
    ############ Read distance matrix from pickle #####################
    with open(fname,"rb") as f:
        dist_matrix = pickle.load(f)
    dist_matrix = np.float64(dist_matrix)

    flattened_fps = fingerprints.reshape(fingerprints.shape[0], fingerprints.shape[1]*fingerprints.shape[2])

    pca = PCA(n_components=3)
    red_fps = pca.fit_transform(flattened_fps)
    print("Explained variance:", pca.explained_variance_ratio_)

    clusterer = KMedoids(n_clusters=200, metric="precomputed")
    clusterer.fit(dist_matrix)

    lclustdict = {}
    for i, lab in enumerate(clusterer.labels_):
        if lab in lclustdict:
            lclustdict[lab].append(i)
        else:
            lclustdict[lab] = [i]


    #### Plotting clustered points in 3D ###
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    for l in lclustdict:
        clust_nums = np.array(lclustdict[l])
        print(l, clust_nums)
        ax.scatter(red_fps[clust_nums, 0], red_fps[clust_nums,1], red_fps[clust_nums,2], c=np.random.rand(3))
    plt.show()
    #################################


    ### Randomly select out 200 atoms ###
    selected_atoms = []
    for l in lclustdict:
        ind_choice = np.random.choice(lclustdict[l])
        selected_atoms.append(atoms[ind_choice])

    write("clustered_atoms.extxyz", selected_atoms, format="extxyz")


with open("atoms_fingerprints.pkl", "rb") as f:
    atoms, fingerprints = pickle.load(f)
fingerprints = np.array(fingerprints)



create_and_save_fp_dist_matrix(atoms, fingerprints)
read_and_cluster_atoms()

