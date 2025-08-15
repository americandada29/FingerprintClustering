from ase.io import read
import numpy as np
import pickle
import numpy as np
from pymatgen.io.ase import AseAtomsAdaptor as AAA
from pymatgen.analysis.local_env import CrystalNN
import libfp


### If any atoms are closer than min_dist, return None. Otherwise, return the distances.
def print_nn_dists(struct, min_dist=3.0):
  cnn = CrystalNN()
  natoms = len(struct)
  dists = []
  for i in range(natoms):
    for j in range(i):
      dist = struct.get_distance(i, j)
      dists.append(dist)
  dists = np.array(dists)/0.529177
  passed = np.all(dists > min_dist)
  if passed:
    return dists
  else:
    return None



### Create fingerprint from data
def create_fingerprints(atom, positions, cell, natx):
    all_symbols = atom.get_chemical_symbols()
    unique_symbols = {}
    for a in all_symbols:
        unique_symbols[a] = 0
    unique_symbols = [a for a in unique_symbols]
    type_nums = [all_symbols.count(x) for x in unique_symbols]
    types = []
    for i in range(len(type_nums)):
        types += [i+1]*type_nums[i]
    types = np.array(types, int)
    znucl = np.array(list(set(atom.get_atomic_numbers())), int)

    lmax = 2
    cutoff = 4.0
    contract = False
    ntyp = len(set(types))

    fp1 = libfp.get_lfp((cell, positions, types, znucl), cutoff, log=False, orbital='s', natx=natx)

    return fp1



atoms = read("sim.traj", index=":")

final_atoms = []
final_fps = []
failcount = 0
passcount = 0
for i in range(len(atoms)):
    try:
      testatom = atoms[i].copy()
      dists = print_nn_dists(AAA.get_structure(testatom))
      testatom.wrap()
      fingerprint = create_fingerprints(testatom, testatom.positions, testatom.get_cell()[:], natx=100)
      final_fps.append(fingerprint)
      final_atoms.append(testatom)
      print(str(i) + " DONE")
      passcount += 1
      del testatom
    except:
        print(str(i) + " FAILED")
        failcount += 1


print("Passed: ", passcount, "Failed: ", failcount)
print("Passed (%) : ", passcount/len(atoms), "Failed (%) : ", failcount/len(atoms))


with open("atoms_fingerprints.pkl","wb") as f:
    pickle.dump([final_atoms, final_fps], f)

