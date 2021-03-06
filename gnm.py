#!/usr/bin/env python
"""
GNM
===

Description
-----------
GNM codes that does something.
Compares the bfactors of Hessian, PDB, and
EvFold. https://github.com/avishek-r-kumar/gnm_project.git

Usage
-----
```
./gnm.py pdbid evfold 
```

"""

import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
import prody as prdy
import pandas as pd
import sys 

import dfi.pdbio as io 


# ##Build kirchoff matrix using EVfold contacts
def _build_kirchhoff(evod_file,n):
    """
    Creates a kirchoff matrix using EVfold contacts

    Input
    -----
    evfold input file: str
       file from evfold
    n: size of the square matrix

    Output
    ------
    kirchoff: NxN numpy matrix
       output matrix
    """

    chain = []
    chain_connection = np.zeros((n,n))
    
    #assign a -1 for residues in contact in the chain
    for i in range(2, n-2):
        chain_connection[i, i+1] = -1
        chain_connection[i, i+2] = -1
        #chain_connection[i, i+3] = -1
        chain_connection[i+1, i] = -1
        chain_connection[i+2, i] = -1
        #chain_connection[i+3, i] = -1
        chain_connection[i, i-1] = -1
        chain_connection[i, i-2] = -1
        #chain_connection[i, i-3] = -1
        chain_connection[i-1, i] = -1
        chain_connection[i-2, i] = -1
        #chain_connection[i-3, i] = -1
        
        chain.append([i, i+1, chain_connection[i, i+1]])
        chain.append([i, i+2, chain_connection[i, i+2]])
        #chain.append([i, i+3, chain_connection[i, i+3]])
        chain.append([i+1, i, chain_connection[i+1, i]])
        chain.append([i+2, i, chain_connection[i+2, i]])
        #chain.append([i+3, i, chain_connection[i+3, i]])
        chain.append([i, i-1, chain_connection[i, i-1]])
        chain.append([i, i-2, chain_connection[i, i-2]])
        #chain.append([i, i-3, chain_connection[i, i-3]])
        chain.append([i-1, i, chain_connection[i-1, i]])
        chain.append([i-2, i, chain_connection[i-2, i]])
        #chain.append([i-3, i, chain_connection[i-3, i]])
        
    #assign a -1 for EC pairs
    evol = []
    contact_pairs = open(evod_file, 'rU').readlines() 
    evol_const = np.zeros((n,n))
    for line in contact_pairs:
        a = line.split()
        i = int(a[0]) - 1 
        j = int(a[2]) - 1 
        if (chain_connection[i, j] != -1):
            evol_const[i, j] = -1.0*float(a[5])
            evol_const[j, i] = -1.0*float(a[5])
            evol.append([i, j, evol_const[i, j]])
            evol.append([j, i, evol_const[j, i]])
    
    #build kirchoff matrix
    kirchhoff = np.zeros((n,n))
    kirchhoff = chain_connection + evol_const
    print 'generated kirchhoff using evolutionary constraints'
    print 'kirchhoff shape: ', kirchhoff.shape
    
    #calculate the diagonal
    diag = []
    for i in range(0, n):
        kirchhoff[i, i] = -np.sum(kirchhoff[i])
        diag.append([i, i, kirchhoff[i, i]])
    
    #put everything together for a file
    all = chain + evol + diag
    f = open('evfold_kirchhoff.txt', 'w')
    for x in all:
        f.write('%s \t %s \t %s \n' % (x[0], x[1], x[2]))
    f.close()
    
    return kirchhoff;


def calc_bfactors_from_alphaCAs(pdbid):
    """
    Calculate b-factors from the alpha CA network 
    
    Input
    -----
    pdbid: fname or pdbID
       PDB file or pdbID 

    Output
    ------
    bfact_alphaCA: numpy 
       bfactors calculated from the alpha carbon network 
    """
    calphas = prdy.parsePDB(pdbid).select('calpha and chain A')
    gnm1 = prdy.GNM()
    gnm1.buildKirchhoff(calphas)
    gnm1.calcModes()
    return prdy.calcTempFactors(gnm1[:],calphas) 

def calc_bfactors_from_pdb(pdbid):
    """
    Pull out b-facotrs from the PDB file 
    
    Input
    -----
    pdbid: fname or pdbID
       PDB file or pdbID 

    Output
    ------
    bfact_exp: numpy 
       bfactors calculated from the alpha carbon network 
    """
    calphas = prdy.parsePDB(pdbid).select('calpha and chain A')
    return calphas.getBetas() # experimental bfactor from pdb


def calc_bfactors_from_evoD(pdbid,evod_fname,nres):
    """
    Calculate b-factors from evoD 
    
    Input
    -----
    pdbid: fname or pdbID
       PDB file or pdbID 

    Output
    ------
    bfact_evfold: numpy 
       bfactors calculated from the alpha carbon network 
    """
    calphas = prdy.parsePDB(pdbid).select('calpha and chain A')
    n = nres  
    _build_kirchhoff(evod_fname,n) 
    
    kirchhoff = prdy.parseSparseMatrix('evfold_kirchhoff.txt',
                                  symmetric=True)
    gnm3 = prdy.GNM('GNM for RASH_HUMAN (5p21)')
    gnm3.setKirchhoff(kirchhoff)
    gnm3.calcModes()
    return prdy.calcTempFactors(gnm3[:],calphas)



def calc_bfactors(pdbid,evod_fname,nres):
    """
    Calculate b-factors from pdb, kirchoff and evod

    Input
    -----
    pdbid: file
       pdb file 
    evod_fname: file
       file containing evoD info 

    Output
    -----
    bfact_alphaCA, bfact_exp, bfact_evfold: numpy 
       numpy array of bfactors 
    pdbid.csv: file
       Output file of the bfactor results 
    """

    bfact_alphaCA = calc_bfactors_from_alphaCAs(pdbid)
    bfact_exp = calc_bfactors_from_pdb(pdbid)
    bfact_evfold = calc_bfactors_from_evoD(pdbid,evod_fname,nres)

    return bfact_alphaCA, bfact_exp, bfact_evfold 

def calc_hessian(x,y,z):
    """
    Calculate the hessian given the coordinates 
    Inspired by ProDy
    Input
    -----
    (x,y,z) numpy array
       Must all be the same length
    Output
    ------
    kirchhoff: NxN numpy matrix
       
    """
    cutoff = 10
    gamma = 1
    xyz = np.column_stack((x,y,z))
    numres = xyz.shape[0]
    kirchhoff = np.zeros((numres,numres))
    for i in range(numres):
        print i
        xyz_i = xyz[i]
        i_p1 = i + 1
        xyz_ij = xyz[i_p1:] - xyz_i
        xyz_ij2=np.multiply(xyz_ij,xyz_ij)
        cutoff2 = cutoff * cutoff
        for j, dist2 in enumerate(xyz_ij2.sum(1)):
            if dist2 > cutoff2:
                continue
            if Verbose:
                print(j, dist2)
            j += i_p1
            kirchhoff[i,j] = -gamma 
            kirchhoff[j,i] = -gamma 
            kirchhoff[i,i] += gamma 
            kirchhoff[j,j] += gamma 

def _writeCSV(pdbid,**kwargs):
    df = pd.DataFrame().from_dict(kwargs)
    df.to_csv(pdbid+'.csv',index=False)

if __name__ == "__main__":


    if len(sys.argv) < 3:
        print __doc__
        exit(1)
    
    pdb_fname = sys.argv[1]
    evod_fname = sys.argv[2]
    pdbid = pdb_fname.split('/')[-1].split('.')[0]
    ATOMS = io.pdb_reader(pdb_fname,CAonly=True,chainA=True,
                          chain_name='A')
    #nres = len(ATOMS)
    nres = 498
    res_ind = [atom.res_index for atom in ATOMS]
    seq = [atom.res_name for atom in ATOMS]
    
    bfact_alphaCA, bfact_exp,bfact_evfold = calc_bfactors(
        pdb_fname,evod_fname,nres)
    _writeCSV(pdbid,
              ResI = res_ind,
              Res = seq,
              bfact_alphaCA=bfact_alphaCA,
              bfact_exp=bfact_exp,
              bfact_evfold=bfact_evfold)

    
    # Calculate correlation coefficients 
    correlation1 = np.corrcoef(bfact_alphaCA,bfact_exp) # ProDy w. Exp
    d1 = correlation1.round(2)[0,1]
    print 'correlation (ProDy vs. exp): ',d1

    correlation8 = np.corrcoef(bfact_evfold,bfact_exp) # EVfold w. Exp
    d8 = correlation8.round(2)[0,1]
    print 'correlation (EVfold vs. exp): ',d8

    correlation9 = np.corrcoef(bfact_evfold,bfact_alphaCA) # EVfold w. ProDy
    d9 = correlation9.round(2)[0,1]
    print 'correlation (EVfold vs. ProDy): ',d9


    # Plot the b-factors 
    sns.set_style('white')
    sns.set_context("poster", font_scale=2.5, rc={"lines.linewidth": 2.25, "lines.markersize": 8 })
    plt.plot(bfact_alphaCA, color="orange", label='ProDy vs. Experiment Correlation: %0.2f' % d1)
    plt.plot(bfact_evfold, color="blue", label='EVfold vs. Experiment Correlation: %0.2f' % d8)
    plt.plot(bfact_exp, color="black", label='Experiment')
    plt.xlabel('Residue Index')
    plt.ylabel('B-factor')
    plt.xlim(-2.0,nres)
    plt.ylim(0,100)
    plt.legend(loc="upper right",fontsize='large')
    plt.legend(loc=1,prop={'size':16})
    plt.tight_layout()
    plt.savefig(pdbid+'-bfactors.png')

