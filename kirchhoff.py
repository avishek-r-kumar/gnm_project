#!/usr/bin/env python

from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
import dfi
import dfi.fasta_convert
import pandas as pd
import dfi.fastaseq 
from dfi.fastaseq import mapres





def calc_hessian(x,y,z,Verbose=False):
    """
    Calculate the hessian given the coordinates 
    
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
    return kirchhoff


# In[3]:

# Invert this matrix
def invert_kirchhoff(kirchhoff):
    """
    Invert matrix 
    """
    from scipy import linalg as LA
    U, w, Vt = LA.svd(kirchhoff,full_matrices=False)
    S = LA.diagsvd(w,len(w),len(w))
    np.allclose(kirchhoff,np.dot(U,np.dot(S,Vt)))
    tol = 1e-6
    singular = w < tol
    assert np.sum(singular) == 1.
    invw = 1/w
    invw[singular] = 0
    inv_kirchhoff = np.dot(np.dot(U,np.diag(invw)),Vt)
    return inv_kirchhoff


# In[4]:

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
    print('generated kirchhoff using evolutionary constraints')
    print('kirchhoff shape: ', kirchhoff.shape)
    
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


# In[5]:

if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("pdbfile", help="expt pdb file")
    parser.add_argument("pdbmodel",help="pdb model")
    parser.add_argument("evodinput",help="EvoD Output")
    parser.add_argument("uniprotID",help ="Uniprot")

    args = parser.parse_args()
    mdlpdbfile = args.pdbmodel
    exptpdbfile= args.pdbfile
    evoDfile= args.evodinput
    uniprotID=args.uniprotID


    print(mdlpdbfile, exptpdbfile, evoDfile, uniprotID)
    
    #mdlpdbfile='1CB0.pdbmdl'
    #exptpdbfile='1cb0.pdb'
    #evoDfile='1CB0_MI_DI.txt'
    #uniprotID='Q13126'



    ATOMS = dfi.pdbio.pdb_reader(mdlpdbfile,CAonly=True)
    x,y,z = dfi.getcoords(ATOMS)
    numres = len(x)
    mdlseq=[mapres[atom.res_name] for atom in ATOMS]
    kirchhoff = calc_hessian(x,y,z)
    inv_kirchhoff = invert_kirchhoff(kirchhoff)
    mdl_diag = np.array([inv_kirchhoff[i,i] for i in range(numres)])


    #Experimental PDB 
    expt_ATOM = dfi.pdbio.pdb_reader(exptpdbfile,CAonly=True)
    expt_betafactors = np.array([atom.temp_factor for atom in expt_ATOM])
    exptseq=[mapres[atom.res_name] for atom in expt_ATOM]


    #EVFOLD Contacts 
    str_seq=''.join(
        dfi.fastaseq.get_fastaseq(uniprotID).split('\n')[1:] )
    fastaseq=[s for s in str_seq]
    numseq = len(fastaseq)
    evodkirchhoff=_build_kirchhoff(evoDfile,numseq)
    inv_evodkirchhoff=invert_kirchhoff(evodkirchhoff)
    evo_diag = np.array([inv_evodkirchhoff[i,i] for i in range(numseq)])


    # # Need to align sequences properly


    print(len(fastaseq))
    print(len(mdlseq))
    print(len(exptseq))




    align={'fastaseq':''.join(fastaseq),
           'mdlseq':''.join(mdlseq),
           'exptseq':''.join(exptseq)}



    from clustalo import clustalo


    aligned=clustalo(align,seqtype=3)



    dft = pd.DataFrame()




    dft['EVfoldR'] = [s for s in aligned['fastaseq']]
    dft['mdlR'] = [s for s in aligned['mdlseq']]
    dft['exptR'] = [s for s in aligned['exptseq']]


    #align b-factors with the corresponding sequence 
    seqtype='fastaR'
    bfactype='fastaB'


    def align_bfac(seqtype,bfactype,ls_bfac,dft):
        """
    
        """
        align_bfac = []
        i=0
        for r in dft[seqtype]:
        #print(r)
            if i >= len(ls_bfac):
                break
            if r == '-':
                align_bfac.append(np.nan)
            else:
                align_bfac.append(ls_bfac[i])
                i+=1
        while len(align_bfac) < len(dft[seqtype]):
            align_bfac.append(np.nan)

        dft[bfactype] = align_bfac




    align_bfac('EVfoldR','EVfoldB',evo_diag,dft)
    align_bfac('mdlR','mdlB',mdl_diag,dft)
    align_bfac('exptR','exptB',expt_betafactors,dft)


    dft.to_csv('align.csv',index=False)



    dfx.to_csv('DI-bfactor.csv',index=False)

    

    dfx[[i for i in dfx.columns if '_B' in i]].dropna().corr()




