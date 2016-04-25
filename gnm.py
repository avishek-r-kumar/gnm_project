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
./gnm.py 
```

"""

import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
import prody as prdy
import pandas as pd


pdbid = '5pnt' #Doesn't make sense
n = 158 #magic number

# ##Build kirchoff matrix using EVfold contacts
def build_kirchhoff(n):
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
    contact_pairs = open('./data/5pnt_MI_DI.txt', 'rU').readlines() #Get rid of this hard code
    evol_const = np.zeros((n,n))
    for line in contact_pairs:
        a = line.split()
        i = int(a[0])
        j = int(a[2])
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

build_kirchhoff(n) #this part of the code is called but the matrix isn't saved.

#bfactor from hessian 
cal = prdy.parsePDB(pdbid)
calphas = cal.select('calpha and chain A')
gnm1 = prdy.GNM('kirchhoff from ProDy')
gnm1.buildKirchhoff(calphas)
gnm1.calcModes()
sqflucts1 = prdy.calcSqFlucts(gnm1[:]) #bfactor from the hessian
np.savetxt('sqflucts_ProDy.txt',sqflucts1)
bfact1 = prdy.calcTempFactors(gnm1[:],calphas) # scaled with exp bfactor
np.savetxt('bfactor_ProDy.txt',bfact1)


#bfactor from experiment 
bfactexp = calphas.getBetas() # experimental bfactor from pdb
np.savetxt('bfactor_exp.txt',bfactexp)


# ##Calculate square fluctuations using evfold kirchoff 
# get square fluctuations using custom kirchoff matrix
#this is the evolutionary part 
kirchhoff = prdy.parseSparseMatrix('evfold_kirchhoff.txt',
                                  symmetric=True)
gnm3 = prdy.GNM('GNM for RASH_HUMAN (5p21)')
gnm3.setKirchhoff(kirchhoff)
gnm3.calcModes()
sqflucts = prdy.calcSqFlucts(gnm3[:])
np.savetxt('sqlflucts_evfold_test.txt',sqflucts)
bfact_evfold = prdy.calcTempFactors(gnm3[:],calphas)
np.savetxt('bfactor_evfold.txt',bfact_evfold)



# Calculate correlation coefficients 
correlation1 = np.corrcoef(bfact1,bfactexp) # ProDy w. Exp
d1 = correlation1.round(2)[0,1]
print 'correlation (ProDy vs. exp): ',d1

correlation8 = np.corrcoef(bfact_evfold,bfactexp) # EVfold w. Exp
d8 = correlation8.round(2)[0,1]
print 'correlation (EVfold vs. exp): ',d8

correlation9 = np.corrcoef(bfact_evfold,bfact1) # EVfold w. ProDy
d9 = correlation9.round(2)[0,1]
print 'correlation (EVfold vs. ProDy): ',d9


# Plot the b-factors 
sns.set_style('white')
sns.set_context("poster", font_scale=2.5, rc={"lines.linewidth": 2.25, "lines.markersize": 8 })
plt.plot(bfact1, color="orange", label='ProDy vs. Experiment Correlation: %0.2f' % d1)
plt.plot(bfact_evfold, color="blue", label='EVfold vs. Experiment Correlation: %0.2f' % d8)
plt.plot(bfactexp, color="black", label='Experiment')
plt.xlabel('Residue Index')
plt.ylabel('B-factor')
plt.xlim(-2.0,n)
plt.ylim(0,100)
plt.legend(loc="upper right",fontsize='large')
plt.legend(loc=1,prop={'size':16})
plt.tight_layout()
plt.savefig(pdbid+'-bfactors.png')

