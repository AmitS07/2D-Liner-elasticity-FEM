# -*- coding: utf-8 -*-
"""
Created on Sun Jun 15 10:57:24 2025
@author: Amit Singh, The University of Alabama, USA
Description: Linear Elastic deformation of 2D sheet under tensile loading, 
            FEM code based on Textbook "First Course on FEM" by Jacob Fish and Ted Belytschko, Chapter:9
"""
#%% Importing required Libaries/FUnction
import numpy as np
import pandas as pd
from fem_functions_lib import *
import matplotlib.pyplot as plt
#%% Section-1 Materials Parameters
## Elastic properties of Aluminium alloys
E = 75e9 
nu = 0.3
#% 1.2: D matrix
D = (E/(1-nu**2))*np.array([[1, nu, 0],[nu, 1, 0],[0, 0, (1-nu)/2]])
#%% 2: Geometry of components
## 2.1 Elements i.e. Connectivity matrix
elements = pd.read_excel(f"D:/Data_E_Drive/University_of_Alabama/Courses_at_UA/ME591_LFEM/FEM_Ted_solution/linear_elastic_amit/elements20.xlsx", header =None)
elements -= 1 # This is to make sure elements index from 0  not from 1
## 2.2 Node position
nodecords = pd.read_excel(f"D:/Data_E_Drive/University_of_Alabama/Courses_at_UA/ME591_LFEM/FEM_Ted_solution/linear_elastic_amit/nodal_coordinates.xlsx", header =None)
nel = elements.shape[0] # Number of elements
nnd = nodecords.shape[0] # number of nodes
#%% Elements coordinates and mesh plot
elem_cord = []
for ii in range(nel):
    elm = elements.loc[ii]
    # print(ii)
    node1 = nodecords.loc[elm[0]]
    node2 =  nodecords.loc[elm[1]]
    node3 =  nodecords.loc[elm[2]]
    node4 =  nodecords.loc[elm[3]]
    elem_cord.append(np.vstack([node1, node2, node3, node4])) # just to save element cord
#     elm_cord = np.vstack([node1, node2, node3, node4, node1])
#     plt.plot(elm_cord[:,0],elm_cord[:,1], color = "k", linestyle = "-")
# plt.axis("equal")
# plt.xlim(-0.5, 1.5)
# plt.ylim(-0.1, 2.5)
# plt.grid(True)
# plt.close()

#%% Guass Quadrature point and weights for Quadrilateral elements
ngp = 2 # Number of Guass-Qadrature point
qp = np.array([-1/np.sqrt(3), 1/np.sqrt(3)]) # Quadrature Points
W = np.array([1,1]) # Weights of QP
#%%
## Now Solving for each element
nne = 4 # Number of Nodes per elements
ndf = 2 # Number of Degree of freedom per node
#%% System of equations for entire geometry i.e Kd = f
neq = nnd*ndf # Number of equations
## Initialization for K, d and r
K = np.zeros([neq, neq]) # Node wise stiffness matrix
U = np.zeros([neq]) # Node wise displacemnet vector
F = np.zeros([neq]) # Node wise force vector
## Intializing the Boundary conditions
ebc = np.zeros([neq]) # Essential i.e dispalcement BC
nbc = np.zeros([neq]) # Natural i.e stress/load BC
## Body force
b = np.zeros([nne*ndf, nel]) # element wise, body force matrix, most of the time zero
#%%  Prescribing Force and Displacements
##-------------- Prescribing Natural BC -----------------------------
nodes_nbc = np.array([26,27,28,29,30])-1 # Defing the nodes at which force is being applied
nbc_df = prescribe_nodal_bc(nodes_nbc,ndf) #nbc_df = nodes for natural BC's degree of freedom (x,y) i.e. 2 = 5*2
# Applied force prescribed nodes
load = 700e6
nbc_f = np.array([0,load ,0,load ,0,load ,0,load ,0,load ])
##-------------- Prescribing Essential BC -----------------------------
nodes_ebc = np.array([1,2,3,4,5])-1 # Defining nodes at which displacemnt is prescribed
ebc_df = prescribe_nodal_bc(nodes_ebc,ndf) # ebc_df = nodes for essential BC's degree of freedom (x,y) i.e.2 = 5*2
# Applied displacement at the perscribed nodes
ebc_d = np.array([0,0,0,0,0,0,0,0,0,0])

#%% solving for ke and fe for each element
for el in range(nel):
    cord_e = elem_cord[el] # elemental coordinates, element wise
    bf_e = b[:,el] # body force elemnet wise
    Ke, fe = elemental_stiffness(nne, ndf,ngp, cord_e, bf_e, qp,W,D)
    elm_node = elements.loc[el,:]
    index = gather_index(nne, ndf, elm_node)
    K, F = assembly(K,F, Ke,fe, index)
#%%
F[nbc_df] = nbc_f
K,F  = constraint(K, F, ebc_df, ebc_d)
U = np.linalg.solve(K, F)
#%%
Ux = [U[ii] for ii in range(0,neq,2)]
Uy = [U[ii] for ii in range(1,neq,2)]
U_cord = np.array([Ux, Uy]).T
# node_cord = nodecords.to_numpy()
#%%
disp_elm = []
x_cord_el,y_cord_el = [],[]
def_cord = []
for ii in range(nel):
    elm = elements.loc[ii]
    # print(ii)
    node1 = nodecords.loc[elm[0]]
    node2 =  nodecords.loc[elm[1]]
    node3 =  nodecords.loc[elm[2]]
    node4 =  nodecords.loc[elm[3]]
    elm_cord = np.vstack([node1, node2, node3, node4, node1])
    x_cord_el.append(elm_cord[:,0])
    y_cord_el.append(elm_cord[:,1])
    plt.plot(elm_cord[:,0],elm_cord[:,1], color = "k", linestyle = "-")
    ##
    dis_node1 = U_cord[elm[0]]
    dis_node2 = U_cord[elm[1]]
    dis_node3 = U_cord[elm[2]]
    dis_node4 = U_cord[elm[3]]
    disp_elm.append(np.vstack([dis_node1, dis_node2, dis_node3, dis_node4]))
    dis_cord = np.vstack([dis_node1, dis_node2, dis_node3, dis_node4, dis_node1])
    ## add 
    u_cord = elm_cord+dis_cord
    def_cord.append(u_cord)
    plt.plot(u_cord[:,0],u_cord[:,1], color = "r", linestyle = "--")
    ##
    # plt.fill([0,0,1,1],[-0.1,0,-0.1,0],color = "k")
plt.axis("equal")
# plt.xlim(-0.5, 1.5)
# plt.ylim(-0.2, 2.5)
plt.grid(True)
#%% Now doing stress and strain analysis 
strain_el, stress_el = [], []
for el in range(nel):
    ## Strain
    cord_el = elem_cord[el]
    B_el = B_elemental(ngp, cord_el, qp)
    disp_vec_el = matrix_to_vector(disp_elm[el])
    epsilon_el = np.matmul(B_el, disp_vec_el)
    strain_el.append(epsilon_el)
    ## Stress
    sigma_el = np.matmul(D, epsilon_el)
    stress_el.append(sigma_el)
#%% Plooting stress and strain
# for ii in range(nel):
#     ## Cordinates
#     elm = elements.loc[ii]
#     # print(ii)
#     node1 = nodecords.loc[elm[0]]
#     node2 =  nodecords.loc[elm[1]]
#     node3 =  nodecords.loc[elm[2]]
#     node4 =  nodecords.loc[elm[3]]
#     elm_cord = np.vstack([node1, node2, node3, node4, node1])
#     x_cord_el.append(elm_cord[:,0])
#     y_cord_el.append(elm_cord[:,1])
#     ## stress
#     s = stress_el[ii][1]
    
#%%import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection

# Select the stress component you want to visualize
# For example: σ_xx = stress_el[el][0], σ_yy = stress_el[el][1], τ_xy = stress_el[el][2]
stress_component = [stress_el[el][1] for el in range(nel)]  # σ_xx

fig, ax = plt.subplots()
patches = []
color_vals = []

for el in range(nel):
    elm_nodes = elements.loc[el].values.astype(int)

    # Get nodal coordinates for this element
    coords = np.vstack([nodecords.loc[ni].values for ni in elm_nodes])
    coords = np.vstack([coords, coords[0]])  # Close the loop for polygon

    # Create patch
    polygon = Polygon(coords, closed=True)
    patches.append(polygon)
    color_vals.append(stress_component[el])  # Constant value per element

# Add all patches as a collection
collection = PatchCollection(patches, cmap='viridis', edgecolor='k', linewidth=0.5)
collection.set_array(np.array(color_vals))
collection.set_clim(min(color_vals), max(color_vals))  # Optional: fix color range

ax.add_collection(collection)
fig.colorbar(collection, ax=ax, label='Stress $\sigma_{yy}$ (Pa)')
ax.set_aspect('equal')
ax.set_xlabel('X')
ax.set_ylabel('Y')
plt.grid(True)
plt.tight_layout()
plt.show()
