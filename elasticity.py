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
#%% Functions 
def N_matrix(xi,eta):
    ''' 
    Input:
    xi,eta = Cordinate of Quadrature Points
    Output: N matrix (Shape function matrix)
    '''
    # Shape functions for quadrilateral element in parent coordinate for QP
    N1 = (1/4)*(1-xi)*(1-eta)
    N2 = (1/4)*(1+xi)*(1-eta)
    N3 = (1/4)*(1+xi)*(1+eta)
    N4 = (1/4)*(1-xi)*(1+eta)
    # Shape function matrix 
    N = np.array([[N1, 0, N2, 0, N3, 0, N4, 0],
                 [0, N1, 0, N2, 0, N3, 0, N4]])
    return N

def B_matrix(xi,eta,elm_cord):
    ''' 
    Input:
    xi,eta = Cordinate of Quadrature Points
    elm_cord = Coordiante of elements (4X2, size for Quadrilateral emelents) 
    Output:detJ = Jacobain 
        B matrix (derivative of Shape-functions)
    '''
    # Derivative of shape function matrix (N) for given QP a 2x4 matrix
    dN = (1/4)*np.array([[eta-1, 1-eta, 1+eta, -eta-1],
                         [xi-1, -xi-1, 1+xi, 1-xi]])
    # Jacobian Maitrix and determinat
    J = np.matmul(dN,elm_cord) # 2x2 matrix
    detJ = np.linalg.det(J)
    # B matrix
    B_mat = np.matmul(np.linalg.inv(J), dN) # 2X4 matrix
    ## Finally B
    B = np.array([[ B_mat[0,0], 0, B_mat[0,1], 0, B_mat[0,2], 0, B_mat[0,3], 0],
                  [ 0, B_mat[1,0], 0, B_mat[1,1], 0, B_mat[1,2], 0, B_mat[1,3]],
                  [ B_mat[1,0], B_mat[0,0], B_mat[1,1], B_mat[0,1], B_mat[1,2], B_mat[0,2], B_mat[1,3], B_mat[0,3]]])
    return detJ, B
#%% Guass Quadrature point and weights for Quadrilateral elements
ngp = 2 # Number of Guass-Qadrature point
qp = np.array([-1/np.sqrt(3), 1/np.sqrt(3)]) # Quadrature Points
W = np.array([1,1]) # Weights of QP
#%%
## Now Solving for each element
nne = 4 # Number of Nodes per elements
ndf = 2 # Number of Degree of freedom per node
#%% Solving for per elements
# def elemental_stiffness_force(nne, ndf, nel,ngp, elem_cord, qp,W):
#     ## Defing the Size of Ke, Be, and Fe, element basis
#     Ke = np.zeros([nne*ndf, nne*ndf]) # Element stiffenss, 
#     fe = np.zeros([nne*ndf]) # element body force vector
#     b = np.zeros([nne*ndf, nel]) # element wise, body force matrix, most of the time zero
#     ## Interating over per element
#     for e in range(nel):
#         cord_e = elem_cord[e] ## element cordinates, 4x2
#         ##Looping Guass-quadarure points
#         for ii in range(ngp): # for parent xi-axis
#             xi = qp[ii]
#             w_xi = W[ii]
#             for jj in range(ngp): # for parent eta-axis
#                 eta = qp[jj]
#                 w_eta = W[jj]
#                 N = N_matrix(xi, eta) # Shape Function , 2x8 matrix
#                 detJ, B = B_matrix(xi,eta,cord_e) # Derivative of Shape function
#                 ## Solving for Ke
#                 Ke = Ke + w_xi*w_eta*np.matmul(B.T,np.matmul(D,B))*detJ
#                 ## elemental body force matrix 
#                 be = np.matmul(N, b[:,e])
#                 fe = fe + w_xi*w_eta*np.matmul(N.T, be)*detJ #np.matmul(N, be[:,e] = body force, interploating with shape function, though it is zero
#     return Ke, fe

def elemental_stiffness(nne, ndf,ngp, cord_e,bf_e,qp,W):
    ## Defing the Size of Ke, Be, and Fe, element basis
    Ke = np.zeros([nne*ndf, nne*ndf]) # Element stiffenss,
    fe = np.zeros([nne*ndf]) # element body force vector
    #     be = np.zeros([nne*ndf, nel]) # element wise, body force matrix, most of the time zero
    ## Interating over per element
    for ii in range(ngp): # for parent xi-axis
        xi = qp[ii]
        w_xi = W[ii]
        for jj in range(ngp): # for parent eta-axis
            eta = qp[jj]
            w_eta = W[jj]
            N = N_matrix(xi, eta) # Shape Function , 2x8 matrix
            detJ, B = B_matrix(xi,eta,cord_e) # Derivative of Shape function
            ## Solving for Ke
            Ke = Ke + w_xi*w_eta*np.matmul(B.T,np.matmul(D,B))*detJ
            ## elemental body force matrix 
            be = np.matmul(N, bf_e)
            fe = fe + w_xi*w_eta*np.matmul(N.T, be)*detJ
    return Ke, fe

def B_elemental(ngp, cord_e, qp):
    for ii in range(ngp): # for parent xi-axis
        xi = qp[ii]
        for jj in range(ngp): # for parent eta-axis
            eta = qp[jj]
            # N = N_matrix(xi, eta) # Shape Function , 2x8 matrix
            detJ, B = B_matrix(xi,eta,cord_e) # Derivative of Shape function
    return B
    
# ## Gather matrix function
# def gather_matrix(nel, nne, ndf,elements):
#     L_matrix  = np.zeros([nel, nne*ndf], dtype=int) 
#     for ii in range(nel):
#         elm_node = elements.loc[ii,:]
#         count = 0
#         for jj in range(nne):
#             start = (elm_node[jj])*ndf #start = (elm_node[jj]-1)*ndf
#             for kk in range(ndf):
#                 L_matrix[ii, count] = start + kk
#                 count += 1
#     return L_matrix

## Gather matrix function
def gather_index(nne, ndf,elm_node):
    index = np.zeros([nne * ndf], dtype=int)
    count = 0
    for jj in range(nne):
        start = elm_node[jj]*ndf #start = (elm_node[jj]-1)*ndf
        for kk in range(ndf):
            index[count] = start + kk
            count += 1
    return index

## Assembly
def assembly(K, F,ke, fe, index):
    for ii in range(len(index)):
        aa = index[ii]
        F[aa] = F[aa] + fe[ii] 
        for jj in range(len(index)):
            bb = index[jj]
            ## Stiffness Matrix
            K[aa,bb] = K[aa,bb] + ke[ii,jj]
    return K, F

## constrain 
def constraint(K, F, ebc_df, ebc_d):
    neq = K.shape[0]
    ebc_df = np.array(ebc_df, dtype=int)
    for i, ii in enumerate(ebc_df):
        for jj in range(neq):
            K[ii, jj] = 0 
        K[ii,ii] = 1 
        F[ii] = ebc_d[i]
    return K, F

## Prescribing Boundary conditions function
def prescribe_nodal_bc(nodes_pbc,ndf):
    pbc_df = np.zeros([len(nodes_pbc)*ndf],dtype=int) # nodal degree of freedom(x,y i.e. 2) for perscribed BC 
    count = 0
    for ii in nodes_pbc:
        pbc_df[count] = ndf*(ii) # ndf*(ii)-1
        pbc_df[count+1] = ndf*(ii)+1 #ndf*(ii)
        count +=2
    return pbc_df
## reshape 2D array to vector
def matrix_to_vector(array):
    c = array.shape[1]# Number of column
    r = array.shape[0] # Number of row
    vec = np.zeros([c*r])
    for ii in range(r):
        vec[ii] = array[ii][0]
        vec[ii+1] = array[ii][1]
    return vec
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
    Ke, fe = elemental_stiffness(nne, ndf,ngp, cord_e, bf_e, qp,W)
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
