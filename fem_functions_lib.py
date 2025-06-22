# -*- coding: utf-8 -*-
"""
Created on Sat Jun 21 18:34:07 2025
@author: Amit Singh, The University of Alabama, USA
Description: This contains the functions related to 2D linear elastic FEM of plate
"""
#%% Importing required Libaries/FUnction
import numpy as np
import pandas as pd
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

def elemental_stiffness(nne, ndf,ngp, cord_e,bf_e,qp,W,D):
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