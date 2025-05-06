#####################################################
import autograd.numpy as np
# This file is created automatically
def Model(Y,t,pars):
    # Parameters
    a_g1_g5 = pars[0]
    a_g2_g1 = pars[1]
    a_g3_g2 = pars[2]
    a_g4_g4 = pars[3]
    a_g5_g4 = pars[4]
    a_g6_g1 = pars[5]
    alpha_g1 = pars[6]
    alpha_g2 = pars[7]
    alpha_g3 = pars[8]
    alpha_g4 = pars[9]
    alpha_g5 = pars[10]
    alpha_g6 = pars[11]
    k_g1 = pars[12]
    k_g2 = pars[13]
    k_g3 = pars[14]
    k_g4 = pars[15]
    k_g5 = pars[16]
    k_g6 = pars[17]
    l_p_g1 = pars[18]
    l_p_g2 = pars[19]
    l_p_g3 = pars[20]
    l_p_g4 = pars[21]
    l_p_g5 = pars[22]
    l_p_g6 = pars[23]
    l_x_g1 = pars[24]
    l_x_g2 = pars[25]
    l_x_g3 = pars[26]
    l_x_g4 = pars[27]
    l_x_g5 = pars[28]
    l_x_g6 = pars[29]
    m_g1 = pars[30]
    m_g2 = pars[31]
    m_g3 = pars[32]
    m_g4 = pars[33]
    m_g5 = pars[34]
    m_g6 = pars[35]
    n_g1 = pars[36]
    n_g2 = pars[37]
    n_g3 = pars[38]
    n_g4 = pars[39]
    n_g5 = pars[40]
    n_g6 = pars[41]
    r_g1 = pars[42]
    r_g2 = pars[43]
    r_g3 = pars[44]
    r_g4 = pars[45]
    r_g5 = pars[46]
    r_g6 = pars[47]
    sigmaH_g1 = pars[48]
    sigmaH_g2 = pars[49]
    sigmaH_g3 = pars[50]
    sigmaH_g4 = pars[51]
    sigmaH_g5 = pars[52]
    sigmaH_g6 = pars[53]
    # Variables
    x_g1 = Y[0]
    p_g1 = Y[1]
    x_g2 = Y[2]
    p_g2 = Y[3]
    x_g3 = Y[4]
    p_g3 = Y[5]
    x_g4 = Y[6]
    p_g4 = Y[7]
    x_g5 = Y[8]
    p_g5 = Y[9]
    x_g6 = Y[10]
    p_g6 = Y[11]
    dx_g1 = m_g1*(( alpha_g1 + a_g1_g5*(p_g5/k_g5)**n_g5 )/( 1 +(p_g5/k_g5)**n_g5 ))-l_x_g1*x_g1
    dp_g1 = r_g1*x_g1- l_p_g1*p_g1
    dx_g2 = m_g2*(( alpha_g2 + a_g2_g1*(p_g1/k_g1)**n_g1 )/( 1 +(p_g1/k_g1)**n_g1 ))-l_x_g2*x_g2
    dp_g2 = r_g2*x_g2- l_p_g2*p_g2
    dx_g3 = m_g3*(( alpha_g3 + a_g3_g2*(p_g2/k_g2)**n_g2 )/( 1 +(p_g2/k_g2)**n_g2 ))-l_x_g3*x_g3
    dp_g3 = r_g3*x_g3- l_p_g3*p_g3
    dx_g4 = m_g4*(( alpha_g4 + a_g4_g4*(p_g4/k_g4)**n_g4 )/( 1 +(p_g4/k_g4)**n_g4 ))-l_x_g4*x_g4
    dp_g4 = r_g4*x_g4- l_p_g4*p_g4
    dx_g5 = m_g5*(( alpha_g5 + a_g5_g4*(p_g4/k_g4)**n_g4 )/( 1 +(p_g4/k_g4)**n_g4 ))-l_x_g5*x_g5
    dp_g5 = r_g5*x_g5- l_p_g5*p_g5
    dx_g6 = m_g6*(( alpha_g6 + a_g6_g1*(p_g1/k_g1)**n_g1 )/( 1 +(p_g1/k_g1)**n_g1 ))-l_x_g6*x_g6
    dp_g6 = r_g6*x_g6- l_p_g6*p_g6
    dY = np.array([dx_g1,dp_g1,dx_g2,dp_g2,dx_g3,dp_g3,dx_g4,dp_g4,dx_g5,dp_g5,dx_g6,dp_g6,])
    return(dY)
#####################################################