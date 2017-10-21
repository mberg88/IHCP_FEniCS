# -*- coding: utf-8 -*-
'''
Created on Sun Apr 16 08:04:20 2017

@author: MattiaL
'''
import numpy as np
from math import atan2, degrees
from scipy.interpolate import NearestNDInterpolator
import matplotlib.pyplot as plt
import dolfin as do
from cyl_tr_adj_par_h_regul_CG_reinh_funs import mesh_import_fun, \
     solve_tr_dir__const_rheo, nablaT_fun, nablap_fun, nablau_fun, gamma_fun, \
     eta_fun, err_est_fun, conj_coef_fun, make_sure_path_exists
import os
import logging

# <codecell>    
if __name__ == '__main__': 
    '''
    Solves DHCP.
    Solves IHCP on a coarser mesh.
    Runs with
    DOLFIN_NOPLOT=1 python cyl_tr_adj_par_h_regul_CG_reinh.py > cyl_tr_adj_par_h_regul_CG_reinh.log
    '''
    
    #comm2 = MPI.COMM_WORLD
    
    #current proc
    #rank2 = comm2.Get_rank()
    
    bool_plot1 = 1
    savings_dol1 = os.path.join(os.getcwd(), 'pics_dolfin')
    savings_pic1 = os.path.join(os.getcwd(), 'pics_plt')
    make_sure_path_exists(savings_pic1)
    
    #mesh name
    mesh_name1 = 'reinh_circle_DHCP.xml.gz'
    
    #logging
    logger1 = logging.getLogger()
    
    #turns off 'FFC Reusing form from cache'
    logging.getLogger('UFL').setLevel(logging.WARNING)
    logging.getLogger('FFC').setLevel(logging.WARNING)
    
    try:
        #remove old logfile
        os.remove('{}__tr_dir_proc.txt'.format(mesh_name1.split('.')[0]))
    except OSError:
        #file does not exist
        pass
    
    #create file handler which logs warning messages
    fh1 = logging.FileHandler('{}__tr_dir_proc.txt'.format(mesh_name1.split('.')[0]))
    fh1.setLevel(logging.WARNING)
    
    #create formatter and add it to fh1
    formatter1 = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    fh1.setFormatter(formatter1)    
    
    #add h1 to logger1
    logger1.addHandler(fh1)
    
    #example    
    deg_choice1 = 1
    
    theta1 = .5
    gamma_reg2 = 1e-10 #5e-4 or 1e-10
    gamma_reg2_old = 0. #used in the 1st iteration, since the 1st T_step is regul.-free
    
    #example
    neighbor = False
    
    hol_cyl1 = mesh_import_fun(os.path.join(savings_dol1, mesh_name1))
    
    #K - T at time = 0 s
    T_old1 = 273.
    
    #m ** 2 / s
    alpha_mesh1 = 1.43e-7
    
    '''
    CHANGE IF 3D
    CHECK IF hol_cyl1.hmin() = hol_cyl2.hmin()
    '''   
    dt1 = hol_cyl1.hmin() ** 2. / 4. / alpha_mesh1
    
    print 'dt = {:.6f} s'.format(dt1)
    
    #time (s)
    time_v1 = np.linspace(0., 80., num = int((80. - 0.) / dt1 + 1))
    
    #example
    cut_end1 = int(.8 * len(time_v1))
    
    #K - T at the inner boundary
    T_in1_expr1 = 273.
    
    #K - T of the fluid at the outer boundary
    T_inf_expr1 = 313.
                               
    #316LN (T in K) -> k (W / m / K)                      
    k_mesh1 = .3
                            
    k_mesh_old1 = 1. * k_mesh1
    
    #316LN (T in K) -> rho (kg / m ** 3.)                      
    rho_mesh1 = k_mesh1 / alpha_mesh1
                               
    rho_mesh_old1 = 1. * rho_mesh1
    
    cp_mesh1 = 1.
    
    cp_mesh_old1 = 1.
    
    class MyExpression0(do.Expression):
        
        def eval(self, values, x):
            
            ang_rad = atan2(x[1], x[0])
            ang_deg = degrees(ang_rad)
            
            if abs(ang_deg) <= 40.:
                values[0] = 10.
            elif 40. <= abs(ang_deg) <= 90.:
                values[0] = + 4.8 * abs(ang_deg) - 182.
            elif 90. <= abs(ang_deg) <= 140.:
                values[0] = - 4.8 * abs(ang_deg) + 682.
            elif 140. <= abs(ang_deg) <= 180.:
                values[0] = 10.
                
    HTC1 = MyExpression0(degree = deg_choice1,
                         domain = hol_cyl1)
    
    #CHECK
    V1 = do.FunctionSpace(hol_cyl1, 'CG', deg_choice1)  
    HTC1_plot = do.Function(V1)
    HTC1_plot = do.project(HTC1, V1)
    do.plot(HTC1_plot)
    
    bool_solv1 = 1
          
    _, _, _, \
    _, _, _, \
    _, _, _, _, \
    R_in1, R_ex1, T_sol_d1, _, _, \
    _, dofs_x1 = solve_tr_dir__const_rheo(os.path.join(savings_dol1, mesh_name1), 
                                    hol_cyl1,
                                    deg_choice1, 
                                    T_in1_expr1, T_inf_expr1, HTC1,
                                    T_old1,
                                    k_mesh1,     cp_mesh1,     rho_mesh1, 
                                    k_mesh_old1, cp_mesh_old1, rho_mesh_old1, 
                                    dt1,
                                    time_v1,
                                    theta1,
                                    bool_plot1, bool_solv1, savings_dol1, logger1)
    
    print 'deg_choice = ', deg_choice1
    print 'R_ex1 = ', R_ex1
    
    #mesh name
    #coarser mesh
    mesh_name2 = 'reinh_circle_IHCP.xml.gz'
    hol_cyl2 = mesh_import_fun(os.path.join(savings_dol1, mesh_name2))
    
    HTC2 = MyExpression0(degree = deg_choice1,
                         domain = hol_cyl2)
    
    bool_solv2 = 0
    omega_c2 = .8  #contraction factor
    mu_c2 = 1. #from Lu's paper      

    #divisions in gamma_fun() are allowed after this iteration                                          ,
    itera_div2 = 5 #5, not 10    
    exit_cri2 = 10. #example    
    whil_tol2 = 1e-12 #example
    itera1 = 0   
    max_itera1 = 25    
    
    # ditera is an even number
    ditera = 10
    mitera = 4
    max_itera2 = max_itera1 + ditera * mitera                         
         
    #g_d2 is useless
    A2, L2, g_d2, \
    V2, v2, _, \
    mark_in2, mark_ex2, boundary_faces2, bool_ex2, \
    _, _, _, _, _, \
    unitNormal2, dofs_x2 = solve_tr_dir__const_rheo(os.path.join(savings_dol1, mesh_name2), 
                                                    hol_cyl2,
                                                    deg_choice1, 
                                                    T_in1_expr1, T_inf_expr1, HTC2,
                                                    T_old1,
                                                    k_mesh1,     cp_mesh1,     rho_mesh1, 
                                                    k_mesh_old1, cp_mesh_old1, rho_mesh_old1, 
                                                    dt1,
                                                    time_v1,
                                                    theta1,
                                                    bool_plot1, bool_solv2, savings_dol1, logger1)
    
    print 'deg_choice = ', deg_choice1
    print 'R_ex1 = ', R_ex1
    print 'len(dofs_x2) = ', len(dofs_x2)
    print 'max(g2) = ', max(g_d2[len(g_d2) - 1].vector().array())
    print 'min(g2) = ', min(g_d2[len(g_d2) - 1].vector().array()), '\n'
    
    #from T_sol_d1 to T_sol_d2
    T_sol_d2 = {}
    T_sol_d3 = {}  
    
    #stores gamma_reg's and errors
    err_d2 = []
    
    # mean and standard deviation
    #from Lu's paper
    mu_ND, sigma_ND, delta_ND = 0., 1., .1
    
    for cou_ti, val_ti in T_sol_d1.iteritems():
        #cou_ti = each time identifier
        #val_ti = value at each time
        #val_ti is sorted according to dofs_x1
        #T_sol_d2[cou_ti] is sorted according to dofs_x2
        T_sol_d2[cou_ti] = do.Function(V2)
        T_sol_d3[cou_ti] = do.Function(V2)
        
        #nearest-neighbor interpolation
        NDI = NearestNDInterpolator(dofs_x1, val_ti.vector().array())
        
        #ADD NOISE
        #delete 2nd line in 3D simulations
        T_sol_d2[cou_ti].vector()[:] = NDI(dofs_x2) + \
                         delta_ND * np.random.normal(mu_ND, sigma_ND, len(dofs_x2))
                         
        T_sol_d3[cou_ti].vector()[:] = NDI(dofs_x2) 
        
        if cou_ti == 10:
            #CHECK
            NDI_DHCP = do.File(os.path.join(savings_dol1, 'T_NDI_DHCP.pvd'))  
            NDI_IHCP = do.File(os.path.join(savings_dol1, 'T_NDI_IHCP.pvd'))  
            
            NDI_DHCP << val_ti
            NDI_IHCP << T_sol_d2[cou_ti]
    
    while itera1 < max_itera2: # and exit_cri2 > whil_tol2:
        #primal problem
        #heat flux at S_in = g1
        T_step_d2 = nablaT_fun(V2, v2, hol_cyl2,
                               A2, 
                               boundary_faces2,
                               mark_in2, mark_ex2, 
                               T_old1, #T_old1 = T_old2
                               g_d2, #S_in
                               T_sol_d2, None, #S_ex 
                               unitNormal2,
                               k_mesh_old1, cp_mesh_old1, rho_mesh_old1, 
                               dt1, #dt1 = dt2
                               time_v1,
                               theta1,
                               itera1, os.path.join(savings_dol1, mesh_name2), savings_dol1)
              
        #dual problem
        phi_step_d2 = nablap_fun(V2, v2, hol_cyl2,
                                 A2, 
                                 boundary_faces2,
                                 mark_ex2, 
                                 T_step_d2, T_sol_d2, #S_ex           
                                 k_mesh_old1, cp_mesh_old1, rho_mesh_old1, 
                                 dt1,
                                 time_v1,
                                 theta1,
                                 itera1, os.path.join(savings_dol1, mesh_name2), savings_dol1)
        
        if itera1 == 0:
            
            derJ_step_old_d2 = {}
            derJ_step_d2 = {}
            p_step_d2 = {}
            alpha_step2 = 0.
            
            for count_t_i1, t_i1 in enumerate(time_v1): 
                derJ_step_old_d2[count_t_i1] = do.Function(V2)
                derJ_step_d2[count_t_i1] = do.Function(V2)
                p_step_d2[count_t_i1] = do.Function(V2)            
        
        print 'type(p_step_d2[0]) = ', type(p_step_d2[0])
        print 'type(T_step_d2[0]) = ', type(T_step_d2[0])
        print 'type(p_step_d2[10]) = ', type(p_step_d2[10])
        print 'type(T_step_d2[10]) = ', type(T_step_d2[10])
        print 'type(k_mesh1) = ', type(k_mesh1)
        print 'type(gamma_reg2) = ', type(gamma_reg2)
        print 'type(alpha_step1) = ', type(alpha_step2)
        
        #at any itera1
        for count_t_i1, t_i1 in enumerate(time_v1): 
            '''
            TO BE UPDATED:
            k_mesh1 -> dictionary of conductivities at time t
            if conductivity changes with time t
            
            interpolation could be improved
            '''    
            '''
            derJ_step_d1[count_t_i1].vector()[:] = phi_step_d1[count_t_i1].vector().array() - \
                                                   gamma_reg1 * alpha_step1 * \
                                                   T_step_d1[count_t_i1].vector().array() / \
                                                   k_mesh1 * \
                                                   p_step_d1[count_t_i1].vector().array() 
            '''
            derJ_step_d2[count_t_i1].vector()[:] = phi_step_d2[count_t_i1].vector().array() + \
                                                   gamma_reg2 * g_d2[count_t_i1].vector().array()                            
            print 'derJ: count_t = ', count_t_i1
                                                   
            if count_t_i1 == 0:
                print 'derJ: count_t = {}, min(derJ) = {}'.format(count_t_i1, 
                                           min(derJ_step_d2[count_t_i1].vector().array()))
                
                print 'derJ: count_t = {}, max(derJ) = {}'.format(count_t_i1, 
                                           max(derJ_step_d2[count_t_i1].vector().array()))
             
        if itera1 == 0:
             gamma_CG2 = 0.
             exit_crit_v     = np.zeros(np.shape(time_v1))
             e11_v           = np.zeros(np.shape(time_v1))
             e21_v           = np.zeros(np.shape(time_v1))
             
        else:           
             gamma_CG2 = conj_coef_fun(derJ_step_d2, 
                                       derJ_step_old_d2,
                                       hol_cyl2,
                                       boundary_faces2,
                                       mark_in2, mark_ex2, 
                                       time_v1, 
                                       cut_end1,
                                       'PR', #or 'FR'
                                       itera1,
                                       savings_pic1)
            
        #print 'a-max(p_step1) = ', max(p_step1.vector().array())
        #print 'a-min(p_step1) = ', min(p_step1.vector().array())

        #at any itera1
        for count_t_i1, t_i1 in enumerate(time_v1): 
            p_step_d2[count_t_i1].vector()[:] = gamma_CG2 * p_step_d2[count_t_i1].vector().array() - \
                                                derJ_step_d2[count_t_i1].vector().array()
        
            print 'p_step: count_t = {}, min(p_step) = {}'.format(count_t_i1, 
                                         min(p_step_d2[count_t_i1].vector().array()))
                
            print 'p_step: count_t = {}, max(p_step) = {}'.format(count_t_i1, 
                                         max(p_step_d2[count_t_i1].vector().array())), '\n'
            
        #sensitivity problem
        '''
        CHECK p_step1
        '''               
        u_step_d2 = nablau_fun(V2, v2, hol_cyl2,
                               A2, 
                               boundary_faces2,
                               mark_in2,
                               p_step_d2, #dg_in on surface S_in
                               k_mesh_old1, cp_mesh_old1, rho_mesh_old1, 
                               dt1,
                               time_v1,
                               theta1,
                               itera1)
    
        
        #regularization parameter
        #obj_f1 = 1/2 * \int_S_out (T_step1 - T_ex1) \, dS
        #obj_f2 = 1/2 * \int_V \nabla T \dot \nabla T dV
        gamma_reg2_unused, gamma_reg2, \
        obj_f1, obj_f1a, obj_f1b, obj_f2 = gamma_fun(v2, hol_cyl2,
                                                     T_step_d2, #from primal pbm
                                                     boundary_faces2, 
                                                     mark_in2, mark_ex2, 
                                                     T_sol_d2, #from direct pbm
                                                     g_d2,
                                                     gamma_reg2, #the old gamma_reg
                                                     omega_c2,   #contraction factor
                                                     mu_c2,
                                                     itera_div2,  
                                                     unitNormal2,
                                                     time_v1,
                                                     cut_end1,
                                                     itera1,
                                                     savings_pic1)      
        
        if itera1 == max_itera1:
            #brute force
            #err_d2 = list of [gamma_reg2, trapz(error)]
            err_d2_np = np.array(err_d2)
            
            #line where min(trapz(error)) 
            min_err_iter = np.argmin(err_d2_np[:, 1])
            
            print 'error matrix = ', err_d2_np
            
            print 'iter where min(trapz(error)) = ', min_err_iter
            
            gamma_reg_m = err_d2_np[min_err_iter, 0]
            
            if neighbor:
                #neighbor gamma_reg2's 
                if min_err_iter > 0:
                    gamma_reg_l = err_d2_np[min_err_iter - 1, 0]
                else:
                    gamma_reg_l = 1. * gamma_reg_m
                
                if min_err_iter < len(err_d2) - 1:
                    gamma_reg_r = err_d2_np[min_err_iter + 1, 0]
                else:
                    gamma_reg_r = 1. * gamma_reg_m
            else:
                #radius gamma_reg2's 
                gamma_reg_l = .8 * gamma_reg_m
                gamma_reg_r = 1.2 * gamma_reg_m
                 
            #10 candidates * 4 = 40 iterations
            gamma_reg_lv = np.linspace(gamma_reg_l, gamma_reg_m, 
                                       num = (ditera / 2 + 2))[1 : -1]
            
            gamma_reg_rv = np.linspace(gamma_reg_m, gamma_reg_r, 
                                       num = (ditera / 2 + 2))[1 : -1]
            
            gamma_reg_lrv = np.append(gamma_reg_lv, gamma_reg_rv)
            
            gamma_reg_v = np.repeat(gamma_reg_lrv, mitera)
            
            count_gamma_reg_v = 0
                 
        if itera1 >= max_itera1:
            #overwrites the value returned by gamma_fun()
            gamma_reg2 = 1. * gamma_reg_v[count_gamma_reg_v]
            
            count_gamma_reg_v += 1
        
        #step size  
        alpha_step2 = eta_fun(v2, hol_cyl2,
                              #k1, #conductivity
                              gamma_reg2, #regularization
                              T_step_d2,   #from primal pbm
                              p_step_d2,   #from dual pbm and gamma_CG1
                              u_step_d2,   #from sensitivity pbm
                              boundary_faces2, 
                              mark_in2, mark_ex2, 
                              g_d2,
                              None, #dg_in on surface S_in (before: p_step_d1)
                              T_sol_d2,  #from direct pbm
                              unitNormal2,
                              time_v1,
                              cut_end1,
                              itera1, savings_pic1)
        
        #plot T_step11
        if bool_plot1 == 1 and itera1 % 10 == 0:
            #plot T from IHCP
            do.plot(T_step_d2[len(g_d2) - 10], title = 'T from IHCP') 
            
        print 'err: len(keys in T_sol_d2) = ', len(T_sol_d2.keys())
        print 'err: len(keys in T_step_d2) = ', len(T_step_d2.keys())
            
        #update variables
        for count_t_i1, t_i1 in enumerate(time_v1):
            #heat flux    
            g_d2[count_t_i1].vector()[:] = g_d2[count_t_i1].vector().array() + \
                                           alpha_step2 * p_step_d2[count_t_i1].vector().array() 
        
            #adjoint variable
            derJ_step_old_d2[count_t_i1].vector()[:] = derJ_step_d2[count_t_i1].vector().array()
            
            #exit criterion        
            exit_crit_v[count_t_i1] = do.assemble(pow(alpha_step2 * p_step_d2[count_t_i1], 2.) * 
                                      do.ds(mark_in2, 
                                            domain = hol_cyl2,
                                            subdomain_data = boundary_faces2)) 
            
            #do.File(os.path.join(savings_dol1, 
            #'{}__T_from_IHCP.pvd'.format(mesh_name1.split('.')[0]))) << T_step11  
            print 'err: count_t_i1 = ', count_t_i1
            #T_sol_d3 instead of T_sol_d2
            e11_v[count_t_i1], e21_v[count_t_i1] = err_est_fun(mesh_name2, 
                                                               hol_cyl2, 
                                                               T_sol_d3[count_t_i1],                                                                
                                                               T_step_d2[count_t_i1], 
                                                               deg_choice1,                                                                
                                                               hol_cyl2, count_t_i1)  
        
        #functional
        J2 = .5 * obj_f1 + .5 * gamma_reg2 * obj_f2
        
        #store the error
        err_d2.append([gamma_reg2_old, np.trapz(e21_v[: cut_end1], x = time_v1[: cut_end1])])
        gamma_reg2_old = 1. * gamma_reg2
        
        print 'itera1 = ', itera1
        print 'gamma_reg2 = {}, {}'.format(gamma_reg2, gamma_reg2_unused)
        print 'obj_f1 = ',  obj_f1
        print 'obj_f1a = ', obj_f1a
        print 'obj_f1b = ', obj_f1b
        print 'obj_f2 = ',  obj_f2
        print 'gamma_CG2 = ', gamma_CG2
        print 'alpha2 = ', alpha_step2
        print 'type(T2) = ', type(T_step_d2[0])
        print 'type(g2) = ', type(g_d2[0])
        print 'J2 = ', J2 
        
        fig_inf2 = plt.figure(figsize = (20, 10))    
            
        ax_inf4 = fig_inf2.add_subplot(211) 
        ax_inf4.plot(exit_crit_v, 'b.')
         
        ax_inf5 = fig_inf2.add_subplot(223)
        ax_inf5.plot(exit_crit_v[: 10], 'b.')
        
        ax_inf6 = fig_inf2.add_subplot(224)
        ax_inf6.plot(exit_crit_v[-10 :], 'b.')
        
        nam_inf2 = os.path.join(savings_pic1, 'exit_criterion_{}.pdf'.format(itera1))
        fig_inf2.savefig(nam_inf2, dpi = 150)
        
        exit_cri2 = np.trapz(exit_crit_v, x = time_v1)
        print 'exit criterion = ', exit_cri2
        
        #print 'max(T1) = ', max(T_step1.vector().array())
        #print 'min(T1) = ', min(T_step1.vector().array())
        print 'max(g2) = ', max(g_d2[0].vector().array())
        print 'min(g2) = ', min(g_d2[0].vector().array()), '\n' 
        
        itera1 += 1