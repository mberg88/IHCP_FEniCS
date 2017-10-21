# -*- coding: utf-8 -*-
'''
Created on Sun Apr 16 08:44:07 2017

@author: MattiaL
'''

import dolfin as do
import numpy as np
import matplotlib.pyplot as plt
#import scipy.sparse as sp
#from scipy.sparse import linalg as sp_linalg
#import copy
import os
import errno
#from mpi4py import MPI #it must go first

# <codecell>
def make_sure_path_exists(path):
    try:
        os.makedirs(path)
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise

# <codecell>
def mesh_import_fun(mesh_naming):
    #import mesh
    return do.Mesh(mesh_naming)
    
# <codecell>
def g_in_mesh(mesh_namef, x_cf, y_cf, R_inf):
    
    if 'hollow' in mesh_namef and 'cyl' in mesh_namef:
        #hollow cyl    
        #x_c is a scalar here
        #y_c is a scalar here
        return lambda x: \
        abs(((x[0] - x_cf) ** 2. + (x[1] - y_cf) ** 2.) ** .5 - R_inf) < 1e-1
    
    elif 'four' in mesh_namef and 'cyl' in mesh_namef:
        #four-hole cyl
        #x_cf is an array here
        #y_cf is an array here
        return lambda x: \
        abs(((x[0] - x_cf[0]) ** 2. + (x[1] - y_cf[0]) ** 2.) ** .5 - R_inf) < 5e-2 or \
        abs(((x[0] - x_cf[1]) ** 2. + (x[1] - y_cf[1]) ** 2.) ** .5 - R_inf) < 5e-2 or \
        abs(((x[0] - x_cf[2]) ** 2. + (x[1] - y_cf[2]) ** 2.) ** .5 - R_inf) < 5e-2 or \
        abs(((x[0] - x_cf[3]) ** 2. + (x[1] - y_cf[3]) ** 2.) ** .5 - R_inf) < 5e-2
        
    elif 'one_hole_cir' in mesh_namef:
        #four-hole cyl
        #x_cf is an array here
        #y_cf is an array here
        return lambda x: \
        abs(((x[0] - x_cf[0]) ** 2. + (x[1] - y_cf[0]) ** 2.) ** .5 - R_inf) < 1e-3
        
    elif 'reinh_cir' in mesh_namef:
        #four-hole cyl
        #x_cf is an array here
        #y_cf is an array here
        return lambda x: \
        abs(((x[0] - x_cf[0]) ** 2. + (x[1] - y_cf[0]) ** 2.) ** .5 - R_inf) < 1e-5

    elif 'small_cir' in mesh_namef:
        print 'found'
        #four-hole cyl
        #x_cf is an array here
        #y_cf is an array here
        return lambda x: \
        abs(((x[0] - x_cf[0]) ** 2. + (x[1] - y_cf[0]) ** 2.) ** .5 - R_inf) < 5e-6 or \
        abs(((x[0] - x_cf[1]) ** 2. + (x[1] - y_cf[1]) ** 2.) ** .5 - R_inf) < 5e-6 or \
        abs(((x[0] - x_cf[2]) ** 2. + (x[1] - y_cf[2]) ** 2.) ** .5 - R_inf) < 5e-6 or \
        abs(((x[0] - x_cf[3]) ** 2. + (x[1] - y_cf[3]) ** 2.) ** .5 - R_inf) < 5e-6
    
# <codecell>
def g_ex_mesh(mesh_namef, x_cf, y_cf, R_exf):
    
    if 'small_cir' in mesh_namef:
        return lambda x: abs(((x[0] - x_cf) ** 2. + (x[1] - y_cf) ** 2.) ** .5 - R_exf) < 1e-4
    elif 'reinh_cir' in mesh_namef:
        return lambda x: abs(((x[0] - x_cf) ** 2. + (x[1] - y_cf) ** 2.) ** .5 - R_exf) < 1e-4
    else:
        return lambda x: abs(((x[0] - x_cf) ** 2. + (x[1] - y_cf) ** 2.) ** .5 - R_exf) < .5e-2

# <codecell>
def solve_tr_dir__const_rheo(mesh_name, hol_cyl, deg_choice, 
                             T_in_expr, T_inf_expr, HTC,
                             T_old_v, 
                             k_mesh,     cp_mesh,     rho_mesh,
                             k_mesh_old, cp_mesh_old, rho_mesh_old,
                             dt, 
                             time_v,
                             theta,
                             bool_plot, bool_solv, savings_do, logger_f):
    
    '''
    mesh_name: a proper XML file.
    bool_plot: plots if bool_plot = 1.
    
    Solves a direct, steady-state heat conduction problem, and
    returns A_np, b_np, D_np, T_np, bool_ex, bool_in.
    
    A_np: stiffness matrix, ordered by vertices.
    
    b_np: integrated volumetric heat sources and surface heat fluxes, ordered by vertices.
    The surface heat fluxes come from the solution to the direct problem;
    hence, these terms will not be there in a real IHCP. 
    
    D_np: integrated Laplacian of T, ordered by vertices.
    Option 2.
    The Laplacian of q is properly assembled from D_np.
    If do.dx(domain = hol_cyl) -> do.Measure('ds')[boundary_faces] and
    do.Measure('ds')[boundary_faces] -> something representative of Gamma,
    I would get option 1.
    
    T_np: solution to the direct heat conduction problem, ordered by vertices.
    
    bool_ex: boolean array declaring which vertices lie on the outer boundary.
    
    bool_in: boolean array indicating which vertices lie on the inner boundary.
    
    T_sol: solution to the direct heat conduction problem.
    
    deg_choice: degree in FunctionSpace.
    
    hol_cyl: mesh.
    '''
    
    #comm1 = MPI.COMM_WORLD
    
    #current proc
    #rank1 = comm1.Get_rank()
    
    V = do.FunctionSpace(hol_cyl, 'CG', deg_choice)
    
    if 'hollow' in mesh_name and 'cyl' in mesh_name:
        from hollow_cyl_inv_mesh import geo_fun as geo_fun_hollow_cyl
        geo_params_d = geo_fun_hollow_cyl()[1]
        #x_c is a scalar here
        #y_c is a scalar here
        
    elif 'four' in mesh_name and 'cyl' in mesh_name:
        from four_hole_cyl_inv_mesh import geo_fun as geo_fun_four_hole_cyl
        geo_params_d = geo_fun_four_hole_cyl()[1]
        #x_c is an array here
        #y_c is an array here
        x_c_l = [geo_params_d['x_0_{}'.format(itera)] for itera in xrange(4)]
        y_c_l = [geo_params_d['y_0_{}'.format(itera)] for itera in xrange(4)]
                              
    elif 'one_hole_cir' in mesh_name:
        from one_hole_cir_adj_mesh import geo_fun as geo_fun_one_hole_cir
        geo_params_d = geo_fun_one_hole_cir()[1]
        #x_c is an array here
        #y_c is an array here
        x_c_l = [geo_params_d['x_0']]
        y_c_l = [geo_params_d['y_0']]
        
    elif 'reinh_cir' in mesh_name:
        from reinh_cir_adj_mesh import geo_fun as geo_fun_one_hole_cir
        geo_params_d = geo_fun_one_hole_cir()[1]
        #x_c is an array here
        #y_c is an array here
        x_c_l = [geo_params_d['x_0']]
        y_c_l = [geo_params_d['y_0']]
    
    elif 'small_circle' in mesh_name:
        from four_hole_small_cir_adj_mesh import geo_fun as geo_fun_four_hole_cir
        geo_params_d = geo_fun_four_hole_cir()[1]
        #x_c is an array here
        #y_c is an array here
        x_c_l = [geo_params_d['x_0_{}'.format(itera)] for itera in xrange(4)]
        y_c_l = [geo_params_d['y_0_{}'.format(itera)] for itera in xrange(4)]
   
    #center of the cylinder base
    x_c = geo_params_d['x_0']
    y_c = geo_params_d['y_0']
    
    R_in = geo_params_d['R_in']
    R_ex = geo_params_d['R_ex']
    
    #define variational problem
    T = do.TrialFunction(V)
    g = do.Function(V)
    v = do.TestFunction(V)
    
    T_old = do.Function(V)
    T_inf = do.Function(V)
    
    #scalar
    T_old.vector()[:] = T_old_v
    T_inf.vector()[:] = T_inf_expr

    #solution
    T_sol = do.Function(V)    
    
    #scalar
    T_sol.vector()[:] = T_old_v
    
    # Create boundary markers
    mark_all = 3
    mark_in = 4
    mark_ex = 5
                              
    #x_c is an array here
    #y_c is an array here
    g_in = g_in_mesh(mesh_name, x_c_l, y_c_l, R_in) 
        
    g_ex = g_ex_mesh(mesh_name, x_c, y_c, R_ex)
        
    in_boundary = do.AutoSubDomain(g_in)
    ex_boundary = do.AutoSubDomain(g_ex)
    
    #normal
    unitNormal = do.FacetNormal(hol_cyl)
    boundary_faces = do.MeshFunction('size_t', hol_cyl, hol_cyl.topology().dim() - 1)
    
    boundary_faces.set_all(mark_all)
    in_boundary.mark(boundary_faces, mark_in)
    ex_boundary.mark(boundary_faces, mark_ex)
    
    bc_in = do.DirichletBC(V, T_in_expr, boundary_faces, mark_in)
    #bc_ex = do.DirichletBC(V, T_ex_expr, boundary_faces, mark_ex)
    bcs = [bc_in]
    
    #k = do.Function(V)  #W/m/K    
    #k.vector()[:] = k_mesh
    
    #A0 = k * do.dot(do.grad(T), do.grad(v)) * do.dx(domain = hol_cyl)
    A = dt / 2. * k_mesh * do.dot(do.grad(T), do.grad(v)) * do.dx(domain = hol_cyl) + \
        rho_mesh * cp_mesh * T * v * do.dx(domain = hol_cyl)
    
    A_full = A + dt / 2. * HTC * T * v * do.ds(mark_ex, 
                                                         domain = hol_cyl, 
                                                         subdomain_data = boundary_faces)
        
    L = -dt / 2. * k_mesh_old * do.dot(do.grad(T_old), do.grad(v)) * \
        do.dx(domain = hol_cyl) + \
        rho_mesh_old * cp_mesh_old * T_old * v * do.dx(domain = hol_cyl) - \
        dt / 2. * HTC * (T_old) * v * do.ds(mark_ex, 
                                                    domain = hol_cyl, 
                                                    subdomain_data = boundary_faces)  + \
        dt * HTC * T_inf * v * do.ds(mark_ex, 
                                                    domain = hol_cyl, 
                                                    subdomain_data = boundary_faces)
    
    #numpy version of A, T, and (L + int_fluxT)
    #A_np__not_v2d = do.assemble(A).array() #before applying BCs - needs v2d
    #L_np__not_v2d = do.assemble(L).array() #before applying BCs - needs v2d  
    
    #Laplacian of T, without any -1/k int_S q*n*v dS
    '''
    Approximated integral of the Laplacian of T.
    Option 2.
    The Laplacian of q is properly assembled from D_np.
    If do.dx(domain = hol_cyl) -> do.Measure('ds')[boundary_faces] and
    do.Measure('ds')[boundary_faces] -> something representative of Gamma,
    I would get option 1.
    '''
    #D_np__not_v2d = do.assemble(-do.dot(do.grad(T), do.grad(v)) * do.dx(domain = hol_cyl) +
    #                             do.dot(unitNormal, do.grad(T)) * v * 
    #                             do.Measure('ds')[boundary_faces]).array() 
    #print np.max(D_np__not_v2d)#, np.max(A_np__not_v2d)
    #logger_f.warning('shape of D_np = {}, {}'.format(D_np__not_v2d.shape[0], 
    #D_np__not_v2d.shape[1]))
    
    #nonzero_entries = []
    #for row in D_np__not_v2d:
    #    nonzero_entries += [len(np.where(abs(row) > 1e-16)[0])]
        
    #logger_f.warning('max, min, and mean of nonzero_entries = {}, {}, {}'.format(
    #      max(nonzero_entries), min(nonzero_entries), np.mean(nonzero_entries)))

    #solver parameters 
    #linear solvers from
    #list_linear_solver_methods()
    #preconditioners from
    #do.list_krylov_solver_preconditioners() 
    solver = do.KrylovSolver('gmres', 'ilu')
    do.info(solver.parameters, True) #prints default values
    solver.parameters['relative_tolerance'] = 1e-16
    solver.parameters['maximum_iterations'] = 20000000
    solver.parameters['monitor_convergence'] = True #on the screen
    #http://fenicsproject.org/qa/1124/is-there-a-way-to-set-the-inital-guess-in-the-krylov-solver
    '''solver.parameters['nonzero_initial_guess'] = True'''
    solver.parameters['absolute_tolerance'] = 1e-15
    #uses whatever in q_v as my initial condition
    
    #the next lines are used for CHECK 3 only
    #A_sys, b_sys = do.assemble_system(A, L, bcs)
    
    do.File(os.path.join(savings_do, 
            '{}__markers.pvd'.format(mesh_name.split('.')[0]))) << boundary_faces
                
    if bool_plot:
        do.plot(boundary_faces, '3D mesh', title = 'boundary markers')
        
    #storage
    T_sol_d = {} 
    g_d = {}
    
    if bool_solv == 1:
    
        xdmf_DHCP_T = do.File(os.path.join(savings_do, 'DHCP', 'T.pvd'))     
        xdmf_DHCP_q = do.File(os.path.join(savings_do, 'DHCP', 'q.pvd'))
    
        for count_t_i, t_i in enumerate(time_v[1 : ]): 
            
            #T_in_expr.ts = t_i
            #T_ex_expr.ts = t_i
            
            #storage                                 
            T_sol_d[count_t_i] = do.Function(V)
            T_sol_d[count_t_i].vector()[:] = T_sol.vector().array()
            
            do.solve(A_full == L, T_sol, bcs)
            
            '''
            TO BE UPDATED:
            rheology is not updated
            '''
            
            #updates L
            T_old.assign(T_sol)
            
            T_sol.rename('DHCP_T', 'temperature from DHCP')
        
            #write solution to file
            #paraview format
            xdmf_DHCP_T << (T_sol, t_i)
            
            #plot solution
            if bool_plot:
                do.plot(T_sol, title = 'T')#, interactive = True) 
            
            logger_f.warning('len(T) = {}'.format(len(T_sol.vector().array())))
            
            print 'T: count_t = {}, min(T_DHCP) = {}'.format(count_t_i, 
                                                        min(T_sol_d[count_t_i].vector().array()))
            print 'T: count_t = {}, max(T_DHCP) = {}'.format(count_t_i, 
                                                        max(T_sol_d[count_t_i].vector().array())), '\n'
        
            #save flux - required for solving IHCP
            #same result if do.ds(mark_ex, subdomain_data = boundary_faces) 
            #instead of do.Measure('ds')[boundary_faces]
            #Langtangen, p. 37:
            #either do.dot(do.nabla_grad(T), unitNormal)
            #or do.dot(unitNormal, do.grad(T))
        
            #int_fluxT = do.assemble(-k * do.dot(unitNormal, do.grad(T_sol)) * v * 
            #                          do.Measure('ds')[boundary_faces]) 
        
            fluxT = do.project(-k_mesh * do.grad(T_sol), 
                                do.VectorFunctionSpace(hol_cyl, 'CG', deg_choice, dim = 2))
        
            if bool_plot:
                do.plot(fluxT, title = 'flux at iteration = {}'.format(count_t_i))
                
            fluxT.rename('DHCP_flux', 'flux from DHCP')
            
            xdmf_DHCP_q << (fluxT, t_i)
                                 
            print 'DHCP: iteration = {}'.format(count_t_i)
                                 
            ####################################################
            #full solution            
            #T_sol_full = do.Vector()
            #T_sol.vector().gather(T_sol_full, np.array(range(V.dim()), 'intc'))
            ####################################################  
                   
        count_t_i += 1    
        
        #copy previous lines
        #storage                                 
        T_sol_d[count_t_i] = do.Function(V)
        T_sol_d[count_t_i].vector()[:] = T_sol.vector().array()
        
    for count_t_i, t_i in enumerate(time_v): 
        #storage 
        g_d[count_t_i] = do.Function(V)
        g_d[count_t_i].vector()[:] = g.vector().array()
        
    gdim = hol_cyl.geometry().dim()
    dofmap = V.dofmap()
    dofs = dofmap.dofs()
    
    #Get coordinates as len(dofs) x gdim array
    dofs_x = V.tabulate_dof_coordinates().reshape((-1, gdim))   
    
    #booleans corresponding to the outer boundary -> ints since they are sent to root = 0
    bool_ex = 1. * np.array([g_ex(dof_x) for dof_x in dofs_x])
    #booleans corresponding to the inner boundary -> ints since they are sent to root = 0
    bool_in = 1. * np.array([g_in(dof_x) for dof_x in dofs_x]) 
    
    T_np_ex = []
    T_np_in = []
        
    for i_coor, coor in enumerate(dofs_x):
        if g_ex(coor):
            T_np_ex += [T_sol.vector().array()[i_coor]]
        if g_in(coor):
            T_np_in += [T_sol.vector().array()[i_coor]]

    print 'CHECK: mean(T) on the outer boundary = ', np.mean(np.array(T_np_ex))
    print 'CHECK: mean(T) on the inner boundary = ', np.mean(np.array(T_np_in))
    print 'CHECK: mean(HTC) = ', np.mean(do.project(HTC, V).vector().array())
        
    #v2d = do.vertex_to_dof_map(V) #orders by hol_cyl.coordinates()    
    if deg_choice == 1:
        print 'len(dof_to_vertex_map) = ', len(do.dof_to_vertex_map(V))
        
    print 'min(dofs) = ', min(dofs), ', max(dofs) = ', max(dofs)
    print 'len(bool ex) = ', len(bool_ex)
    print 'len(bool in) = ', len(bool_in)
    print 'bool ex[:10] = ', repr(bool_ex[:10])
    print 'type(T) = ', type(T_sol.vector().array())
   
    #first global results, then local results
    return A, L, g_d, \
           V, v, k_mesh, \
           mark_in, mark_ex, \
           boundary_faces, bool_ex, \
           R_in, R_ex, T_sol_d, deg_choice, hol_cyl, \
           unitNormal, dofs_x
               
# <codecell> 
def nablaT_fun(V, v, hol_cyl,
               A, 
               boundary_faces,
               mark_in, mark_ex,
               T_old_v, 
               g_d, #S_in
               T_sol_d, T_ex, #S_ex   
               unitNormal,
               k_mesh_old, cp_mesh_old, rho_mesh_old, 
               dt, 
               time_v,
               theta,
               itera, mesh_name, savings_do):
    '''
    Direct problem.
    Solves A T = L.
    A = dt * theta * k_mesh * do.dot(do.grad(T), do.grad(v)) * do.dx(domain = hol_cyl) + \
        rho_mesh * cp_mesh * T * v * do.dx(domain = hol_cyl)
        
    int_fluxT_ex = -k_mesh * do.dot(unitNormal, do.grad(T_sol)) * v * \
                         do.ds(mark_ex, 
                               domain = hol_cyl,
                               subdomain_data = boundary_faces)
                         
    no T_ex on outer boundary.                         
    '''
    #solver parameters 
    #linear solvers from
    #list_linear_solver_methods()
    #preconditioners from
    #do.list_krylov_solver_preconditioners() 
    solver = do.KrylovSolver('gmres', 'ilu')
    do.info(solver.parameters, False) #prints default values
    solver.parameters['relative_tolerance'] = 1e-13
    solver.parameters['maximum_iterations'] = 2000000
    solver.parameters['monitor_convergence'] = True #on the screen
    #http://fenicsproject.org/qa/1124/
    #is-there-a-way-to-set-the-inital-guess-in-the-krylov-solver
    solver.parameters['nonzero_initial_guess'] = True
    #solver.parameters['absolute_tolerance'] = 1e-15
    
    #T_stept = do.TrialFunction(V)
    T_step1 = do.Function(V)
    T_old = do.Function(V)
        
    #T_old_v is a scalar
    T_step1.vector()[:] = T_old_v
    T_old.vector()[:] = T_old_v
    
    T_step1_d = {}     
    
    print 'T: min(g_d[0]) = ', min(g_d[0].vector().array())
    print 'T: max(g_d[0]) = ', max(g_d[0].vector().array())
    
    theta = do.Constant(theta)
    
    dt = do.Constant(dt)
    
    T_step1_f = do.File(os.path.join(savings_do, str(itera), 'dir_IHCP.pvd'))
    
    #A = dt / 2. * k_mesh_old * do.dot(do.grad(T_stept), do.grad(v)) * do.dx(domain = hol_cyl) + \
    #    rho_mesh_old * cp_mesh_old * T_stept * v * do.dx(domain = hol_cyl)
    
    for count_t_i, t_i in enumerate(time_v[1 : ]):
        #storage
        T_step1_d[count_t_i] = do.Function(V)
        T_step1_d[count_t_i].vector()[:] = T_step1.vector().array()
        
        #g is a k * dot(grad(T), n)
        int_fluxT_in = g_d[count_t_i + 1] * v * do.ds(mark_in, 
                                                      domain = hol_cyl, 
                                                      subdomain_data = boundary_faces)
        
        int_fluxT_in_old = g_d[count_t_i] * v * do.ds(mark_in, 
                                                      domain = hol_cyl, 
                                                      subdomain_data = boundary_faces)
        
        int_fluxT_ex = + k_mesh_old * do.dot(unitNormal, do.grad(T_sol_d[count_t_i + 1])) * \
                                  v * do.ds(mark_ex, 
                                            domain = hol_cyl, 
                                            subdomain_data = boundary_faces)
                                
        int_fluxT_ex_old = + k_mesh_old * do.dot(unitNormal, do.grad(T_sol_d[count_t_i])) * \
                                      v * do.ds(mark_ex, 
                                                domain = hol_cyl, 
                                                subdomain_data = boundary_faces)
        
        #print 'T: type(int_fluxT_in) = ', type(int_fluxT_in)
        #print 'T: type(int_fluxT_in_old) = ', type(int_fluxT_in_old)
        #print 'T: min(int_fluxT_in) = ', min(do.assemble(int_fluxT_in))
        
        #L -= (int_fluxT_in + int_fluxT_ex)    
        L_old = -dt * (1. - theta) * k_mesh_old * do.dot(do.grad(T_old), do.grad(v)) * \
                do.dx(domain = hol_cyl) + \
                rho_mesh_old * cp_mesh_old * T_old * v * do.dx(domain = hol_cyl)
                
        L = L_old + (dt * theta * int_fluxT_in + dt * (1. - theta) * int_fluxT_in_old + 
                     dt * theta * int_fluxT_ex + dt * (1. - theta) * int_fluxT_ex_old)
        
        do.solve(A == L, T_step1) #, bc_ex)
        
        print 'T: count_t = {}, avg(g) = {}'.format(count_t_i, 
                                                    np.mean(g_d[count_t_i + 1].vector().array()))
        print 'T: count_t = {}, min(L) = {}'.format(count_t_i, min(do.assemble(L).array()))
        print 'T: count_t = {}, max(L) = {}'.format(count_t_i, max(do.assemble(L).array()))
        print 'T: count_t = {}, min(T) = {}'.format(count_t_i, min(T_step1.vector().array()))
        print 'T: count_t = {}, max(T) = {}'.format(count_t_i, max(T_step1.vector().array()))
        print 'T: count_t = {}, min(T_DHCP) = {}'.format(count_t_i, 
                                                    min(T_sol_d[count_t_i].vector().array()))
        print 'T: count_t = {}, max(T_DHCP) = {}'.format(count_t_i, 
                                                    max(T_sol_d[count_t_i].vector().array())), '\n'
        
        T_old.assign(T_step1)
        
        #rename
        T_step1.rename('IHCP_T', 'temperature from IHCP')
        
        T_step1_f << (T_step1, t_i)
        
    count_t_i += 1
    
    #storage
    T_step1_d[count_t_i] = do.Function(V)
    T_step1_d[count_t_i].vector()[:] = T_step1.vector().array()
    
    return T_step1_d
    
# <codecell> 
def nablap_fun(V, v, hol_cyl,
               A,  
               boundary_faces,
               mark_ex, 
               T_step_d, T_exp_d, #S_ex 
               k_mesh_oldp, cp_mesh_oldp, rho_mesh_oldp, 
               dt, 
               time_v,
               theta,
               itera, mesh_name, savings_do):
    
    '''
    Adjoint problem.
    Solves A p = L.
    A = dt * theta * k_mesh * do.dot(do.grad(p), do.grad(v)) * do.dx(domain = hol_cyl) + \
        rho_mesh * cp_mesh * p * v * do.dx(domain = hol_cyl)
        
    T_stepp: from T_step flipped.
    T_exp  : from T_ex flipped.
    '''
    #solver parameters 
    #linear solvers from
    #list_linear_solver_methods()
    #preconditioners from
    #do.list_krylov_solver_preconditioners() 
    solver = do.KrylovSolver('gmres', 'ilu')
    do.info(solver.parameters, False) #prints default values
    solver.parameters['relative_tolerance'] = 1e-13
    solver.parameters['maximum_iterations'] = 2000000
    solver.parameters['monitor_convergence'] = True #on the screen
    #http://fenicsproject.org/qa/1124/
    #is-there-a-way-to-set-the-inital-guess-in-the-krylov-solver
    '''solver.parameters['nonzero_initial_guess'] = True'''
    #solver.parameters['absolute_tolerance'] = 1e-15
    #uses whatever in q_v as my initial condition
        
    #p_stept = do.TrialFunction(V)
    p_step2 = do.Function(V)    
    
    #starts from 0
    p_old = do.Function(V)
    
    p_step2_d = {} 
    
    #e.g., e.g., count_t_i = 100
    #storage
    p_step2_d[len(time_v) - 1] = do.Function(V)
    p_step2_d[len(time_v) - 1].vector()[:] = p_old.vector().array()
              
    theta = do.Constant(theta)
    
    dt = do.Constant(dt)
    
    p_step2_f = do.File(os.path.join(savings_do, str(itera), 'adj_IHCP.pvd'))
    
    #A = dt / 2. * k_mesh_oldp * do.dot(do.grad(p_stept), do.grad(v)) * do.dx(domain = hol_cyl) + \
    #    rho_mesh_oldp * cp_mesh_oldp * p_stept * v * do.dx(domain = hol_cyl)
    
    #runs backwards in time
    for count_t_i, t_i in reversed(list(enumerate(time_v[ : -1]))):       
    
        #int_fluxT_in = 0
        #e.g., count_t_i = 99
        int_fluxT_ex = (T_step_d[count_t_i] - T_exp_d[count_t_i]) * v * \
                       do.ds(mark_ex, 
                             domain = hol_cyl, 
                             subdomain_data = boundary_faces)
                    
        #'old' = old tau = new time; e.g., count_t_i + 1 = 100
        int_fluxT_ex_old = (T_step_d[count_t_i + 1] - T_exp_d[count_t_i + 1]) * v * \
                           do.ds(mark_ex, 
                                 domain = hol_cyl, 
                                 subdomain_data = boundary_faces)
                           
        L_oldp = -dt * (1. - theta) * k_mesh_oldp * do.dot(do.grad(p_old), do.grad(v)) * \
                 do.dx(domain = hol_cyl) + \
                 rho_mesh_oldp * cp_mesh_oldp * p_old * v * do.dx(domain = hol_cyl) 

        #before:
        #L = L_oldp - (dt * theta * int_fluxT_ex + dt * (1. - theta) * int_fluxT_ex_old)                       
        L = L_oldp + (dt * theta * int_fluxT_ex + dt * (1. - theta) * int_fluxT_ex_old)
        
        do.solve(A == L, p_step2)
        
        print 'p: count_t = {}, min(T_step_d) = {}'.format(count_t_i, 
                                                    min(T_step_d[count_t_i].vector().array()))
        
        print 'p: count_t = {}, max(T_step_d) = {}'.format(count_t_i, 
                                                    max(T_step_d[count_t_i].vector().array()))
        
        print 'p: count_t = {}, min(int_fluxT_ex) = {}'.format(count_t_i, 
                                                        min(do.assemble(int_fluxT_ex).array()))
        print 'p: count_t = {}, max(int_fluxT_ex) = {}'.format(count_t_i,
                                                        max(do.assemble(int_fluxT_ex).array()))
        
        print 'p: count_t = {}, min(L) = {}'.format(count_t_i, min(do.assemble(L).array()))
        print 'p: count_t = {}, max(L) = {}'.format(count_t_i, max(do.assemble(L).array()))
        
        print 'p: count_t = {}, min(p) = {}'.format(count_t_i, min(p_step2.vector()))
        print 'p: count_t = {}, max(p) = {}'.format(count_t_i, max(p_step2.vector())), '\n'
        
        p_old.assign(p_step2)
        
        p_step2.rename('dual_T', 'dual temperature')
        
        #storage
        p_step2_d[count_t_i] = do.Function(V)
        p_step2_d[count_t_i].vector()[:] = p_step2.vector().array()
        
        p_step2_f << (p_step2, t_i)
    
    return p_step2_d
    
# <codecell> 
def nablau_fun(V, v, hol_cyl,
               A,
               boundary_faces,
               mark_in,
               dg_d, #S_in
               k_mesh_old, cp_mesh_old, rho_mesh_old, 
               dt, 
               time_v,
               theta,
               itera):
    '''
    Sensitivity problem.
    Solves A u = L.
    A = dt * theta * k_mesh * do.dot(do.grad(T), do.grad(v)) * do.dx(domain = hol_cyl) + \
        rho_mesh * cp_mesh * T * v * do.dx(domain = hol_cyl)
    
    Solved after setting dg_in = p^k.                         
    '''
    #solver parameters 
    #linear solvers from
    #list_linear_solver_methods()
    #preconditioners from
    #do.list_krylov_solver_preconditioners() 
    solver = do.KrylovSolver('gmres', 'ilu')
    do.info(solver.parameters, False) #prints default values
    solver.parameters['relative_tolerance'] = 1e-13
    solver.parameters['maximum_iterations'] = 200000
    solver.parameters['monitor_convergence'] = True #on the screen
    #http://fenicsproject.org/qa/1124/
    #is-there-a-way-to-set-the-inital-guess-in-the-krylov-solver
    '''solver.parameters['nonzero_initial_guess'] = True'''
    #solver.parameters['absolute_tolerance'] = 1e-15
    #uses whatever in q_v as my initial condition
    
    #starts from 0
    #u_stept = do.TrialFunction(V)
    u_step3 = do.Function(V)
    
    #starts from 0
    u_old = do.Function(V)
    
    u_step3_d = {} 
    
    theta = do.Constant(theta)
    
    dt = do.Constant(dt)
    
    #A = dt / 2. * k_mesh_old * do.dot(do.grad(u_stept), do.grad(v)) * do.dx(domain = hol_cyl) + \
    #   rho_mesh_old * cp_mesh_old * u_stept * v * do.dx(domain = hol_cyl)
    
    for count_t_i, t_i in enumerate(time_v[1 : ]): 
        #storage
        u_step3_d[count_t_i] = do.Function(V)
        u_step3_d[count_t_i].vector()[:] = u_step3.vector().array()
    
        #int_fluxT_ex = 0
        int_fluxT_in = dg_d[count_t_i + 1] * v * do.ds(mark_in, 
                                                       domain = hol_cyl, 
                                                       subdomain_data = boundary_faces)
        int_fluxT_in_old = dg_d[count_t_i] * v * do.ds(mark_in, 
                                                       domain = hol_cyl, 
                                                       subdomain_data = boundary_faces)
    
        #L -= (int_fluxT_in + int_fluxT_ex)    
        L_old = -dt * (1. - theta) * k_mesh_old * do.dot(do.grad(u_old), do.grad(v)) * \
                do.dx(domain = hol_cyl) + \
                rho_mesh_old * cp_mesh_old * u_old * v * do.dx(domain = hol_cyl)
                         
        L = L_old + (dt * theta * int_fluxT_in + dt * (1. - theta) * int_fluxT_in_old)
            
        do.solve(A == L, u_step3)
        
        print 'u: count_t = {}, min(dg) = {}'.format(count_t_i, 
                                                     min(dg_d[count_t_i].vector().array()))
        
        print 'u: count_t = {}, max(dg) = {}'.format(count_t_i, 
                                                     max(dg_d[count_t_i].vector().array()))
    
        print 'u: count_t = {}, min(int_fluxT_in) = {}'.format(count_t_i, 
                                                               min(do.assemble(int_fluxT_in).array()))
        print 'u: count_t = {}, max(int_fluxT_in) = {}'.format(count_t_i,
                                                               max(do.assemble(int_fluxT_in).array()))
        
        print 'u: count_t = {}, min(L) = {}'.format(count_t_i, min(do.assemble(L).array()))
        print 'u: count_t = {}, max(L) = {}'.format(count_t_i, max(do.assemble(L).array()))
        
        print 'u: count_t = {}, min(u) = {}'.format(count_t_i, min(u_step3.vector().array()))
        print 'u: count_t = {}, max(u) = {}'.format(count_t_i, max(u_step3.vector().array())), '\n'
        
        u_old.assign(u_step3)
        
    count_t_i += 1
    
    #storage
    u_step3_d[count_t_i] = do.Function(V)
    u_step3_d[count_t_i].vector()[:] = u_step3.vector().array()
        
    return u_step3_d
    
# <codecell>
def eta_fun(v, hol_cyl,
            #k, #conductivity
            gamma, #regularization
            T_step_d, #from direct pbm
            p_step_d, #from adjoint pbm and gamma_CG1
            u_step_d, #from sensitivity pbm
            boundary_faces, 
            mark_in, mark_ex,
            g_d,
            dg_d,
            T_ex_d, 
            unitNormal,
            time_v,
            cut_end,
            itera, path_to_do):
    '''
    Returns the optimal step size.
    
    T_step from direct problem.
    p_step from adjoint problem.
    u_step from sensitivity problem.
    '''
    num1_v = np.zeros((len(T_step_d), ))
    num2_v = np.zeros((len(T_step_d), ))
    den1_v = np.zeros((len(T_step_d), ))
    den2_v = np.zeros((len(T_step_d), ))
    
    for count_it in sorted(T_step_d):
        #scalar
        '''
        num1_v[count_it] = do.assemble(dg_d[count_it] * p_step_d[count_it] * 
                                       do.ds(mark_in, 
                                             domain = hol_cyl,
                                             subdomain_data = boundary_faces))
        '''
        
        num1_v[count_it] = do.assemble((T_step_d[count_it] - T_ex_d[count_it]) * 
                                       u_step_d[count_it] * 
                                       do.ds(mark_ex, 
                                             domain = hol_cyl,
                                             subdomain_data = boundary_faces))
        #scalar
        num2_v[count_it] = do.assemble(p_step_d[count_it] * 
                                       g_d[count_it] * 
                                       do.ds(mark_in, 
                                             domain = hol_cyl,
                                             subdomain_data = boundary_faces)) 
        #scalar                             
        den1_v[count_it] = do.assemble(pow(u_step_d[count_it], 2.) * 
                                       do.ds(mark_ex, 
                                             domain = hol_cyl,
                                             subdomain_data = boundary_faces))
        #scalar
        den2_v[count_it] = do.assemble(pow(p_step_d[count_it], 2.) *
                                       do.ds(mark_in, 
                                             domain = hol_cyl,
                                             subdomain_data = boundary_faces)) 
    
        print 'eta: count_t = {}, num1 = {}'.format(count_it, num1_v[count_it])
        print 'eta: count_t = {}, num2 = {}'.format(count_it, num2_v[count_it])
        print 'eta: count_t = {}, den1 = {}'.format(count_it, den1_v[count_it])
        print 'eta: count_t = {}, den2 = {}'.format(count_it, den2_v[count_it]), '\n'
            
    fig_inf = plt.figure(figsize = (20, 10))
    
    for i_fig in xrange(12):
        vari = 1. * num1_v * (i_fig / 3 == 0) + \
               1. * num2_v * (i_fig / 3 == 1) + \
               1. * den1_v * (i_fig / 3 == 2) + \
               1. * den2_v * (i_fig / 3 == 3)  
               
        ytag = 'num1' * (i_fig / 3 == 0) + \
               'num2' * (i_fig / 3 == 1) + \
               'den1' * (i_fig / 3 == 2) + \
               'den2' * (i_fig / 3 == 3)   
                  
        stai =  0 * (i_fig % 3 == 0) + \
                0 * (i_fig % 3 == 1) + \
              -50 * (i_fig % 3 == 2)    
              
        endi = len(vari) * (i_fig % 3 == 0) + \
               50 * (i_fig % 3 == 1) + \
               len(vari) * (i_fig % 3 == 2) 
               
        ax_inf = fig_inf.add_subplot(4, 3, i_fig + 1) 
        ax_inf.plot(vari[stai : endi])
        ax_inf.set_ylabel(ytag)
        ax_inf.grid()
    
    nam_inf = os.path.join(path_to_do, 'eta_{}.pdf'.format(itera))
    fig_inf.savefig(nam_inf, dpi = 150)
    
    num1_avg = np.trapz(num1_v[ : cut_end], x = time_v[ : cut_end])
    num2_avg = np.trapz(num2_v[ : cut_end], x = time_v[ : cut_end])
    den1_avg = np.trapz(den1_v[ : cut_end], x = time_v[ : cut_end])
    den2_avg = np.trapz(den2_v[ : cut_end], x = time_v[ : cut_end])
        
    return - (num1_avg + gamma * num2_avg) / (den1_avg + gamma * den2_avg)

# <codecell>
def gamma_fun(v, hol_cyl,
              T_step_d, #from direct pbm
              boundary_faces, 
              mark_in, mark_ex, 
              T_ex_d, 
              g_d,
              gamma_reg0, #the old gamma_reg
              omegac, #contraction factor
              muc, 
              itera_div,
              unitNormal,
              time_v,
              cut_end,
              itera, path_to_do): 
    
    '''
    Returns the optimal regularization parameter.
    
    T_step from direct problem.
    '''
    num1_v = np.zeros((len(T_step_d), ))
    num2_v = np.zeros((len(T_step_d), ))
    num3_v = np.zeros((len(T_step_d), ))
    den1_v = np.zeros((len(T_step_d), ))
    
    for count_it in sorted(T_step_d):
        #scalar    
        num1_v[count_it] = do.assemble(pow(T_step_d[count_it] - T_ex_d[count_it], 2.) * 
                                       do.ds(mark_ex, 
                                             domain = hol_cyl,
                                             subdomain_data = boundary_faces))
        
        num2_v[count_it] = do.assemble(pow(T_step_d[count_it], 2.) * 
                                       do.ds(mark_ex, 
                                             domain = hol_cyl,
                                             subdomain_data = boundary_faces))
        
        num3_v[count_it] = do.assemble(pow(T_ex_d[count_it], 2.) * 
                                       do.ds(mark_ex, 
                                             domain = hol_cyl,
                                             subdomain_data = boundary_faces))
        
        #scalar
        den1_v[count_it] = do.assemble(pow(g_d[count_it], 2.) * 
                                       do.ds(mark_in, 
                                             domain = hol_cyl,
                                             subdomain_data = boundary_faces)) 
    
        print 'gamma: count_t = {}, num1 = {}'.format(count_it, num1_v[count_it])
        print 'gamma: count_t = {}, den1 = {}'.format(count_it, den1_v[count_it]), '\n'
    
    fig_inf = plt.figure(figsize = (20, 10))    
          
    for i_fig in xrange(12):
        vari = 1. * num1_v * (i_fig / 3 == 0) + \
               1. * num2_v * (i_fig / 3 == 1) + \
               1. * num3_v * (i_fig / 3 == 2) + \
               1. * den1_v * (i_fig / 3 == 3)  
               
        ytag = 'num1' * (i_fig / 3 == 0) + \
               'num2' * (i_fig / 3 == 1) + \
               'num3' * (i_fig / 3 == 2) + \
               'den1' * (i_fig / 3 == 3)   
                  
        stai =  0 * (i_fig % 3 == 0) + \
                0 * (i_fig % 3 == 1) + \
              -50 * (i_fig % 3 == 2)    
              
        endi = len(vari) * (i_fig % 3 == 0) + \
               50 * (i_fig % 3 == 1) + \
               len(vari) * (i_fig % 3 == 2) 
               
        ax_inf = fig_inf.add_subplot(4, 3, i_fig + 1) 
        ax_inf.plot(vari[stai : endi])
        ax_inf.set_ylabel(ytag)
        ax_inf.grid()
        
    nam_inf = os.path.join(path_to_do, 'gamma_{}.pdf'.format(itera))
    fig_inf.savefig(nam_inf, dpi = 150)
       
    #num1_avg = np.trapz(num1_v[5 : -5], x = time_v[5 : -5])
    #den1_avg = np.trapz(den1_v[5 : -5], x = time_v[5 : -5])
    
    num1_avg = np.trapz(num1_v[ : cut_end], x = time_v[ : cut_end])
    num2_avg = np.trapz(num2_v[ : cut_end], x = time_v[ : cut_end])
    num3_avg = np.trapz(num3_v[ : cut_end], x = time_v[ : cut_end])
    den1_avg = np.trapz(den1_v[ : cut_end], x = time_v[ : cut_end])
    
    if itera > itera_div:
        #divisions are allowed
        C_avg = - 1. / den1_avg * (num2_avg + gamma_reg0 * den1_avg) ** 2. 
    
        T_avg = num2_avg / den1_avg
        
        #Eq. (2.10) in Heng's paper
        aH = 2. * muc * num3_avg
        bH = + (2. + 4. * muc) * C_avg
        cH = - (2. + 2. * muc) * C_avg * T_avg
        x10 = (-bH + (bH ** 2. - 4. * aH * cH) ** .5) / (2. * aH)
        x20 = (-bH - (bH ** 2. - 4. * aH * cH) ** .5) / (2. * aH)
        
        x11 = x10 - T_avg
        x21 = x20 - T_avg
        
        if x11 > 1e-14 and x21 > 1e-14:
            #Two positive roots
            #take the smaller one
            gamma_reg = min(x11, x21)
        else:
            #Remark 3.8 in Heng's paper
            #restart
            gamma_reg = -omegac * (T_avg +
                                   cH / bH +
                                   aH / bH * (T_avg + gamma_reg0) ** 2.)
    
        gamma_reg_unused = 2. * num1_avg / den1_avg
        
    else:
        #divisions are not allowed
        gamma_reg = 1. * gamma_reg0
        gamma_reg_unused = 1. * gamma_reg0
    
    return gamma_reg, gamma_reg_unused, \
           num1_avg, num2_avg, num3_avg, den1_avg
               
# <codecell> 
def conj_coef_fun(derJ, 
                  derJ_old,
                  hol_cyl,
                  boundary_faces,
                  mark_in, mark_ex, 
                  time_v, 
                  cut_end,
                  method, 
                  itera,
                  path_to_do):
    '''
    Conjugation coefficient.
    method = 'PR' (Polak-Ribière) or 'FR' (Fletcher-Reeves).
    '''
    num_v = np.zeros(np.shape(time_v))
    den_v = np.zeros(np.shape(time_v))
     
    for count_t_i, t_i in enumerate(time_v): 
        if method == 'PR':
            #Polak-Ribière method
            #scalar                                     
            num_v[count_t_i] = do.assemble(derJ[count_t_i] * 
                                           (derJ[count_t_i] - derJ_old[count_t_i]) * 
                                           do.ds(mark_in, 
                                                 domain = hol_cyl,
                                                 subdomain_data = boundary_faces))
            
        elif method == 'FR':
            #Fletcher-Reeves method
            #scalar                                     
            num_v[count_t_i] = do.assemble(pow(derJ[count_t_i], 2.) * 
                                           do.ds(mark_in, 
                                                 domain = hol_cyl,
                                                 subdomain_data = boundary_faces))
    
        #scalar
        den_v[count_t_i] = do.assemble(pow(derJ_old[count_t_i], 2.) * 
                                       do.ds(mark_in, 
                                             domain = hol_cyl,
                                             subdomain_data = boundary_faces))
        
    fig_inf = plt.figure(figsize = (20, 10))    
    
    ax_inf = fig_inf.add_subplot(211) 
    ax_inf.plot(num_v, 'r.', label = 'num1')
    ax_inf.plot(den_v, 'b.', label = 'den1')
    ax_inf.legend()
    
    ax_inf = fig_inf.add_subplot(223)
    ax_inf.plot(num_v[: 10], 'r.', label = 'num1')
    ax_inf.plot(den_v[: 10], 'b.', label = 'den1')
    ax_inf.legend()
    
    ax_inf = fig_inf.add_subplot(224)
    ax_inf.plot(num_v[-10 :], 'r.', label = 'num1')
    ax_inf.plot(den_v[-10 :], 'b.', label = 'den1')
    ax_inf.legend()
    
    nam_inf = os.path.join(path_to_do, 'gamma_CG_{}_{}.pdf'.format(itera, method))
    fig_inf.savefig(nam_inf, dpi = 150)
                                    
    #integration
    num_gamma_CG = np.trapz(num_v[ : cut_end], x = time_v[ : cut_end]) 
    den_gamma_CG = np.trapz(den_v[ : cut_end], x = time_v[ : cut_end])                            
    
    #https://en.wikipedia.org/wiki/Nonlinear_conjugate_gradient_method
    gamma_CG = max(0., num_gamma_CG / den_gamma_CG)
    
    return gamma_CG

# <codecell>           
def err_est_fun(mesh_name2, hol_cyl, 
                T_sol1, T_sol2, 
                deg_choice2, hol_cyl2, count_it):
    '''
    Error estimate.
    '''
    #comm1 = MPI.COMM_WORLD
    
    #rank1 = comm1.Get_rank()
    
    #e1 = sqrt(int_Omega (T - T_ex)**2. dx)/sqrt(int_Omega T**2. dx)
    #do.assemble yields integration                     
    e1 = do.sqrt(do.assemble(pow(T_sol2 - T_sol1, 2.) * do.dx(domain = hol_cyl))) / \
         do.sqrt(do.assemble(pow(T_sol1, 2.) * do.dx(domain = hol_cyl)))
    
    #sometimes e1 does not work properly (it might be unstable) -> e2
    #the degree of piecewise polynomials used to approximate T_th and T...
    #...will be the degree of T + degree_rise
    e2 = do.errornorm(T_sol1, T_sol2, norm_type = 'l2', degree_rise = 2, mesh = hol_cyl2) / \
         do.sqrt(do.assemble(pow(T_sol1, 2.) * do.dx(domain = hol_cyl)))
         
    print 'err: count_t = {}, error_1 = {}, error_2 = {}'.format(count_it, e1, e2)
    
    #print 'rank = ', rank1, ', 
    print 'max(T_sol) = ', max(T_sol2.vector().array())
    #print 'rank = ', rank1, ', 
    print 'min(T_sol) = ', min(T_sol2.vector().array()), '\n'
    
    return e1, e2    