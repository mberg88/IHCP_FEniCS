#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  2 13:48:58 2017

@author: mattiaub
"""

import dolfin as do
import mshr as ms 
import numpy as np
import matplotlib.pyplot as plt
import os

# <codecell> 
def geo_fun():
    '''
    Creates hollow circle.
    '''
    geo_params = {}
    geo_params['R_in'] = 8e-3 #m
    geo_params['R_ex'] = 16e-3 #m
    geo_params['x_0'] = 0.
    geo_params['y_0'] = 0.
    geo_params['z_0'] = 0.
    
    #number of faces on the side when generating a polyhedral approximation
    #CGAL meshes the polyhedral approximation; hence, it is better to keep it high
    #https://bitbucket.org/fenics-project/mshr/issues/26/creation-of-mesh-with-generate_mesh-most
    fragments1 = 1000
    fragments2 = 400
    
    cyl_ho = ms.Circle(do.Point(geo_params['x_0'], geo_params['y_0'], geo_params['z_0']), 
                       geo_params['R_ex'], 
                       fragments2)
    
    #x0_l = [-.5, 0., .5, 0.]
    #y0_l = [0., -.5, 0., .5]
   
    x0_l = [0.]
    y0_l = [0.]
   
    for itera in xrange(len(x0_l)):
        geo_params['x_0_{}'.format(itera)] = x0_l[itera]
        geo_params['y_0_{}'.format(itera)] = y0_l[itera]
        geo_params['z_0_{}'.format(itera)] = 0.
        
        #top, bottom, top radius, bottom radius, segments
        cyl_in = ms.Circle(do.Point(geo_params['x_0_{}'.format(itera)], 
                                    geo_params['y_0_{}'.format(itera)], 
                                    geo_params['z_0_{}'.format(itera)]), 
                           geo_params['R_in'],
                           fragments1)
                         
        #hollow cylinder
        cyl_ho -= cyl_in
    
    return cyl_ho, geo_params
    
# <codecell>  
def mes_fun(shape, resol_n):
    '''
    Meshes a figure.
    '''
    # Creating a mesh generator object gives access to parameters of the
    # meshing backend    
    generator = ms.CSGCGALMeshGenerator2D()
    
    '''
    edge_size: a scalar field (resp. a constant) 
    providing a space varying (resp. a uniform) upper bound 
    for the lengths of curve segment edges. This parameter has to be set to a positive value 
    when 1-dimensional features protection is used.
    facet_angle: a lower bound for the angles (in degrees) of the surface mesh facets.
    facet_size: a scalar field (resp. a constant) 
    describing a space varying (resp. a uniform) upper-bound 
    for the radii of the surface Delaunay balls:
    surface facets have by definition an empty circumsphere centered on the surface.
    facet_distance: a scalar field (resp. a constant) 
    describing a space varying (resp. a uniform) upper bound 
    for the same distance.
    facet_topology: the set of topological constraints 
    which have to be verified by each SURFACE facet. 
    The default value is CGAL::FACET_VERTICES_ON_SURFACE. 
    See Mesh_facet_topology manual page to get all possible values.
    cell_radius_edge_ratio: an upper bound for the radius-edge ratio of the mesh tetrahedra.
    cell_size: a scalar field (resp. a constant) 
    describing a space varying (resp. a uniform) upper-bound 
    for the circumradii of the mesh tetrahedra.

    Note that each size or distance parameter can be specified using two ways: 
    either as scalar field or as a numerical value when the field is uniform.
    
    https://github.com/FEniCS/mshr/blob/master/include/mshr/CSGCGALMeshGenerator3D.h
    Parameters:
    p.add('mesh_resolution', 64.0);
    p.add('perturb_optimize', false);
    p.add('exude_optimize', false);
    p.add('lloyd_optimize', false);
    p.add('odt_optimize', false);
    p.add('edge_size', 0.025);
    p.add('facet_angle', 25.0);
    p.add('facet_size', 0.05);
    p.add('facet_distance', 0.005);
    p.add('cell_radius_edge_ratio', 3.0);
    p.add('cell_size', 0.05);
    p.add('detect_sharp_features', true);
    p.add('feature_threshold', 70.);
    
    Other examples:
    facet_angle=30, facet_size=0.1, facet_distance=0.025, cell_radius_edge_ratio=2
    CGAL::make_mesh_3<C3t3>(domain, criteria, no_exude(), no_perturb()
    ******************************************************************************
    edge_size = 0.15, facet_angle = 25, facet_size = 0.15, cell_radius_edge_ratio = 2, 
    cell_size = 0.15
    ******************************************************************************
    edge_size = 0.025, facet_angle = 25, facet_size = 0.05, facet_distance = 0.005, 
    cell_radius_edge_ratio = 3, cell_size = 0.05
    ******************************************************************************
    After fixing radius bound of surface facets to 0.01,
    the surface mesh gets denser. 
    '''
    
    print generator.parameters.keys()
    
    generator.parameters['mesh_resolution'] = resolution_n
    
    #generator.parameters['edge_size'] = .025
    #generator.parameters['facet_angle'] = 25. #degrees
    #generator.parameters['facet_size'] = .025 
    #generator.parameters['cell_size'] = 0.00015 #crucial 
    #max(generator.parameters['cell_size']) = 0.0005
    #generator.parameters['cell_size'] = 0.0002 is quite acceptable. However, 
    #outer boundary is meshed in an asymmetric way
    #min(generator.parameters['cell_size']) = 0.0001 -> leads to 297387 vertices
    
    #generator.parameters['cell_radius_edge_ratio'] = 3.
    #generator.parameters['perturb_optimize'] = True
    #generator.parameters['lloyd_optimize'] = True
    
    #https://bitbucket.org/fenics-project/mshr/issues/61/typeerror-in-method
    domain_ho = ms.CSGCGALDomain2D(shape)
    
    mes_ho = generator.generate(domain_ho)
    
    return mes_ho, generator
    
# <codecell>
def refine_bo_fun(mes_ho, boundary_name):
    '''
    Refines faces on a given boundary.
    '''
    geo_params_d = geo_fun()[1]
    
    #center of the cylinder base
    x_c = geo_params_d['x_0']
    y_c = geo_params_d['y_0']
    #z_c = geo_params_d['z_0']
    
    x_c_l = [geo_params_d['x_0_{}'.format(itera)] for itera in xrange(1)]
    y_c_l = [geo_params_d['y_0_{}'.format(itera)] for itera in xrange(1)]
    #z_c_l = [geo_params_d['z_0_{}'.format(itera)] for itera in xrange(4)]
    
    R_in = geo_params_d['R_in']
    R_ex = geo_params_d['R_ex']  
        
    markers = do.CellFunction('bool', mes_ho)
    markers.set_all(False)
    
    for cell in do.cells(mes_ho):
        for facet in do.facets(cell):
            if 'outer' in boundary_name:
                if abs(((facet.midpoint()[0] - x_c) ** 2. + 
                        (facet.midpoint()[1] - y_c) ** 2.) ** .5 - R_ex) < 5e-2:
                    #mark cells with facet midpoints close to the outer boundary
                    markers[cell] = True
                    print 'refinement close to the outer boundary'
            elif 'inner' in boundary_name:
                if abs(((facet.midpoint()[0] - x_c_l[0]) ** 2. + \
                        (facet.midpoint()[1] - y_c_l[0]) ** 2.) ** .5 - R_in) < 1e-4:
                         #mark cells with facet midpoints close to the inner boundary
                         markers[cell] = True
                         print 'refinement close to the inner boundary'
    
    mes_ho_refined = do.refine(mes_ho, markers)    
    
    return mes_ho_refined
    
# <codecell>
def refine_vo_fun(mes_ho):
    '''
    Refines huge cells.
    '''
        
    markers = do.CellFunction('bool', mes_ho)
    markers.set_all(False)
    
    avg_cell_volume = np.mean([cell.volume() for cell in do.cells(mes_ho)]) 
    
    for cell in do.cells(mes_ho):
        if cell.volume() > 5. * avg_cell_volume:
            #mark huge cells
            markers[cell] = True
    
    mes_ho_refined = do.refine(mes_ho, markers)  
    
    print 'mean(cell_volume) = ', avg_cell_volume
    
    return mes_ho_refined
    
# <codecell> 
if __name__ == '__main__':     
    '''
    Generation and mesh of a simplified puck.
    DHCP: resolution = 64.
    IHCP: resolution = 32.
    '''
    plt.close('all')
    
    #parameters
    mesh_name1 = 'reinh_circle'
    refined_patch = 'inner_boundary'
    smoothing = 1
    savings_dol1 = os.path.join(os.getcwd(), 'pics_dolfin')
    savings_pic1 = os.path.join(os.getcwd(), 'pics_plt') 
    
    for directory in [savings_dol1, savings_pic1]:
        if not os.path.exists(directory):
            os.makedirs(directory)
   
    cyl_ho1 = geo_fun()[0]
    
    
    #mes_ho_partial1, generator1 = mes_fun(cyl_ho1)
    
    #refinement
    #mes_ho1 = refine_bo_fun(mes_ho_partial1, refined_patch)
    
    resolution_s = ['DHCP', 'IHCP']
    resolution_l = [64.,     32.]
    
    for resolution_c, resolution_n in zip(resolution_s, resolution_l):
        print ' '
        print 'resolution = ', resolution_c, resolution_n
        mes_ho1, generator1 = mes_fun(cyl_ho1, resolution_n)
        
        #perturb_optimize after refinement
        #generator1.parameters['lloyd_optimize'] = True
        #exude_optimize after refinement
        #generator1.parameters['exude_optimize'] = True
        #generator1.generate(cyl_ho1, mes_ho1)
        
        #mes_ho1 = refine_bo_fun(mes_ho1, 'outer_boundary')
        #mes_ho1 = refine_vo_fun(refine_bo_fun(mes_ho_partial1, 'outer_boundary'))
        
        if smoothing > 1:
            #smoothing
            mes_ho1.smooth(smoothing)
        
        #print 'info = ', do.info(mes_ho)
        print 'number of vertices = ', len(mes_ho1.coordinates())
        print 'number of cells = ', len(mes_ho1.cells())
        print 'largest size = ', mes_ho1.hmax()
        print 'smallest size = ', mes_ho1.hmin()
        cell_volumes = [cell.volume() for cell in do.cells(mes_ho1)]
        print 'min(cell volume) = {}, max(cell volume) = {}'.format(min(cell_volumes), 
                                                                    max(cell_volumes))
        
        #histogram
        fig1 = plt.figure(figsize = (10, 10))
        ax1 = fig1.add_subplot(111) #p
        ax1.hist(cell_volumes, bins = 30, histtype = 'stepfilled', color = 'b', alpha = .5)
        ax1.set_title('Cell volumes')
        ax1.set_xlabel('Value')
        ax1.set_ylabel('Frequency')
        filename1 = os.path.join(savings_pic1, 
                                 '{}__cell_volumes_{}.pdf'.format(mesh_name1, resolution_c))
        fig1.savefig(filename1, dpi = 150)
    
        do.plot(mes_ho1, '3D mesh')
        
        #dolfin format
        do.File(os.path.join(savings_dol1, '{}_{}.xml.gz'.format(mesh_name1, 
                             resolution_c))) << mes_ho1
        
        #paraview format
        do.File(os.path.join(savings_dol1, '{}_{}.pvd'.format(mesh_name1, 
                             resolution_c))) << mes_ho1
    
        #do.interactive()