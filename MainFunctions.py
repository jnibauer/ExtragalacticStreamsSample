import numpy as np
import astropy.units as u
import gala.dynamics as gd
import gala.integrate as gi
import gala.potential as gp
from gala.units import galactic, UnitSystem
from gala.dynamics import mockstream as ms
import astropy.coordinates as coord
_ = coord.galactocentric_frame_defaults.set('v4.0')
from astropy.constants import G
from astropy.coordinates.matrix_utilities import rotation_matrix


def get_trial_pot(q=None, mass_halo_disk_ratio=1, origin=np.array([0.,0.]), r_s_halo=16*u.kpc):
    Mass_halo = 1*mass_halo_disk_ratio*u.Msun
    Mass_disk = 1*u.Msun
    v_c_halo = (np.sqrt(2.0*G*Mass_halo/r_s_halo)).to(u.km/u.s)
    pot_log = gp.LogarithmicPotential(v_c=v_c_halo,r_h=r_s_halo,q1=1,q2=1,q3=q,units=galactic,origin=np.array([origin[0],0.,origin[1]])*u.kpc)
    pot_trial = pot_log 
    return pot_trial


def get_trial_pot_rot(q=0.75, rot=25., Halo_Disk_Ratio=50.):
    R_arr = rotation_matrix(rot*u.deg, 'y')
    
    Mass_Halo_true = 6e11*u.Msun
    Mass_Disk_true = Mass_Halo_true/50
    
    Mass_Halo = Halo_Disk_Ratio*Mass_Disk_true
    
    r_s_halo = 10*u.kpc
    v_c_halo = (np.sqrt(2.0*G*Mass_Halo/r_s_halo)).to(u.km/u.s)
    ###############Mass_Disk = Mass_Halo/50 

    disk_pot = gp.MiyamotoNagaiPotential(m=Mass_Disk_true,a=3*u.kpc,b=.5*u.kpc,units=galactic)
    halo_pot = gp.LogarithmicPotential(v_c=v_c_halo,r_h=r_s_halo,q3=q,units=galactic,R=R_arr)

    pot_trial = disk_pot + halo_pot
    return pot_trial



def get_trial_pot_rot_no_disk(q=0.75, rot=25.):
    R_arr = rotation_matrix(rot*u.deg, 'y')
    
    Mass_Halo_true = 6e11*u.Msun
    #Mass_Disk_true = Mass_Halo_true/50
    
    #Mass_Halo = Halo_Disk_Ratio*Mass_Disk_true
    
    r_s_halo = 10*u.kpc #10kpc
    v_c_halo = (np.sqrt(2.0*G*Mass_Halo_true/r_s_halo)).to(u.km/u.s)
    #Mass_Disk = Mass_Halo/50 

    #disk_pot = gp.MiyamotoNagaiPotential(m=Mass_Disk_true,a=3*u.kpc,b=.5*u.kpc,units=galactic)
    halo_pot = gp.LogarithmicPotential(v_c=v_c_halo,r_h=r_s_halo,q3=q,units=galactic,R=R_arr)

    pot_trial = halo_pot
    return pot_trial






def get_your_fractions(kappa_hat, d_T_hat_dgamma, a_xy_hat, thresh_f0 = .01):
    N = float(len(kappa_hat))
    kappa_undef_bool = np.sqrt( np.sum(d_T_hat_dgamma**2,axis=1) ) < thresh_f0
    
    kappa_hat_defined = kappa_hat[~kappa_undef_bool,:]
    N_kappa = float(len(kappa_hat_defined))
    a_xy_hat_kappa_defined = a_xy_hat[~kappa_undef_bool,:]
    
    f_less = (1./N)*np.sum( 0.5*( 
        np.abs( 1.0 + np.sign( np.sum( a_xy_hat_kappa_defined*kappa_hat_defined, axis=1 ) ) )
        ) )
    
    f_gtr = (N_kappa/N) - f_less
    f0_check = 1.0 - (f_gtr + f_less)
        
    return f_less, f_gtr, f0_check, kappa_undef_bool

def log_like(N,f_less, f_gtr, f0=None, a_xy_hat=None, T_hat=None, kappa_undef_bool=np.array([0])):
    """
    Log likelihood for a potential with symmetry along the line-of-sight (i.e., varying the LOS coordinate z does not
    change the direction of unit vector accelerations in the plane of the sky, called the x-y plane. The is true for a 
    flattened logarithmic potential with the flattening axis orthogonal to the LOS). 
    
    N is the number of evaluation points, f_less is the fraction of eval. pts with theta [angle between curvature and planar acceleration]
    less the pi/2, f_gtr is the fraction of eval. pts with theta > pi/2, f0 is the fraction of evaluation points with zero curvature,
    a_xy_hat are the trial unit vector accelerations in the x-y plane at each evaluation point, T_hat are the tangent vectors for each point,
    and kappa_undef_bool is a boolean array that is True for evaluation points with ZERO curvature, and False for evaluation points
    with non-zero curvature. 
    
    Note that f_less and f_gtr are both functions of a potential model, but f0 is FIXED. 
    """
    if np.isclose(f_less,0.0):
        f_less_log_f_less = 0.
    else:
        f_less_log_f_less = f_less*np.log(f_less)
        
    if np.isclose(f_gtr,0.0):
        f_gtr_log_f_gtr = 0.
    else:
        f_gtr_log_f_gtr = f_gtr*np.log(f_gtr)
        
    if np.isclose(f0,0.0):
        f0_log_f0 = 0.
    else:
        f0_log_f0 = f0*np.log(f0)
    if f_less < f_gtr:
        return -np.inf
    
    if kappa_undef_bool.sum() > 0:
        if f_less > 0.5:
            sigma_theta = np.deg2rad(10.) #10
            Normal_Vec =  np.vstack([ T_hat[kappa_undef_bool,1], -T_hat[kappa_undef_bool,0] ]).T
            theta_T = np.pi/2 - np.arccos( np.sum(a_xy_hat[kappa_undef_bool,:]*Normal_Vec, axis=1) )
            log_gauss = np.log(1./np.sqrt(2*np.pi*(sigma_theta**2))) - (1./(2*sigma_theta**2))*((theta_T - 0.)**2)
            return  N*(f_less_log_f_less + f_gtr_log_f_gtr + 0.) + np.sum( log_gauss )
    if kappa_undef_bool.sum() == 0:
        return N*(f_less_log_f_less + f_gtr_log_f_gtr + 0.)

    
    
    

def get_acc_stream_angle(d_T_hat_dgamma_over_mag, xz_acc_unit):
    cross_prod = np.cross(d_T_hat_dgamma_over_mag, xz_acc_unit)
    dot_prod = np.sum(d_T_hat_dgamma_over_mag*xz_acc_unit ,axis=1)
    acc_stream_angle = np.arctan2(cross_prod,dot_prod)
    return acc_stream_angle
    
