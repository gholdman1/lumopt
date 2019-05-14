""" Copyright chriskeraly
    Copyright (c) 2019 Lumerical Inc. 
    Author: Greg Holdman (gholdman1)
    """

import sys
import numpy as np
import scipy as sp
import scipy.constants
import lumapi

from lumopt.utilities.wavelengths import Wavelengths
from lumopt.lumerical_methods.lumerical_scripts import get_fields

class PointElectric(object):

    """ 
    Planned Purpose
    ----------
    Calculates the figure of merit as the squared magnitude of the electric field.
        
    Follow the "Simple Example" in Section 5.1 of the following reference:

        Miller, O.D. (2012). Photonic Design: From Fundamental Solar Cell Physics to
        Computational Inverse Design. University of California, Berkeley.
        
    Parameters
    ----------
    
    :param monitor_name: name of the power monitor that records the electric field at a point.
    """

    def __init__(self, monitor_name):
        self.monitor_name = str(monitor_name)
        if not self.monitor_name:
            raise UserWarning('empty monitor name.')
        self.mode_expansion_monitor_name = monitor_name + '_mode_exp'
        self.adjoint_source_name = monitor_name + '_dipole_src'

    def add_to_sim(self, sim):
        """
        Probably not used in this FOM, but I'm not sure what its purpose
        is in ModeMatch.
        """
        pass

    def get_fom(self, sim):
        """
        FOM is the squared magnitude of the E-field at monitor 'fom', a la
        Section 5.1 of Ref 1

        References
        ----------

        [1] Miller, O.D. (2012). Photonic Design: From Fundamental Solar Cell Physics
            to Computational Inverse Design. University of California, Berkeley.
            http://optoelectronics.eecs.berkeley.edu/ThesisOwenMiller.pdf


        """
        self.wavelengths = PointElectric.get_wavelengths(sim)
        E_field_object = get_fields(sim.fdtd, 'fom',False,False,False,False)

        self.forward_E = E_field_object.E.squeeze()
        self.phase_prefactors = np.linalg.norm(self.forward_E)
        fom = np.linalg.norm(self.forward_E)**2
        return fom

    def get_adjoint_field_scaling(self, sim):
        # Only getting an order of magnitude scaling
        # What is the correct expression from simulation constants?
        omega = 2 * np.pi * sp.constants.speed_of_light / self.wavelengths
        return self.phase_prefactors * 10**12 *omega* 1j * np.ones_like(self.wavelengths)

    @staticmethod
    def get_wavelengths(sim):
        return Wavelengths(sim.fdtd.getglobalsource('wavelength start'), 
                           sim.fdtd.getglobalsource('wavelength stop'),
                           sim.fdtd.getglobalmonitor('frequency points')).asarray()

    def add_adjoint_sources(self, sim):
        """
        Adds 2 (3) dipole sources in a 2D (3D) simulation
        at the location of the figure of merit.
        """

        # Only address the relevant directions
        if sim.fdtd.getnamed('FDTD','dimension')=='3D':
            dirs = ('x','y','z')
        if sim.fdtd.getnamed('FDTD','dimension')=='2D':
            dirs = ('x','y')

        for i,cartesian in enumerate(dirs):
            E_conj = np.conj(self.forward_E[i])
            amplitude = float(np.abs(E_conj))
            phase     = float(np.angle(E_conj)) * 360 / (2*np.pi) # degrees
            PointElectric.add_dipole_source(sim,self.monitor_name,
                                        self.adjoint_source_name,
                                        cartesian,amplitude,phase)

    @staticmethod
    def add_dipole_source(sim, monitor_name, source_name, cartesian, amplitude, phase):
        '''
        Places a dipole at the location of a point monitor 'monitor_name'

        Parameters
        ----------

        sim:            Simulation object
        monitor_name:   string, name of the monitor where the dipole source will be placed
        source_name:    name of the dipole to be placed
        cartesian:      string, one of 'x', 'y', or 'z', the cartesian direction
                        in which the dipole will point.
        amplitude:      scalar (real), the amplitude of the dipole
        phase:          scalar (real), the phase of the dipole
        '''
        sim.fdtd.adddipole()
        source_name_cart = source_name + '_' + cartesian
        sim.fdtd.set('name', source_name_cart)

        for coord in ('x','y','z'):
            monitor_coord=sim.fdtd.getnamed(monitor_name,coord)
            sim.fdtd.setnamed(source_name_cart,coord,monitor_coord)

        sim.fdtd.setnamed(source_name_cart,'amplitude',amplitude)
        sim.fdtd.setnamed(source_name_cart,'phase',phase)

        # Rotate dipole to point in cartesian direction
        if cartesian=='z':
            sim.fdtd.setnamed(source_name_cart,'theta',0)
            sim.fdtd.setnamed(source_name_cart,'phi',0)
        if cartesian=='x':
            sim.fdtd.setnamed(source_name_cart,'theta',90)
            sim.fdtd.setnamed(source_name_cart,'phi',0)
        if cartesian=='y':
            sim.fdtd.setnamed(source_name_cart,'theta',90)
            sim.fdtd.setnamed(source_name_cart,'phi',90)

    @staticmethod
    def fom_wavelength_integral(T_fwd_vs_wavelength, wavelengths, target_T_fwd, norm_p):
        target_T_fwd_vs_wavelength = target_T_fwd(wavelengths).flatten()
        if len(wavelengths) > 1:
            wavelength_range = wavelengths.max() - wavelengths.min()
            assert wavelength_range > 0.0, "wavelength range must be positive."
            T_fwd_integrand = np.power(np.abs(target_T_fwd_vs_wavelength), norm_p) / wavelength_range
            const_term = np.power(np.trapz(y = T_fwd_integrand, x = wavelengths), 1.0 / norm_p)
            T_fwd_error = np.abs(T_fwd_vs_wavelength.flatten() - target_T_fwd_vs_wavelength)
            T_fwd_error_integrand = np.power(T_fwd_error, norm_p) / wavelength_range
            error_term = np.power(np.trapz(y = T_fwd_error_integrand, x = wavelengths), 1.0 / norm_p)
            fom = const_term - error_term
        else:
            fom = np.abs(target_T_fwd_vs_wavelength) - np.abs(T_fwd_vs_wavelength.flatten() - target_T_fwd_vs_wavelength)
        return fom.real

    def fom_gradient_wavelength_integral(self, partial_derivs_vs_wl, wl):
        '''
        
        :param partial_derivs_vs_wl:    n x m numpy array where n is number of wls
                                        and m is number of parameters
        :param wl:                      wavelengths
        '''
        assert partial_derivs_vs_wl.shape[0] == wl.size
        assert np.allclose(wl, self.wavelengths)
        # target_T_fwd_vs_wavelength = self.target_T_fwd(self.wavelengths).flatten()
        if self.wavelengths.size > 1:
            num_opt_param = partial_derivs_vs_wl.shape[1]
            wavelength_range = self.wavelengths.max() - self.wavelengths.min()
            T_fwd_error = self.T_fwd_vs_wavelength - target_T_fwd_vs_wavelength
            T_fwd_error_integrand = np.power(np.abs(T_fwd_error), self.norm_p) / wavelength_range
            const_factor = -1.0 * np.power(np.trapz(y = T_fwd_error_integrand, x = self.wavelengths), 1.0 / self.norm_p - 1.0)
            integral_kernel = np.power(np.abs(T_fwd_error), self.norm_p - 1) * np.sign(T_fwd_error) / wavelength_range
            T_fwd_partial_derivs = np.zeros(num_opt_param, dtype = 'complex')
            for i in range(num_opt_param):
                T_fwd_partial_deriv = np.take(T_fwd_partial_derivs_vs_wl, indices = i, axis = 1)
                T_fwd_partial_derivs[i] = const_factor * np.trapz(y = integral_kernel * T_fwd_partial_deriv, x = self.wavelengths)
        else:
            partial_derivs = partial_derivs_vs_wl.flatten()
        return partial_derivs.flatten().real
