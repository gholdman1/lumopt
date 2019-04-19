""" Copyright chriskeraly
    Copyright (c) 2019 Lumerical Inc. 
    Adapted by gholdman1
    """

import sys
import numpy as np
import scipy as sp
import scipy.constants
import lumapi

from lumopt.utilities.wavelengths import Wavelengths

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
    :param target_T_fwd: function describing the target T_forward vs wavelength (see documentation for mode expansion monitors).
    :param norm_p:       exponent of the p-norm used to generate the figure of merit; use to generate the FOM.
    """

    def __init__(self, monitor_name, target_T_fwd = lambda wl: np.ones(wl.size), norm_p = 1):
        self.monitor_name = str(monitor_name)
        if not self.monitor_name:
            raise UserWarning('empty monitor name.')
        self.mode_expansion_monitor_name = monitor_name + '_mode_exp'
        self.adjoint_source_name = monitor_name + '_dipole_src'

        target_T_fwd_result = target_T_fwd(np.linspace(0.1e-6, 10.0e-6, 1000))
        if target_T_fwd_result.size != 1000:
            raise UserWarning('target transmission must return a flat vector with the requested number of wavelength samples.')
        elif np.any(target_T_fwd_result.min() < 0.0) or np.any(target_T_fwd_result.max() > 1.0):
            raise UserWarning('target transmission must always return numbers between zero and one.')
        else:
            self.target_T_fwd = target_T_fwd
        self.norm_p = int(norm_p)
        if self.norm_p < 1:
            raise UserWarning('exponent p for norm must be positive.')

    def add_to_sim(self, sim):
        ModeMatch.add_mode_expansion_monitor(sim, self.monitor_name, self.mode_expansion_monitor_name, self.mode_number)

    @staticmethod
    def add_mode_expansion_monitor(sim, monitor_name, mode_expansion_monitor_name, mode_number):
        # modify existing DFT monitor
        if sim.fdtd.getnamednumber(monitor_name) != 1:
            raise UserWarning('monitor could not be found or the specified name is not unique.')
        sim.fdtd.setnamed(monitor_name, 'override global monitor settings', False)
        # append a mode expansion monitor to the existing DFT monitor
        if sim.fdtd.getnamednumber(mode_expansion_monitor_name) == 0:
            sim.fdtd.addmodeexpansion()
            sim.fdtd.set('name', mode_expansion_monitor_name)
            sim.fdtd.setexpansion(mode_expansion_monitor_name, monitor_name)
            sim.fdtd.setnamed(mode_expansion_monitor_name, 'mode selection', 'user select')
            sim.fdtd.setnamed(mode_expansion_monitor_name, 'auto update before analysis', True)
            sim.fdtd.setnamed(mode_expansion_monitor_name, 'override global monitor settings', False)
            # properties that must be synchronized
            props = ['monitor type']
            monitor_type = sim.fdtd.getnamed(monitor_name, 'monitor type')
            geo_props, normal = ModeMatch.cross_section_monitor_props(monitor_type)
            props.extend(geo_props)
            # synchronize properties
            for prop_name in props:
                prop_val = sim.fdtd.getnamed(monitor_name, prop_name)
                sim.fdtd.setnamed(mode_expansion_monitor_name, prop_name, prop_val)
            # select mode
            sim.fdtd.select(mode_expansion_monitor_name)
            sim.fdtd.updatemodes(mode_number)
        else:
            raise UserWarning('there is already a expansion monitor with the same name.')

    @staticmethod
    def cross_section_monitor_props(monitor_type):
        geometric_props = ['x', 'y', 'z']
        normal = ''
        if monitor_type == '2D X-normal':
            geometric_props.extend(['y span','z span'])
            normal = 'x'
        elif monitor_type == '2D Y-normal':
            geometric_props.extend(['x span','z span'])
            normal = 'y'
        elif monitor_type == '2D Z-normal':
            geometric_props.extend(['x span','y span'])
            normal = 'z'
        elif monitor_type == 'Linear X':
            geometric_props.append('x span')
            normal = 'y'
        elif monitor_type == 'Linear Y':
            geometric_props.append('y span')
            normal = 'x'
        elif monitor_type == 'Linear Z':
            geometric_props.append('z span')
        else:
            raise UserWarning('monitor should be 2D or linear for a mode expansion to be meaningful.')
        return geometric_props, normal

    def get_fom(self, sim):
        trans_coeff = ModeMatch.get_transmission_coefficient(sim, self.direction, self.monitor_name, self.mode_expansion_monitor_name)
        self.wavelengths = ModeMatch.get_wavelengths(sim)
        source_power = ModeMatch.get_source_power(sim, self.wavelengths)
        self.T_fwd_vs_wavelength = np.real(trans_coeff * trans_coeff.conj() / source_power)
        self.phase_prefactors = trans_coeff / 8.0 / source_power
        fom = ModeMatch.fom_wavelength_integral(self.T_fwd_vs_wavelength, self.wavelengths, self.target_T_fwd, self.norm_p)
        return fom

    def get_adjoint_field_scaling(self, sim):
        omega = 2.0 * np.pi * sp.constants.speed_of_light / self.wavelengths
        adjoint_source_power = ModeMatch.get_source_power(sim, self.wavelengths)
        scaling_factor = np.conj(self.phase_prefactors) * omega * 1j / np.sqrt(adjoint_source_power)
        return scaling_factor

    @staticmethod
    def get_wavelengths(sim):
        return Wavelengths(sim.fdtd.getglobalsource('wavelength start'), 
                           sim.fdtd.getglobalsource('wavelength stop'),
                           sim.fdtd.getglobalmonitor('frequency points')).asarray()

    @staticmethod
    def get_transmission_coefficient(sim, direction, monitor_name, mode_exp_monitor_name):
        mode_exp_result_name = 'expansion for ' + mode_exp_monitor_name
        if not sim.fdtd.haveresult(mode_exp_monitor_name, mode_exp_result_name):
            raise UserWarning('unable to calcualte mode expansion.')
        mode_exp_data_set = sim.fdtd.getresult(mode_exp_monitor_name, mode_exp_result_name)
        fwd_trans_coeff = mode_exp_data_set['a'] * np.sqrt(mode_exp_data_set['N'].real)
        back_trans_coeff = mode_exp_data_set['b'] * np.sqrt(mode_exp_data_set['N'].real)
        if direction == 'Backward':
            fwd_trans_coeff, back_trans_coeff = back_trans_coeff, fwd_trans_coeff
        return fwd_trans_coeff.flatten()

    @staticmethod
    def get_source_power(sim, wavelengths):
        frequency = sp.constants.speed_of_light / wavelengths
        source_power = sim.fdtd.sourcepower(frequency)
        return np.asarray(source_power).flatten()

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

    def add_adjoint_sources(self, sim):
        adjoint_injection_direction = 'Backward' if self.direction == 'Forward' else 'Forward'
        ModeMatch.add_mode_source(sim, self.monitor_name, self.adjoint_source_name, adjoint_injection_direction, self.mode_number)

    @staticmethod
    def add_mode_source(sim, monitor_name, source_name, direction, mode_number):
        sim.fdtd.addmode()
        sim.fdtd.set('name', source_name)
        monitor_type = sim.fdtd.getnamed(monitor_name, 'monitor type')
        geo_props, normal = ModeMatch.cross_section_monitor_props(monitor_type)
        sim.fdtd.setnamed(source_name, 'injection axis', normal.lower() + '-axis')
        for prop_name in geo_props:
            prop_val = sim.fdtd.getnamed(monitor_name, prop_name)
            sim.fdtd.setnamed(source_name, prop_name, prop_val)
        sim.fdtd.setnamed(source_name, 'override global source settings', False)
        sim.fdtd.setnamed(source_name, 'direction', direction)
        sim.fdtd.setnamed(source_name, 'multifrequency mode calculation', True)
        sim.fdtd.setnamed(source_name, 'frequency points', sim.fdtd.getglobalmonitor('frequency points'))
        sim.fdtd.setnamed(source_name, 'mode selection', 'user select')
        sim.fdtd.select(source_name)
        sim.fdtd.updatesourcemode(mode_number)

    def fom_gradient_wavelength_integral(self, T_fwd_partial_derivs_vs_wl, wl):
        assert T_fwd_partial_derivs_vs_wl.shape[0] == wl.size
        assert np.allclose(wl, self.wavelengths)
        target_T_fwd_vs_wavelength = self.target_T_fwd(self.wavelengths).flatten()
        if self.wavelengths.size > 1:
            num_opt_param = T_fwd_partial_derivs_vs_wl.shape[1]
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
            T_fwd_partial_derivs = -1.0 * np.sign(self.T_fwd_vs_wavelength - target_T_fwd_vs_wavelength) * T_fwd_partial_derivs_vs_wl.flatten()
        return T_fwd_partial_derivs.flatten().real
