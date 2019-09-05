import lumopt.lumerical_methods.lumerical_scripts as ls
import numpy as np
import lumapi

from lumopt.utilities.wavelengths import Wavelengths

class FieldIntensity(object):
	'''
	A figure of merit which is simply |E|^2 at a point monitor defined in the base simulation
	'''

	def __init__(self, monitor_name,wavelengths):
		'''
		:param monitor_name: A string: the name of the point monitor
		:param wavelengths: A list of the wavelengths of interest (for the moment supports only a single value)
		'''
		self.monitor_name = monitor_name
		self.wavelengths = wavelengths
		self.multi_freq_src=False
		self.adjoint_source_name='dipole_src'

	def initialize(self,sim):
		self.add_adjoint_sources(sim)

	def make_forward_sim(self, sim):

		for i in range(len(self.dipole_x)):
			for orientation in ['x','y','z']:
				sim.fdtd.setnamed(self.adjoint_source_name+'_'+str(i)+'_'+orientation,
							'enabled', False)

	def make_adjoint_sim(self,sim):
		oris=['x','y','z']
		for j in range(len(self.dipole_x)):
			for i in range(len(oris)):
				ori=oris[i]
				src=self.adjoint_source_name+'_'+str(j)+'_'+ori

				Ec=np.conj(self.forward_field['E'].squeeze())
				print('Ec',Ec.shape)
				amplitude=np.abs(Ec[j,i])
				phase= np.angle(Ec[j,i])*(360/(2*np.pi))
				print('Source ',src)
				print('Amplitude ',amplitude)
				print('Phase',phase)
				sim.fdtd.setnamed(src,'amplitude',amplitude)
				sim.fdtd.setnamed(src,'phase',phase)
				sim.fdtd.setnamed(src,
								'enabled', True)

	def get_fom(self, sim):
		'''
		:param simulation: The simulation object of the base simulation
		:return: The figure of merit
		'''
		field = sim.fdtd.getresult(self.monitor_name,'E')
		self.wavelengths = FieldIntensity.get_wavelengths(sim)
		self.forward_field=field
		fom = np.linalg.norm(field['E'].squeeze())**2
		return fom

	def add_adjoint_sources(self, sim):
		'''
		Adds the adjoint sources required in the adjoint simulation
		:param simulation: The simulation object of the base simulation
		'''

		# Mesh positions
		mesh_x=sim.fdtd.getresult('FDTD','x')
		mesh_y=sim.fdtd.getresult('FDTD','y')

		# Bounds of monitor
		sim.fdtd.select(self.monitor_name)
		monitor_xmin=sim.fdtd.get('x min')
		monitor_xmax=sim.fdtd.get('x max')
		self.dipole_y=mesh_y[np.argmin(np.abs(sim.fdtd.get('y')-mesh_y))]
		self.dipole_z=0


		greater,lesser=mesh_x>monitor_xmin,mesh_x<monitor_xmax
		self.dipole_x=mesh_x[greater & lesser]

		for i in range(len(self.dipole_x)):
			x = self.dipole_x[i]
			pos=[x,self.dipole_y,self.dipole_z]
			for orientation in ['x','y','z']:
				FieldIntensity.add_dipole_source(sim,
								pos,
								self.adjoint_source_name+'_'+str(i)+'_'+orientation,
								orientation)
	
	def get_adjoint_field_scaling(self,sim):
		return np.array([1])

	def fom_gradient_wavelength_integral(self, E_partial_derivs_vs_wl, wl):
		assert np.allclose(wl, self.wavelengths)
		return FieldIntensity.fom_gradient_wavelength_integral_impl(E_partial_derivs_vs_wl, self.wavelengths)

	@staticmethod
	def fom_gradient_wavelength_integral_impl(E_partial_derivs_vs_wl, wl):

		if wl.size > 1:
			assert T_fwd_partial_derivs_vs_wl.shape[1] == wl.size
			
			wavelength_range = wl.max() - wl.min()
			T_fwd_error = T_fwd_vs_wavelength - target_T_fwd_vs_wavelength
			T_fwd_error_integrand = np.power(np.abs(T_fwd_error), norm_p) / wavelength_range
			const_factor = -1.0 * np.power(np.trapz(y = T_fwd_error_integrand, x = wl), 1.0 / norm_p - 1.0)
			integral_kernel = np.power(np.abs(T_fwd_error), norm_p - 1) * np.sign(T_fwd_error) / wavelength_range
			
			## Implement the trapezoidal integration as a matrix-vector-product for performance reasons
			d = np.diff(wl)
			quad_weight = np.append(np.append(d[0], d[0:-1]+d[1:]),d[-1])/2 #< There is probably a more elegant way to do this
			v = const_factor * integral_kernel * quad_weight
			T_fwd_partial_derivs = T_fwd_partial_derivs_vs_wl.dot(v)

			## This is the much slower (but possibly more readable) code
			# num_opt_param = T_fwd_partial_derivs_vs_wl.shape[0]
			# T_fwd_partial_derivs = np.zeros(num_opt_param, dtype = 'complex')
			# for i in range(num_opt_param):
			#     T_fwd_partial_deriv = np.take(T_fwd_partial_derivs_vs_wl.transpose(), indices = i, axis = 1)
			#     T_fwd_partial_derivs[i] = const_factor * np.trapz(y = integral_kernel * T_fwd_partial_deriv, x = wl)
		else:
			E_fwd_partial_derivs = E_partial_derivs_vs_wl.flatten()

		return E_fwd_partial_derivs.flatten().real

	@staticmethod
	def add_dipole_source(sim,pos,source_name,orientation):
		'''
		
		orientation: 'x','y',or 'z'
		'''



		sim.fdtd.adddipole()
		src=source_name
		sim.fdtd.set('name',src)
		sim.fdtd.setnamed(src,'x',pos[0])
		sim.fdtd.setnamed(src,'y',pos[1])
		sim.fdtd.setnamed(src,'z',pos[2])

		if(orientation=='x'):
			sim.fdtd.setnamed(src,'theta',90)
		if(orientation=='y'):
			sim.fdtd.setnamed(src,'phi',90)
			sim.fdtd.setnamed(src,'theta',90)

	def check_monitor_alignment(self, sim):
	  
		## Here, we check that the FOM_monitor is properly aligned with the mesh
		if sim.fdtd.getnamednumber(self.monitor_name) != 1:
			raise UserWarning('monitor could not be found or the specified name is not unique.')
		
		# Get the orientation
		monitor_type = sim.fdtd.getnamed(self.monitor_name, 'monitor type')

		if (monitor_type == 'Linear X') or (monitor_type == '2D X-normal'):
			orientation = 'x'
		elif (monitor_type == 'Linear Y') or (monitor_type == '2D Y-normal'):
			orientation = 'y'
		elif (monitor_type == 'Linear Z') or (monitor_type == '2D Z-normal'):
			orientation = 'z'
		else:
			raise UserWarning('monitor should be 2D or linear for a mode expansion to be meaningful.')

		monitor_pos = sim.fdtd.getnamed(self.monitor_name, orientation)
		if sim.fdtd.getnamednumber('FDTD') == 1:
			grid = sim.fdtd.getresult('FDTD', orientation)
		elif sim.fdtd.getnamednumber('varFDTD') == 1:
			grid = sim.fdtd.getresult('varFDTD', orientation)
		else:
			raise UserWarning('no FDTD or varFDTD solver object could be found.')
		## Check if this is exactly aligned with the simulation mesh. It exactly aligns if we find a point
		## along the grid which is no more than 'tol' away from the position
		tol = 1e-9
		if min(abs(grid-monitor_pos)) > tol:
			print('WARNING: The monitor "{}" is not aligned with the grid. This can introduce small phase errors which sometimes result in inaccurate gradients.'.format(self.monitor_name))

	@staticmethod
	def get_wavelengths(sim):
		return Wavelengths(sim.fdtd.getglobalsource('wavelength start'), 
						   sim.fdtd.getglobalsource('wavelength stop'),
						   sim.fdtd.getglobalmonitor('frequency points')).asarray()


class FieldIntensities(object):
	'''
	A slightly more complex figure of merit than FieldIntensity.
	Now fields at different points in space can be combined linearly
	to create a figure of merit of the form:

		FOM= |a1*E1+ a2*E2 + ... + an*En|^2

	where ai can be a complex vector of length 3.
	With the right amplitudes ai, this can be used to form a wide
	variety of complex meaningful figures of merit (absorption, modematching).

	If the amplitudes are set to None, then
	FOM=|E1|^2+...+|En|^2
	'''

	def __init__(self, monitor_names, weight_amplitudes=[1, 1], wavelengths=[1550e-9], normalize_to_source_power=False):
		'''
		:param positions: A list of tuples representing the 3D coordinate in space of where the fields should be measured
		:param weight_amplitudes: A list of complex amplitudes
		:param wavelengths: The wavelengths of interest (for the moment supports only one wavelength)
		:param normalize_to_source_power: Should everything be normalized to the source power?
		'''
		self.positions=positions
		self.weight_amplitudes=weight_amplitudes
		self.wavelengths=wavelengths
		self.current_fom=None
		self.fields=None
		self.normalize_to_source_power=normalize_to_source_power
		self.monitor_names=['fom_mon_{}'.format(i) for i in range(len(self.positions))]

		self.multi_freq_src=False

	def put_monitors(self,simulation):
		script=''
		for monitor_name,position in zip(self.monitor_names,self.positions):
			script+=ls.add_point_monitor_script(monitor_name,position)
		sim.fdtd.eval(script)

	def get_fom(self,simulation):
		fields=[ls.get_fields(simulation.solver_handle, monitor_name) for monitor_name in self.monitor_names]
		if self.normalize_to_source_power:
			source_power = np.zeros(np.shape(fields[0].wl))
			for i, wl in enumerate(fields[0].wl):
				source_power[i] = ls.get_source_power(simulation.solver_handle, wl=wl)
				self.source_power = source_power
				self.fields=fields
				pointfields=[field.getfield(field.x[0],field.y[0],field.z[0],self.wavelengths[0]) for field in fields]
		if self.weight_amplitudes is None:
			fom=sum([sum(pointfield*np.conj(pointfield)) for pointfield in pointfields])
		else:
			sum_of_pointfields=sum([pointfield*phase_factor for pointfield,phase_factor in zip(pointfields, self.weight_amplitudes)])
			fom=sum(sum_of_pointfields*np.conj(sum_of_pointfields))
		if self.normalize_to_source_power:
			fom=fom/np.array(source_power)
		return fom

	def add_adjoint_sources(self, sim):

		fields=self.fields
		pointfields=[field.getfield(field.x[0],field.y[0],field.z[0],self.wavelengths[0]) for field in fields]
		prefactor=1#eps0#*omega

		#print prefactor
		if self.weight_amplitudes is None:
			adjoint_sources=[prefactor*np.conj(pointfield) for pointfield in pointfields]
		else:
			pointfields = [field.getfield(field.x[0], field.y[0], field.z[0], self.wavelengths[0]) for field in fields]
			sum_of_pointfields=sum([pointfield*phase_factor for pointfield,phase_factor in zip(pointfields, self.weight_amplitudes)])
			prefactor=np.conj(sum_of_pointfields)
			adjoint_sources=[prefactor*phase_factor for phase_factor in self.weight_amplitudes]
		if self.normalize_to_source_power:
			adjoint_sources=adjoint_sources/self.source_power
			script=''
		for i,(adjoint_source,field) in enumerate(zip(adjoint_sources,fields)):
			script+=ls.add_dipole_script(field.x[0],field.y[0],field.z[0],self.wavelengths[0],adjoint_source,name_suffix=str(i))
			
		sim.fdtd.eval(script)
		return

	@staticmethod
	def add_dipole(sim,monitor_name,source_name):
		'''
		Add dipole at location of monitor
		'''
		pass




