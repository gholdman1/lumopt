""" Copyright chriskeraly
    Copyright (c) 2019 Lumerical Inc. """

import lumapi

class Simulation(object):
    """
        Object to manage the FDTD CAD. 

        Parameters
        ----------
        :param workingDir:    working directory for the CAD session.
        :param hide_fdtd_cad: if true, runs the FDTD CAD in the background.
        :param wq_config:     dict, if using WorkQueue, provides the configuration.
                                e.g. wq_config = dict(  {'scheduler':'slurm'},
                                                        {''},
                                                        {},
                                                        )                        
        """

    def __init__(self, workingDir, hide_fdtd_cad, wq_config=None):
        """ Launches FDTD CAD and stores a handle. """
        self.fdtd = lumapi.FDTD(hide = hide_fdtd_cad)
        self.fdtd.cd(workingDir)

        if wq_config:
            self.wq_config=wq_config
            

    def run(self, name, iter):
        """ Saves simulation file and runs the simulation. """
        filename='{}_{}'.format(name,iter)
        self.fdtd.save(filename)
        self.fdtd.run()

    def __del__(self):
        self.fdtd.close()
