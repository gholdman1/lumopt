""" Copyright chriskeraly
    Copyright (c) 2019 Lumerical Inc. """

import os, sys
from shutil import copy2
import string, random
import lumapi

class Simulation(object):
    """
        Object to manage the FDTD CAD. 

        Parameters
        ----------
        :param workingDir:    working directory for the CAD session.
        :param hide_fdtd_cad: if true, runs the FDTD CAD in the background.
        :param wq_config:     dict, if using WorkQueue, provides the
                                configuration.
                                e.g. wq_config = dict({ 'scheduler':'slurm',
                                                        'cores': 4,
                                                        'memory':1024,
                                                        'disk':1024,
                                                        ''
                                                    })                        
        """

    def __init__(self, workingDir, hide_fdtd_cad, wq_config=None):
        """ Launches FDTD CAD and stores a handle. """
        self.fdtd = lumapi.FDTD(hide = hide_fdtd_cad)
        self.fdtd.cd(workingDir)

        if wq_config:
            self.wq_config=wq_config
            self.workqueue_initialize()



    def run(self, name, iter):
        """ Saves simulation file and runs the simulation. """
        filename='{}_{}'.format(name,iter)
        self.fdtd.save(filename)

        if self.wq_config:
            self.workqueue_run()
        else:
            self.fdtd.run()

    def __del__(self):
        self.fdtd.close()

    def workqueue_initialize(self):
        import work_queue as WQ
        import string, random

        # Copy shell script to be run on remote server
        script_src=os.path.join(lumopt.CONFIG['root'],
                                self.wq_config['scheduler'],
                                'run_lumerical.sh')
        script_dst='.'
        copy2(script_src,script_dst)

        # Start a master
        self.wq_config['master']= 'lumopt-'+Simulation.workqueue_master_id()
        print('Initializing Master ',self.wq_config['master'])
        self.wq_config['q']=WQ.WorkQueue(name=self.wq_config['master'],
                                        port=0)
        
        print('Master initialized. Go to your',self.wq_config['scheduler'],
                '-enabled account and run the following command:')
        print(Simulation.submit_worker_command( self.wq_config['scheduler'],
                                                self.wq_config['cores'],
                                                self.wq_config['memory'],
                                                self.wq_config['disk'],
                                                self.wq_config['master']))
        



    def workqueue_run(self,filename):
        fsp_in = filename
        fsp_out= fsp_in+'.out'

        t = WQ.Task('./run_lumerical.sh '+
                    str(self.wq_config['cores'] + ' '+
                    fsp_in + ' ' +
                    fsp_out))

        t.specify_input_file('run_lumerical.sh',cache=True)
        t.specify_input_file(fsp_in)
        t.specify_output_file(fsp_out)

        self.wq_config['q'].submit(t)
        print('Simulation submitted. Waiting.')

        while not self.wq_config['q'].empty():
            t = q.wait(10)
            if t:
                print('Simulation returned')
        print('Saving simulation')


    @staticmethod
    def workqueue_master_id(length=6,chars=string.ascii_uppercase+string.ascii_lowercase):
        return ''.join(random.choice(chars) for _ in range(length))

    @staticmethod
    def submit_worker_command(scheduler,cores,memory,disk,master):

        if scheduler=='slurm':
            command='> slurm_submit_workers --cores '   + str(cores) +
                                            '--memory ' + str(memory)+
                                            '--disk '   + str(disk) +
                                            '-M '       + str(master)+
                                            '-t '       + str(2000)+
                                            '-p '       + '\"-p univ2\" ' +
                                            '1'
        if scheduler=='condor':
            command='UNKNOWN'

        return command