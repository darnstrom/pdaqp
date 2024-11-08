import numpy as np
from collections import namedtuple
from dataclasses import dataclass

from types import ModuleType
from typing import cast
from juliacall import Main as jl
from .plot import plot 

jl = cast(ModuleType, jl)
jl_version = (jl.VERSION.major, jl.VERSION.minor, jl.VERSION.patch)

jl.seval("using ParametricDAQP")
ParametricDAQP = jl.ParametricDAQP

MPQPDATA = namedtuple('MPQPDATA',['H','f','F','A','b','B','bounds_table','out_inds'])
TH = namedtuple('TH', ['lb', 'ub'])

@dataclass
class CriticalRegion:
    Ath: np.ndarray 
    bth: np.ndarray 
    z: np.ndarray
    lam: np.ndarray
    AS: np.ndarray

class MPQP:
    mpQP:MPQPDATA
    TH0:TH
    def __init__(self, H,f,F,A,b,B, thmin,thmax, bounds_table=None, out_inds=None):
        self.mpQP = MPQPDATA(H,f,F,A,b,B,bounds_table,out_inds)
        self.TH0  = TH(thmin,thmax)
        self.solution = None
        self.solution_info = None

    def solve(self,settings=None):
        """ Computes the explicit solution to the mpQP.

        The critical regions are stored in the variable CRs. 
        Information from the solving process is stored in 
        the variable solution_info. 
        The internal Julia solution struct is stored 
        in the variable solution
        """
        self.solution,self.solution_info = ParametricDAQP.mpsolve(self.mpQP,self.TH0,opts=settings)
        self.CRs = [CriticalRegion(np.array(cr.Ath,copy=False, order='F').T,
                             np.array(cr.bth,copy=False),
                             np.array(cr.z,copy=False, order='F').T,
                             np.array(cr.lam,copy=False, order='F').T,
                             np.array(cr.AS)-1
                             ) for cr in ParametricDAQP.get_critical_regions(self.solution)] 

    def plot_regions(self, fix_ids = None, fix_vals = None,backend='tikz'):
        """ A 2D plot of the critical regions of the solution to the mpQP.

        Args:
            fix_ids: ids of parameters to fix. Defaults to all ids except 
              the first and second. 
            fix_vals: Corresponding values for the fixed parameters. Defaults to 0. 
            backend: Determine if tikz or plotly should be used as plotting backend. 
              Defaults to tikz (which is what ParametricDAQP.jl uses)
        """
        if backend == 'tikz':
            jl.display(ParametricDAQP.plot_regions(self.solution,fix_ids=fix_ids,fix_vals=fix_vals))
        elif backend == 'plotly':
            plot(self.CRs, fix_ids=fix_ids,fix_vals=fix_vals)
        else:
            print('Plotting backend '+backend+ ' unknown')

    def plot_solution(self, z_id=0,fix_ids = None, fix_vals = None,backend='tikz'):
        """ A 3D plot of component z_id of the solution to the mpQP.

        Args:
            z_id: id of the component of the solution to plot. 
              Defaults to the first component
            fix_ids: ids of parameters to fix. Defaults to all ids except 
              the first and second. 
            fix_vals: Corresponding values for the fixed parameters. Defaults to 0. 
            backend: Determine if tikz or plotly should be used as plotting backend. 
              Defaults to tikz (which is what ParametricDAQP.jl uses.)
        """
        if backend == 'tikz':
            jl.display(ParametricDAQP.plot_solution(self.solution,z_id=z_id+1,fix_ids=fix_ids,fix_vals=fix_vals))
        elif backend == 'plotly':
            plot(self.CRs, out_id =z_id,fix_ids=fix_ids,fix_vals=fix_vals)
        else:
            print('Plotting backend '+backend+ ' unknown')

    def codegen(self, dir="codegen",fname="pdaqp", float_type="float", int_type="unsigned short"):
        """ Forms a binary search tree and generates C-code for performing the pointlocation.

        In the generated .c contains data for the binary search and the function
        {fname}_evaluate(parameter,solution) which performs the point location 
          "parameter" is the parameter theta (a pointer to a float array)
          "solution" is where the solution z is stored  (a pointer to a float array)

        Args:
            dir: directory where the generated code should be stored. 
            fname: name of the .c and .h files. Also serves as a prefix in the generated code. 
            float_type: type of floating point number that is used in the C-code. 
            int_type: type of integer that is used in the C-code.
        """
        ParametricDAQP.codegen(self.solution,dir=dir,fname=fname, 
                               float_type=float_type, int_type=int_type)
