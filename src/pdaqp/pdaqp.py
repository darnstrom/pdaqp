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

    def solve(self,settings=None):
        self.solution,self.solution_info = ParametricDAQP.mpsolve(self.mpQP,self.TH0,opts=settings)
        self.CRs = [CriticalRegion(np.array(cr.Ath,copy=False, order='F').T,
                             np.array(cr.bth,copy=False),
                             np.array(cr.z,copy=False, order='F').T,
                             np.array(cr.lam,copy=False, order='F').T,
                             np.array(cr.AS)-1
                             ) for cr in ParametricDAQP.get_critical_regions(self.solution)] 

    def plot_regions(self, fix_ids = None, fix_vals = None,backend='tikz'):
        if backend == 'tikz':
            jl.display(ParametricDAQP.plot_regions(self.solution,fix_ids=fix_ids,fix_vals=fix_vals))
        elif backend == 'plotly':
            plot(self.CRs, fix_ids=fix_ids,fix_vals=fix_vals)
        else:
            print('Plotting backend '+backend+ ' unknown')

    def plot_solution(self, z_id=0,fix_ids = None, fix_vals = None,backend='tikz'):
        if backend == 'tikz':
            jl.display(ParametricDAQP.plot_solution(self.solution,z_id=z_id+1,fix_ids=fix_ids,fix_vals=fix_vals))
        elif backend == 'plotly':
            plot(self.CRs, out_id =z_id,fix_ids=fix_ids,fix_vals=fix_vals)
        else:
            print('Plotting backend '+backend+ ' unknown')

    def codegen(self, dir="codegen",fname="pdaqp", float_type="float", int_type="unsigned short"):
        ParametricDAQP.codegen(self.solution,dir=dir,fname=fname, 
                               float_type=float_type, int_type=int_type)
