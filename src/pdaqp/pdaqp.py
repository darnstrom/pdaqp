import numpy as np
from collections import namedtuple
from dataclasses import dataclass

from types import ModuleType
from typing import cast
from juliacall import Main as jl
from juliacall import AnyValue
from .plot import plot 

jl = cast(ModuleType, jl)
jl_version = (jl.VERSION.major, jl.VERSION.minor, jl.VERSION.patch)

jl.seval("using ParametricDAQP")
ParametricDAQP = jl.ParametricDAQP

MPQPDATA = namedtuple('MPQPDATA',['H','f','F','A','b','B','bounds_table','out_inds','eq_ids'])
TH = namedtuple('TH', ['lb', 'ub'])

@dataclass
class CriticalRegion:
    Ath: np.ndarray 
    bth: np.ndarray 
    z: np.ndarray
    lam: np.ndarray
    AS: np.ndarray

@dataclass
class BSTNode:
    """ A node in a binary search tree

    Attributes:
        parent_id: id of parent
        left_id: id of left child (corresponds to affine_mapping*(theta,-1) <= 0)
        right_id: id of left child (corresponds to affine_mapping*(theta,-1) >= 0)
        affine_mapping: if leaf node maps (theta,1) -> solution,
            if not leaf node if defines the half plane that determines the branching

    """
    parent_id: int
    left_id: int
    right_id: int
    affine_mapping: np.ndarray

@dataclass
class BinarySearchTree:
    """ A representation of the binary search tree in Python
    used for point location

    Attributes:
        nodes: List of nodes that comprise the tree
        depth: the depth of the tree
        leaf_ids: ids of nodes that are leaf nodes
        jl_bst: the pure Julia representation of the bst
    """
    nodes: list
    depth: int
    leaf_ids: list
    jl_bst : AnyValue

    def codegen(self, dir="codegen",fname="pdaqp", float_type="float", int_type="unsigned short"):
        """Generates C-code for performing the pointlocation.

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
        ParametricDAQP.codegen(self.jl_bst,dir=dir,fname=fname,
                               float_type=float_type, int_type=int_type)
    def evaluate(self,parameter):
        i = 0
        while self.nodes[i].left_id is not None : # Not a leaf node yet
            if self.nodes[i].affine_mapping@np.append(parameter,-1) <= 0:
                i = self.nodes[i].left_id
            else:
                i = self.nodes[i].right_id
        return self.nodes[i].affine_mapping@np.append(parameter,1)



class MPQP:
    """ A Multi-parametric quadratic program

    minimize_x  0.5 x' H x + (f + F theta)'x
    subject to  A x <= b + B theta

    The parameter theta is in the set
    TH0 defined by  thmin<= theta <=thmax

    Attributes:
        mpQP: The problem data H,f,F,A,b,B that defines the mpQP
        TH0: The bounds thmin,thmax that defines the set of parameters of interest
        CRs: The critical regions that comprise the solution of the mpQP
        solution: The solution structure directly from Julia
        solution_info: Information from the solving process
    """
    mpQP:MPQPDATA
    TH0:TH
    CRs:list
    solution:AnyValue
    solution_info:AnyValue
    def __init__(self, H,f,F,A,b,B, thmin,thmax, bounds_table=None, out_inds=None, eq_inds=None):
        if out_inds is not None: out_inds = [i+1 for i in out_inds]
        if eq_inds is not None: eq_inds = [i+1 for i in eq_inds]

        self.mpQP = MPQPDATA(H,f,F,A,b,B,bounds_table,out_inds,eq_inds)
        self.TH0  = TH(thmin,thmax)
        CRs = None
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

    def build_tree(self):
        bst = ParametricDAQP.build_tree(self.solution)
        hps = np.array(bst.halfplanes,copy=False, order='F').T
        hp_list = np.array(bst.hp_list,copy=True)-1
        jump_list = np.array(bst.jump_list,copy=True)-1

        nodes = [ BSTNode(None,None,None,None) for i in range(len(hp_list))]
        leaf_ids = []

        for i in range(len(hp_list)):
            if jump_list[i] == 0: # Is a leaf node
                nodes[i].affine_mapping =np.array(bst.feedbacks[hp_list[i]],copy=False, order='F').T
                leaf_ids.append(i)
            else:
                rid,lid = jump_list[i],jump_list[i]+1
                nodes[i].right_id,nodes[i].left_id  = rid,lid
                nodes[i].affine_mapping = hps[hp_list[i],:]
                nodes[rid].parent_id, nodes[lid].parent_id = i,i

        return BinarySearchTree(nodes,bst.depth,leaf_ids,bst)
