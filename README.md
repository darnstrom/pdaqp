**pdaqp** is a Python package for solving multi-parametric quadratic programs of the form

$$
\begin{align}
\min_{z} &  ~\frac{1}{2}z^{T}Hz+(f+F \theta)^{T}z \\
\text{s.t.} & ~A z \leq b + B \theta \\
& ~\theta \in \Theta
\end{align}
$$

where $H \succ 0$ and $\Theta \triangleq \lbrace l \leq \theta \leq u : A_{\theta} \theta \leq b_{\theta}\rbrace$.

**pdaqp** is based on the Julia package [ParametricDAQP.jl](https://github.com/darnstrom/ParametricDAQP.jl/) and the Python module [juliacall](https://juliapy.github.io/PythonCall.jl/stable/juliacall/). 

## Installation
```bash
pip install pdaqp
```
## Example
The following code solves the mpQP in Section 7.1 in Bemporad et al. 2002
```python
import pdaqp
import numpy

H =  numpy.array([[1.5064, 0.4838], [0.4838, 1.5258]])
f = numpy.zeros((2,1))
F = numpy.array([[9.6652, 5.2115], [7.0732, -7.0879]])
A = numpy.array([[1.0, 0], [-1, 0], [0, 1], [0, -1]])
b = 2*numpy.ones((4,1));
B = numpy.zeros((4,2));

thmin = -1.5*numpy.ones(2)
thmax = 1.5*numpy.ones(2)

from pdaqp import MPQP
mpQP = MPQP(H,f,F,A,b,B,thmin,thmax)
mpQP.solve()
```
To construct a binary search tree for point location, and to generate corresponding C-code, run 
```python
mpQP.codegen(dir="codegen", fname="pointlocation")
```
which will create the following directory:
```bash
├── codegen
│   ├── pointlocation.c
│   └── pointlocation.h
```
The critical regions and the optimal solution can be plotted with the commands
```python
mpQP.plot_regions()
mpQP.plot_solution()
```
