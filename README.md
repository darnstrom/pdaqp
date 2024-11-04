**pdaqp** is a Python module for solving multi-parametric quadratic programs of the form

$$
\begin{align}
\min_{x} &  ~\frac{1}{2}x^{T}Hx+(f+F \theta)^{T}x \\
\text{s.t.} & ~A x \leq b + B \theta \\
& ~\theta \in \Theta
\end{align}
$$

where $H \succ 0$ and $\Theta \triangleq \lbrace l \leq \theta \leq u : A_{\theta} \theta \leq b_{\theta}\rbrace$.

**pdaqp** is based on the Julia package [ParametricDAQP.jl](https://github.com/darnstrom/ParametricDAQP.jl/) and the Pyhon module [juliacall](https://juliapy.github.io/PythonCall.jl/stable/juliacall/). 

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
A list of the critical regions can be found in the field `CRs` of `mpQP`
```python
regs = pdaqp.critical_regions(sol)
```
To construct a binary search tree for point location, and to generate corresponding C-code, run 

```python
mpQP.codegen(dir="codegen", fname="pdaqp")
```
Which will create the following directory with generate C-code:
```bash
├── codegen
│   ├── pdaqp.c
│   └── pdaqp.h
```
