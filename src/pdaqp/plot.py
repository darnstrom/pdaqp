import numpy as np
from math import atan2

import warnings
try:
    import plotly.graph_objects as go
    plotly_available = True
except ImportError:
    plotly_available = False

def vrep_2d(A,b,tol = 1e-10):
    m,n = np.shape(A)
    vs = []
    if  m==0: return vs
    for i in range(m):
        for j in range(i+1,m):
            try:
                v = np.linalg.solve(A[[i,j],:],b[[i,j]])
            except:# is singular
                continue 
            if np.max(A@v-b) < tol:
                vs.append(v.flatten())
    # Sort in clockwise order
    c= np.mean(vs,axis=0)

    return sorted(vs, key=lambda v: atan2(v[1]-c[1], v[0]-c[0]))

def slice_region(A,b,ids,vals=None,tol=1e-10):
    if len(ids) == 0: return A,b
    m,n = np.shape(A)
    keep_ids= list(set(range(n))-set(ids))
    if vals is None: 
        vals = np.zeros(len(ids)) # default slice at 0
    Aslice,bslice = np.zeros((0,len(keep_ids))), np.zeros(0)
    zs = len(ids)
    for i in range(m):
        bi = b[i]-A[i,ids]@vals;
        print(bi)
        if not np.allclose(bi,zs,atol=tol):# Remove zero rows
            Aslice = np.vstack((Aslice,A[i,keep_ids]))
            bslice = np.append(bslice,bi)
        elif (bi < -tol):# infeasible
            return np.zeros((n-len(ids),0)),zeros(0) 
    return Aslice, bslice

def plot(CRs, out_id = None, fix_ids=None, fix_vals=None, plotly=False):
    if not plotly_available:
        warnings.warn("plotly not installed: plot_regions and plot_solution will not work")
        return 
    N = len(CRs) 
    if N == 0:
        print("No regions to plot")
        return
    nth= np.shape(CRs[0].Ath)[1]
    if fix_ids is None: fix_ids = range(2,nth) 
    if fix_vals is None: fix_vals = np.zeros(len(fix_ids))
    free_ids= list(set(range(nth))-set(fix_ids))
    nfree = nth-len(fix_ids)
    if not nfree ==2: 
        print("Can only plot 2D")
        return
    # Compute vertices of sliced region
    CRsV = [vrep_2d(*slice_region(cr.Ath,cr.bth,fix_ids,vals=fix_vals)) for cr in CRs]

    fig = go.Figure()
    if out_id is None: # Normal region plot
        for i,vs in enumerate(CRsV):
            xs = [v[0] for v in vs]
            ys = [v[1] for v in vs]
            fig.add_trace(go.Scatter(x=xs,y=ys, mode='lines',fill='toself', 
                                     name=f'<b>Region {i}</b><br>Active set: '+str(CRs[i].AS), 
                                     hoverinfo='text'))
        fig.update_layout(showlegend=False,
                          xaxis_title=r'$\huge\theta_{'+str(free_ids[0])+'}$',
                          yaxis_title=r'$\huge\theta_{'+str(free_ids[1])+'}$',
                          font=dict(size=28),
                          )
    else: # Plot the feedback
        for i,vs in enumerate(CRsV):
            xs = [v[0] for v in vs]
            ys = [v[1] for v in vs]
            c = CRs[i].z[out_id,fix_ids]@fix_vals + CRs[i].z[out_id,-1]
            zs = [c+CRs[i].z[out_id,free_ids]@v for v in vs] 
            fig.add_trace(go.Scatter3d(x=xs,y=ys,z=zs, mode='lines',
                                       surfaceaxis=2, hoverinfo='text',
                                       hovertext=f'<b>Region {i}</b><br>Active set: '+str(CRs[i].AS)))
        #fig.update_layout(zaxis_title=r'$x_{'+str(out_id)+'}$')
        fig.update_layout(showlegend=False,
                          scene=dict(
                              xaxis=dict(title='Parameter '+str(free_ids[0])),
                              yaxis=dict(title='Parameter '+str(free_ids[1])),
                              zaxis=dict(title='Minimizer '+str(out_id)),
                              )
                          )

    fig.show()
