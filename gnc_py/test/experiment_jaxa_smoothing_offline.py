"""Smoothing comparison for the JAXA baseline; no ROS nodes or production changes.

Representative selection and control-point fitting are reconstruction assumptions,
not algorithms specified by the IAC-22 paper. All candidates get the same raw paths.
``--retry`` mirrors production jaxa_plan_local_path: attempt a uses RRT* seed+a,
a collided smoothed path retries, and an RRT* failure ends the plan.
"""
import argparse
import csv
import json
from pathlib import Path
import time
import zlib

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
from scipy.interpolate import BSpline, make_interp_spline, splprep
import sobits_intball2_gnc_cpp as core

import experiment_global_planner_minco_jem_cases as base
from experiment_jaxa_baseline_offline import paper_scenario
from sobits_intball2_gnc.guidance.global_planner.path_shortcut import point_is_free, segment_is_free, shortcut_path
from sobits_intball2_gnc.guidance.local_planner.jaxa_rrt_local_planner import (
    JaxaPlanError, RRTStar, bspline_waypoints, path_is_free,
)

METHODS = ('all_points_interpolation', 'representative_interpolation', 'raw_control_points',
           'representative_control_points')
# All-RRT*-node methods contradict Fig.4(b) "simplified" and Fig.9; kept only as references.
CANDIDATES = ('representative_interpolation', 'representative_control_points')
SPACINGS = (.5, .2, .1, .05)
PRODUCTION_SPACING = .5
MAX_ATTEMPTS = 5
# Collision sampling step for shortcut and acceptance. Production pathFree and RRT* edges use
# half a voxel; 1 cm (the fixed-input default) rejects 7/40 raw RRT* paths that clip voxel corners.
CHECK_STEP = .01


def distinct(points):
    pts = np.asarray(points, dtype=float)
    if pts.ndim != 2 or pts.shape[1] != 3 or len(pts) < 2 or not np.all(np.isfinite(pts)):
        raise ValueError('expected finite (N,3) points')
    pts = pts[np.r_[True, np.linalg.norm(np.diff(pts,axis=0),axis=1) > 1e-9]]
    if len(pts) < 2:
        raise ValueError('need two distinct points')
    return pts


def make_curve(raw, method, grid, bounds):
    pts = distinct(raw)
    if method in ('representative_interpolation', 'representative_control_points'):
        pts = distinct(shortcut_path(pts, grid, bounds, collision_step=CHECK_STEP))
    chord = np.linalg.norm(np.diff(pts,axis=0),axis=1)
    u = np.r_[0., np.cumsum(chord)] / chord.sum()
    degree = min(3,len(pts)-1)
    if method in ('all_points_interpolation', 'representative_interpolation'):
        curve = make_interp_spline(u,pts,k=degree)
    elif method in ('raw_control_points', 'representative_control_points'):
        inner = [np.mean(u[j:j+degree]) for j in range(1,len(pts)-degree)]
        knots = np.r_[np.zeros(degree+1),inner,np.ones(degree+1)]
        curve = BSpline(knots,pts,degree)
    else:
        raise ValueError(method)
    return curve, pts, float(chord.sum())


def dense_samples(curve, step=.01):
    # Nonnegative B-spline bases bound speed by the largest derivative coefficient.
    derivative = curve.derivative(1)
    speed_bound = float(np.linalg.norm(derivative.c,axis=1).max())
    count = max(2,int(np.ceil(speed_bound/step))+1)
    if count > 200000:
        raise ValueError('derivative bound requires too many samples')
    u = np.linspace(0,1,count)
    return u, curve(u)


def corners(points):
    delta = np.diff(points,axis=0)
    length = np.linalg.norm(delta,axis=1)
    keep = length > 1e-12
    delta,length = delta[keep],length[keep]
    if len(delta)<2:
        return 0.
    cos = np.sum(delta[:-1]*delta[1:],axis=1)/(length[:-1]*length[1:])
    return float(np.degrees(np.arccos(np.clip(cos,-1,1))).max())


def tracking_switch(points, positions, lookahead=.11):
    nearest = np.argmin(np.linalg.norm(points[None,:,:]-positions[:,None,:],axis=2),axis=1)
    index = np.minimum(nearest,len(points)-2)
    a,b = points[index],points[index+1]
    delta = b-a
    norm = np.linalg.norm(delta,axis=1)
    unit = delta/np.maximum(norm[:,None],1e-12)
    foot = a-np.sum((a-positions)*unit,axis=1)[:,None]*unit
    target = foot+lookahead*unit
    clamp = (index==len(points)-2) & (np.sum((target-b)*unit,axis=1)>0)
    target[clamp | (norm<1e-6)] = b[clamp | (norm<1e-6)]
    change = index[1:] != index[:-1]
    shifts = np.linalg.norm(np.diff(target,axis=0),axis=1)[change]
    return float(shifts.max()) if len(shifts) else 0.


def box_clearance(points, boxes):
    values=[]
    for center,half in boxes:
        q = np.abs(points-center)-half
        distance = np.linalg.norm(np.maximum(q,0),axis=1)+np.minimum(q.max(axis=1),0)-.1
        values.append(float(distance.min()))
    return min(values)


def curve_metrics(curve, dense, grid, bounds, boxes):
    # Pointwise derivatives: corner_spline's double knots make derivative(2) undefined at breaks.
    u = np.linspace(0,1,len(dense))
    velocity = curve(u,nu=1)
    speed = np.linalg.norm(velocity,axis=1)
    curvature = np.zeros(len(u)) if curve.k<2 else (
        np.linalg.norm(np.cross(velocity,curve(u,nu=2)),axis=1)/np.maximum(speed,1e-12)**3)
    lower,upper=np.array(bounds)
    return dict(curve_length_m=float(np.linalg.norm(np.diff(dense,axis=0),axis=1).sum()),
                dense_free=bool(path_is_free(dense,grid,bounds)),
                curve_in_bounds=bool(np.all(dense>=lower) and np.all(dense<=upper)),
                min_box_clearance_m=box_clearance(dense,boxes),
                max_curvature_per_m=float(curvature.max()),
                dense_max_step_m=float(np.linalg.norm(np.diff(dense,axis=0),axis=1).max()))


def fig4_case():
    grid=core.OccupancyGrid(.1,.2)
    boxes=[(np.array([3.2,3.9,0]),np.array([1.,.8,.3]))]
    for center,half in boxes:
        grid.add_box(center.tolist(),half.tolist())
    raw=np.array([[2,.5,0],[2,1,0],[.8,1,0],[.8,1.8,0],[3.8,1.8,0],
                  [3.8,2.4,0],[.8,2.4,0],[.8,5.5,0],[4.8,5.5,0],[4.8,6.5,0]])
    return dict(case='fig4_like',seed=-1,raw=raw,grid=grid,
                bounds=([-2,-1,-1],[8,8,1]),boxes=boxes)


FIG4_CELL_M = .3  # assumed size of one Fig.4 grid cell
# Cells read from the paper's Fig.4 (7 columns from the left, 15 rows from the top).
FIG4_A_CELLS = ((4,15),(4,14),(4,13),(3,13),(2,13),(2,12),(2,11),(3,11),(4,11),(5,11),(5,10),(5,9),(4,9),(3,9),
                (2,9),(2,8),(2,7),(2,6),(2,5),(2,4),(2,3),(2,2),(3,2),(4,2),(5,2),(6,2),(6,1))
FIG4_B_CELLS = frozenset({(4,15),(4,14),(3,13),(3,12),(3,11),(3,10),(3,9),(2,8),(2,7),(2,6),(2,5),(2,4),(2,3),
                          (3,2),(4,2),(5,2),(6,1)})


def fig4_paper_case():
    """Fig.4(a) rebuilt from its cells; the obstacle box is read from the figure."""
    c = FIG4_CELL_M
    raw = np.array([[(col-.5)*c,(15.5-row)*c,0.] for col,row in FIG4_A_CELLS])
    lo,hi = np.array([3.5,7.7,-1.])*c,np.array([7.9,10.1,1.])*c
    center,half = (lo+hi)/2,(hi-lo)/2
    grid = core.OccupancyGrid(.05,.2)
    grid.add_box(center.tolist(),half.tolist())
    return dict(case='fig4_paper',seed=-1,raw=raw,grid=grid,bounds=([0,0,-1],[7*c,15*c,1]),
                boxes=[(center,half)])


def fig4_cells(dense):
    c = FIG4_CELL_M
    return {(int(np.floor(x/c))+1,15-int(np.floor(y/c))) for x,y in dense[:,:2]}


def plot_fig4_cells(case, curves, output):
    c = FIG4_CELL_M
    panels = [('Fig.4(b) cells',None,FIG4_B_CELLS)]+[
        (f'{name}\noverlap {len(fig4_cells(d)&FIG4_B_CELLS)/len(fig4_cells(d)|FIG4_B_CELLS):.2f}',d,fig4_cells(d))
        for name,d in curves.items()]
    fig,axes = plt.subplots(1,len(panels),figsize=(2.6*len(panels),6),constrained_layout=True)
    (center,half), = case['boxes']
    for ax,(name,dense,cells) in zip(axes,panels):
        for col,row in cells:
            ax.add_patch(Rectangle(((col-1)*c,(15-row)*c),c,c,color='tab:orange',alpha=.35))
        ax.add_patch(Rectangle((center-half)[:2],*(2*half)[:2],color='.4'))
        ax.plot(case['raw'][:,0],case['raw'][:,1],'--',color='.5',lw=1)
        if dense is not None:
            ax.plot(dense[:,0],dense[:,1],color='tab:blue',lw=2)
        ax.set_xticks(np.arange(0,7*c+1e-9,c));ax.set_yticks(np.arange(0,15*c+1e-9,c));ax.grid(True,lw=.5)
        ax.set_xticklabels([]);ax.set_yticklabels([]);ax.set_xlim(0,7*c);ax.set_ylim(0,15*c)
        ax.set_aspect('equal');ax.set_title(name,fontsize=8)
    fig.savefig(output,dpi=120)
    plt.close(fig)


def plot_case(case, curves, output, methods=METHODS):
    fig,axes=plt.subplots(1,len(methods),figsize=(4.4*len(methods),5),constrained_layout=True)
    dims=(0,1) if case['case']=='fig4_like' else (1,0)
    for ax,method in zip(np.atleast_1d(axes),methods):
        raw=curves[method][4] if len(curves[method])>4 else case['raw']
        curve,points,dense,valid=curves[method][:4]
        ax.plot(raw[:,dims[0]],raw[:,dims[1]],'--',color='.55',lw=1,label='fixed input path')
        ax.plot(dense[:,dims[0]],dense[:,dims[1]],color='tab:blue' if valid else 'tab:red',lw=2,label='B-spline')
        ax.scatter(points[:,dims[0]],points[:,dims[1]],s=15,color='black',label='fit / control points')
        count=max(2,int(np.ceil(np.linalg.norm(np.diff(points,axis=0),axis=1).sum()/.5))+1)
        coarse=curve(np.linspace(0,1,count))
        ax.plot(coarse[:,dims[0]],coarse[:,dims[1]],'o-',color='tab:orange',lw=.7,ms=3,label='0.5 m setting')
        for center,half in case['boxes']:
            xy=center[list(dims)]-half[list(dims)]
            ax.add_patch(Rectangle(xy,2*half[dims[0]],2*half[dims[1]],color='.3',alpha=.25))
        ax.set_title(method.replace('_',' ')+'\n'+('dense free' if valid else 'dense collision / bounds failure'),fontsize=10)
        ax.set_xlabel('x [m]' if dims[0]==0 else 'y [m]')
        ax.set_ylabel('y [m]' if dims[1]==1 else 'x [m]')
        ax.set_aspect('equal',adjustable='datalim');ax.grid(alpha=.2)
    np.atleast_1d(axes)[0].legend(fontsize=8)
    fig.suptitle(case['case']+' seed '+str(case['seed'])+' (XY projection; collision checks use 3D)',fontsize=11)
    fig.savefig(output,dpi=160)
    fig.savefig(output.with_suffix('.pdf'))
    plt.close(fig)


def run(args):
    args.output.mkdir(parents=True,exist_ok=True)
    _,static=core.load_octomap_points(base.MAP)
    cases=[fig4_case()]
    failures=[]
    for layout in range(5):
        scenario=paper_scenario(layout)
        grid=core.OccupancyGrid(base.RESOLUTION,base.INFLATION)
        grid.add_points(static)
        for center,half in scenario.boxes:
            grid.add_box(center.tolist(),half.tolist())
        for k in range(args.runs):
            seed=100*(k+1)
            try:
                raw=RRTStar(grid,base.BOUNDS,seed=seed).plan(scenario.start,scenario.goal)
            except JaxaPlanError as error:
                failures.append(dict(case=scenario.label,seed=seed,error=str(error)))
                continue
            cases.append(dict(case=scenario.label,seed=seed,raw=raw,grid=grid,bounds=base.BOUNDS,boxes=scenario.boxes))
    raw_saved={f"{c['case']}_seed{c['seed']}":np.array(c['raw']) for c in cases}
    np.savez_compressed(args.output/'fixed_rrt_paths.npz',**raw_saved)
    rows=[]
    for case in cases:
        assert path_is_free(case['raw'],case['grid'],case['bounds'])
        raw_length=float(np.linalg.norm(np.diff(case['raw'],axis=0),axis=1).sum())
        curves={}
        for method in METHODS:
            t0=time.perf_counter()
            curve,points,length=make_curve(case['raw'],method,case['grid'],case['bounds'])
            build_s=time.perf_counter()-t0
            _,dense=dense_samples(curve)
            metrics=curve_metrics(curve,dense,case['grid'],case['bounds'],case['boxes'])
            curves[method]=(curve,points,dense,metrics['dense_free'])
            for spacing in SPACINGS:
                count=max(2,int(np.ceil(length/spacing))+1)
                coarse=curve(np.linspace(0,1,count))
                if method=='all_points_interpolation':
                    np.testing.assert_allclose(coarse,bspline_waypoints(case['raw'],spacing),atol=1e-10,rtol=0)
                np.testing.assert_allclose(coarse[[0,-1]],case['raw'][[0,-1]],atol=1e-10,rtol=0)
                row=dict(case=case['case'],seed=case['seed'],method=method,spacing_setting_m=spacing,
                         raw_points=len(case['raw']),fit_points=len(points),waypoints=len(coarse),
                         raw_length_m=raw_length,build_s=build_s,**metrics,
                         coarse_free=bool(path_is_free(coarse,case['grid'],case['bounds'])),
                         max_corner_deg=corners(coarse),tracking_switch_max_m=tracking_switch(coarse,dense))
                rows.append(row)
        if case['case']=='fig4_like' or (case['case']=='layout0' and case['seed']==100):
            plot_case(case,curves,args.output/(case['case']+'_comparison.png'))
    with (args.output/'metrics.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    summary=[]
    for group in ('fig4_like','layouts'):
        for method in METHODS:
            for spacing in SPACINGS:
                selected=[r for r in rows if (r['case']=='fig4_like')==(group=='fig4_like') and r['method']==method and r['spacing_setting_m']==spacing]
                safe=[r for r in selected if r['dense_free'] and r['coarse_free']]
                item=dict(group=group,method=method,spacing_setting_m=spacing,cases=len(selected),
                          dense_free=sum(r['dense_free'] for r in selected),coarse_free=sum(r['coarse_free'] for r in selected),
                          both_free=len(safe),coarse_false_safe=sum(r['coarse_free'] and not r['dense_free'] for r in selected),
                          median_length_ratio=float(np.median([r['curve_length_m']/r['raw_length_m'] for r in selected])),
                          median_corner_deg=float(np.median([r['max_corner_deg'] for r in selected])),
                          median_tracking_switch_m=float(np.median([r['tracking_switch_max_m'] for r in selected])),
                          safe_median_corner_deg=float(np.median([r['max_corner_deg'] for r in safe])) if safe else None,
                          median_fit_points=float(np.median([r['fit_points'] for r in selected])))
                summary.append(item)
                if spacing in (.5,.1):
                    print(json.dumps(item),flush=True)
    report=dict(rrt_failures=failures,summary=summary,dense_step_bound_m=.01,
                sampling='same chord-parameter count rule as production; setting is not constant arc length',
                collision_map='static OctoMap plus complete inflated boxes, not online depth',
                methods='reconstruction assumptions; neither shortcut selection nor control-point fitting is specified by paper')
    (args.output/'summary.json').write_text(json.dumps(report,indent=2)+'\n')


def corner_spline(points, radius, corner_samples=6):
    """Quadratic B-spline that keeps each edge straight and rounds only within radius of a corner.

    Each corner P gets A/B on its edges at min(radius, half edge) and the piece
    A-P-B is a quadratic Bezier; straight pieces are degree-elevated lines. The
    curve stays in the convex hull of A,P,B, at most radius*sin(turn/2)/2 from P.
    Also returns waypoints: straight pieces keep only their ends and each corner arc
    gets corner_samples points, so spacing is not constant (as IAC-22 3.2 states) and
    the waypoint polyline does not cut corners the way uniform resampling does.
    """
    pts = distinct(points)
    if radius <= 0:
        raise ValueError('radius must be positive')
    edges = np.diff(pts,axis=0)
    lengths = np.linalg.norm(edges,axis=1)
    units = edges/lengths[:,None]
    trims = [min(radius,.5*lengths[i-1],.5*lengths[i]) for i in range(1,len(pts)-1)]
    pieces = []
    start = pts[0]
    for i in range(1,len(pts)-1):
        a,b = pts[i]-trims[i-1]*units[i-1],pts[i]+trims[i-1]*units[i]
        if np.linalg.norm(a-start) > 1e-9:
            pieces.append((start,.5*(start+a),a))
        pieces.append((a,pts[i],b))
        start = b
    if np.linalg.norm(pts[-1]-start) > 1e-9:
        pieces.append((start,.5*(start+pts[-1]),pts[-1]))
    span = np.array([np.linalg.norm(c[1]-c[0])+np.linalg.norm(c[2]-c[1]) for c in pieces])
    breaks = np.r_[0.,np.cumsum(span)]/span.sum()
    control = [pieces[0][0]]
    for piece in pieces:
        control += [piece[1],piece[2]]
    knots = np.r_[0.,0.,0.,np.repeat(breaks[1:-1],2),1.,1.,1.]
    curve = BSpline(knots,np.array(control),2)
    waypoints = [pts[0]]
    for (a,mid,b),u0,u1 in zip(pieces,breaks[:-1],breaks[1:]):
        straight = np.linalg.norm(mid-.5*(a+b)) < 1e-9
        waypoints += [b] if straight else list(curve(np.linspace(u0,u1,corner_samples+1)[1:]))
    return curve, pts, float(lengths.sum()), np.array(waypoints)


def smoothing_spline(points, rms, end_weight=100., waypoint_step=.2):
    """Least-squares cubic B-spline over all RRT* nodes (splprep, s = N*rms^2).

    Fig.4(b) reconstruction: averages zigzags but keeps long straights. Heavy end
    weights pin start/goal (residual there is scaled by end_weight). Waypoints step
    at most waypoint_step along the curve; the residual target is an assumption.
    """
    pts = distinct(points)
    if len(pts) < 4:
        return make_curve(pts,'all_points_interpolation',None,None)+(None,)
    weights = np.ones(len(pts))
    weights[[0,-1]] = end_weight
    (knots,coeffs,degree),_ = splprep(pts.T,w=weights,s=len(pts)*rms**2,k=3)
    count = len(knots)-degree-1
    curve = BSpline(knots,np.array(coeffs).T[:count],degree)
    # Pin exactly: clamped ends make the first/last coefficients the curve's end points.
    curve.c[0],curve.c[-1] = pts[0],pts[-1]
    _,waypoints = dense_samples(curve,waypoint_step)
    return curve, pts, float(np.linalg.norm(np.diff(pts,axis=0),axis=1).sum()), waypoints


def plan_with_retries(raw_for_seed, seed, build, grid, bounds):
    """Return the accepted curve or the failure reason, attempt by attempt.

    build(raw) -> (curve, fit points, polyline length[, waypoints]); without explicit
    waypoints the production 0.5 m chord-parameter resampling is used. grid is the final check.
    """
    rrt_s=smooth_s=0.
    for attempt in range(MAX_ATTEMPTS):
        try:
            raw,elapsed=raw_for_seed(seed+attempt)
        except JaxaPlanError as error:
            return dict(success=False,reason='rrt: '+str(error),attempts=attempt+1,rrt_s=rrt_s,smooth_s=smooth_s)
        rrt_s+=elapsed
        t0=time.perf_counter()
        curve,points,length,*waypoints=build(raw)
        _,dense=dense_samples(curve)
        count=max(2,int(np.ceil(length/PRODUCTION_SPACING))+1)
        coarse=waypoints[0] if waypoints and waypoints[0] is not None else curve(np.linspace(0,1,count))
        dense_free=bool(path_is_free(dense_samples(curve,CHECK_STEP)[1],grid,bounds))
        coarse_free=bool(path_is_free(coarse,grid,bounds))
        smooth_s+=time.perf_counter()-t0
        if dense_free and coarse_free:
            return dict(success=True,reason='',attempts=attempt+1,rrt_s=rrt_s,smooth_s=smooth_s,
                        curve=curve,points=points,dense=dense,coarse=coarse,raw=raw)
    return dict(success=False,reason='smoothed path collided',attempts=MAX_ATTEMPTS,rrt_s=rrt_s,smooth_s=smooth_s)


def run_retry(args):
    args.output.mkdir(parents=True,exist_ok=True)
    _,static=core.load_octomap_points(base.MAP)
    rows=[]
    for layout in range(5):
        scenario=paper_scenario(layout)
        grid=core.OccupancyGrid(base.RESOLUTION,base.INFLATION)
        grid.add_points(static)
        for center,half in scenario.boxes:
            grid.add_box(center.tolist(),half.tolist())
        cache={}
        def raw_for_seed(seed):
            if seed not in cache:
                t0=time.perf_counter()
                try:
                    cache[seed]=(RRTStar(grid,base.BOUNDS,seed=seed).plan(scenario.start,scenario.goal),time.perf_counter()-t0)
                except JaxaPlanError as error:
                    cache[seed]=error
            if isinstance(cache[seed],Exception):
                raise cache[seed]
            return cache[seed]
        straight=float(np.linalg.norm(np.asarray(scenario.goal)-scenario.start))
        for k in range(args.runs):
            seed=100*(k+1)
            curves={}
            for method in CANDIDATES:
                result=plan_with_retries(raw_for_seed,seed,lambda raw:make_curve(raw,method,grid,base.BOUNDS),
                                         grid,base.BOUNDS)
                row=dict(case=scenario.label,seed=seed,method=method,success=result['success'],
                         reason=result['reason'],attempts=result['attempts'],
                         rrt_s=result['rrt_s'],smooth_s=result['smooth_s'])
                if result['success']:
                    metrics=curve_metrics(result['curve'],result['dense'],grid,base.BOUNDS,scenario.boxes)
                    raw_length=float(np.linalg.norm(np.diff(result['raw'],axis=0),axis=1).sum())
                    row.update(fit_points=len(result['points']),raw_length_m=raw_length,
                               length_over_straight=metrics['curve_length_m']/straight,
                               length_over_raw=metrics['curve_length_m']/raw_length,
                               max_corner_deg=corners(result['coarse']),
                               tracking_switch_max_m=tracking_switch(result['coarse'],result['dense']),**metrics)
                    curves[method]=(result['curve'],result['points'],result['dense'],True,result['raw'])
                rows.append(row)
                print(json.dumps({key:row[key] for key in ('case','seed','method','success','reason','attempts')}),flush=True)
            if scenario.label=='layout0' and len(curves)==len(CANDIDATES):
                if not (args.output/'layout0_retry_comparison.png').exists():
                    plot_case(dict(case='layout0',seed=seed,raw=None,boxes=scenario.boxes),curves,
                              args.output/'layout0_retry_comparison.png',CANDIDATES)
    case=fig4_case()
    curves={}
    for method in CANDIDATES:
        curve,points,_=make_curve(case['raw'],method,case['grid'],case['bounds'])
        _,dense=dense_samples(curve)
        curves[method]=(curve,points,dense,bool(path_is_free(dense,case['grid'],case['bounds'])))
    plot_case(case,curves,args.output/'fig4_like_comparison.png',CANDIDATES)
    fields=list(dict.fromkeys(key for row in rows for key in row))
    with (args.output/'retry_metrics.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader();writer.writerows(rows)
    def median(values):
        return float(np.median(values)) if values else None
    summary=[]
    for method in CANDIDATES:
        for group in [f'layout{i}' for i in range(5)]+['all']:
            selected=[r for r in rows if r['method']==method and (group=='all' or r['case']==group)]
            ok=[r for r in selected if r['success']]
            item=dict(method=method,group=group,runs=len(selected),success=len(ok),
                      rrt_failures=sum(r['reason'].startswith('rrt') for r in selected),
                      smoothing_failures=sum(r['reason']=='smoothed path collided' for r in selected),
                      attempts_hist={str(a):sum(r['attempts']==a for r in ok) for a in range(1,MAX_ATTEMPTS+1)},
                      median_rrt_s=median([r['rrt_s'] for r in ok]),max_rrt_s=max([r['rrt_s'] for r in ok],default=None),
                      median_smooth_s=median([r['smooth_s'] for r in ok]),
                      median_length_over_straight=median([r['length_over_straight'] for r in ok]),
                      median_length_over_raw=median([r['length_over_raw'] for r in ok]),
                      median_max_curvature_per_m=median([r['max_curvature_per_m'] for r in ok]),
                      min_box_clearance_m=min([r['min_box_clearance_m'] for r in ok],default=None),
                      median_max_corner_deg=median([r['max_corner_deg'] for r in ok]),
                      median_tracking_switch_m=median([r['tracking_switch_max_m'] for r in ok]))
            summary.append(item)
            print(json.dumps(item),flush=True)
    report=dict(summary=summary,max_attempts=MAX_ATTEMPTS,attempt_seed='seed + attempt (production rule)',
                acceptance=f'curve sampled at <={CHECK_STEP} m and 0.5 m-setting polyline both free in inflated grid',
                timing='rrt_s is native RRT*; smooth_s is Python shortcut/fit/checks, not production cost',
                collision_map='static OctoMap plus complete inflated boxes, not online depth',
                methods='reconstruction assumptions; neither shortcut selection nor control-point fitting is specified by paper')
    (args.output/'summary.json').write_text(json.dumps(report,indent=2)+'\n')


# (shortcut margin m, corner radius m, fall back to the no-margin farthest edge instead of the raw edge)
# margin 0 with r = 0.2..0.6 is the r sweep (2026-10-02 rule: r > lookahead d, highest success, ties -> larger r).
CORNER_VARIANTS = ((0.,.2,False),(0.,.3,False),(0.,.4,False),(0.,.5,False),(0.,.6,False),
                   (.1,.3,False),(.1,.6,False),(.2,.3,False),(.2,.6,False),(.1,.3,True),(.2,.3,True))


def variant_label(margin, radius, hybrid):
    return f'margin{margin:.1f}_r{radius:.1f}'+('_fallback_shortcut' if hybrid else '')


def margin_shortcut(path, margin_grid, bounds, fallback_grid=None):
    """Farthest-visible shortcut judged on a wider inflation; falls back to the raw edge,
    or with fallback_grid to the farthest edge free there (no margin).

    Raw RRT* points may lie inside the margin, so unlike shortcut_path there is no
    all-points-free precondition (the raw edges are free in the check grid).
    """
    pts = distinct(path)
    result,i = [pts[0]],0
    while i < len(pts)-1:
        farthest = None
        for grid in (margin_grid,) if fallback_grid is None else (margin_grid,fallback_grid):
            for j in range(len(pts)-1,i+1,-1):
                if segment_is_free(pts[i],pts[j],grid,bounds,CHECK_STEP):
                    farthest = j
                    break
            if farthest is not None:
                break
        farthest = i+1 if farthest is None else farthest
        result.append(pts[farthest])
        i = farthest
    return np.array(result)


class OmplSimplifier:
    """Python port of OMPL geometric::PathSimplifier::simplifyMax (ompl/ompl main, PathSimplifier.cpp).

    Defaults as SimpleSetup::simplifySolution() with no duration: path-length cost,
    checkMotion at OMPL's longestValidSegmentFraction 0.01 of the bounds diagonal.
    findBetterGoal is a no-op for a single goal state and checkAndRepair never fires
    because every accepted state comes from a checked motion; both are omitted.
    """

    def __init__(self, grid, bounds, seed, motion_step=None):
        self.grid,self.bounds = grid,bounds
        lo,hi = (np.asarray(v,dtype=float) for v in bounds)
        self.step = .01*float(np.linalg.norm(hi-lo)) if motion_step is None else motion_step
        self.rng = np.random.RandomState(seed)

    def valid(self, p):
        return point_is_free(p,self.grid,self.bounds)

    def motion(self, a, b):
        # OMPL DiscreteMotionValidator: ceil(distance/longest valid segment) interior checks plus the end state.
        count = max(1,int(np.ceil(np.linalg.norm(b-a)/self.step)))
        return all(self.valid(a+(b-a)*k/count) for k in range(1,count+1))

    @staticmethod
    def cumulative(states):
        return np.r_[0.,np.cumsum(np.linalg.norm(np.diff(states,axis=0),axis=1))]

    def partial_shortcut(self, states, range_ratio=.33, snap=.005):
        states = list(states)
        if len(states) < 3:
            return states,False
        dists = self.cumulative(np.array(states))
        threshold,rd = dists[-1]*snap,range_ratio*dists[-1]
        result,nochange,i = False,0,0
        max_steps = max_empty = len(states)
        while i < max_steps and nochange < max_empty:
            i += 1; nochange += 1
            def locate(distance):
                pos = int(min(np.searchsorted(dists,distance),len(dists)-1))
                index = -1
                if pos == 0 or dists[pos]-distance < threshold:
                    index = pos
                else:
                    while pos > 0 and distance < dists[pos]:
                        pos -= 1
                    if distance-dists[pos] < threshold:
                        index = pos
                return pos,index
            d0 = self.rng.uniform(0.,dists[-1]); pos0,index0 = locate(d0)
            d1 = self.rng.uniform(max(0.,d0-rd),min(d0+rd,dists[-1])); pos1,index1 = locate(d1)
            if (pos0 == pos1 or index0 == pos1 or index1 == pos0 or pos0+1 == index1 or pos1+1 == index0
                    or (index0 >= 0 and index1 >= 0 and abs(index0-index1) < 2)):
                continue
            def state(pos,index,distance):
                if index >= 0:
                    return states[index],0.
                t = (distance-dists[pos])/(dists[pos+1]-dists[pos])
                return states[pos]+t*(states[pos+1]-states[pos]),t
            s0,_ = state(pos0,index0,d0); s1,_ = state(pos1,index1,d1)
            if not self.motion(s0,s1):
                continue
            if pos0 > pos1:
                pos0,pos1,index0,index1,s0,s1 = pos1,pos0,index1,index0,s1,s0
            along = (0. if index0 >= 0 else np.linalg.norm(states[pos0+1]-s0))
            along += sum(np.linalg.norm(states[k+1]-states[k]) for k in range(pos0+1,pos1))
            along += (0. if index1 >= 0 else np.linalg.norm(s1-states[pos1]))
            if along < np.linalg.norm(s1-s0):
                continue
            if index0 < 0 and index1 < 0:
                if pos0+1 == pos1:
                    states[pos1] = s0.copy(); states.insert(pos1+1,s1.copy())
                else:
                    states[pos0+1] = s0.copy(); states[pos1] = s1.copy(); del states[pos0+2:pos1]
            elif index0 >= 0 and index1 >= 0:
                del states[index0+1:index1]
            elif index0 < 0:
                states[pos0+1] = s0.copy(); del states[pos0+2:index1]
            else:
                states[pos1] = s1.copy(); del states[index0+1:pos1]
            dists = self.cumulative(np.array(states))
            threshold,rd = dists[-1]*snap,range_ratio*dists[-1]
            result,nochange = True,0
        return states,result

    def smooth_bspline(self, states, max_steps=3, min_change=None):
        states = [np.array(p) for p in states]
        if len(states) < 3:
            return states
        for _ in range(max_steps):
            subdivided = [states[0]]
            for p in states[1:]:
                subdivided += [.5*(subdivided[-1]+p),p]
            states,updated,i = subdivided,0,2
            while i < len(states)-1:
                if self.valid(states[i-1]):
                    candidate = .5*(.5*(states[i-1]+states[i])+.5*(states[i]+states[i+1]))
                    if self.motion(states[i-1],candidate) and self.motion(candidate,states[i+1]):
                        if np.linalg.norm(states[i]-candidate) > min_change:
                            states[i] = candidate; updated += 1
                i += 2
            if updated == 0:
                break
        return states

    def reduce_vertices(self, states, range_ratio=.33):
        states = list(states)
        if len(states) < 3:
            return states,False
        if self.motion(states[0],states[-1]):
            return [states[0],states[-1]],True
        result,nochange,i = False,0,0
        max_steps = max_empty = len(states)
        while i < max_steps and nochange < max_empty:
            i += 1; nochange += 1
            count = len(states); max_n = count-1
            span = 1+int(np.floor(.5+count*range_ratio))
            p1 = self.rng.randint(0,max_n+1)
            p2 = self.rng.randint(max(p1-span,0),min(max_n,p1+span)+1)
            if abs(p1-p2) < 2:
                if p1 < max_n-1: p2 = p1+2
                elif p1 > 1: p2 = p1-2
                else: continue
            p1,p2 = min(p1,p2),max(p1,p2)
            if self.motion(states[p1],states[p2]):
                del states[p1+1:p2]; nochange = 0; result = True
        return states,result

    def collapse_close_vertices(self, states):
        states = list(states)
        if len(states) < 3:
            return states
        blocked = set()
        for _ in range(len(states)):
            best = None
            for i in range(len(states)):
                for j in range(i+2,len(states)):
                    key = (tuple(states[i]),tuple(states[j]))
                    if key in blocked:
                        continue
                    d = np.linalg.norm(states[i]-states[j])
                    if best is None or d < best[0]:
                        best = (d,i,j,key)
            if best is None:
                break
            _,i,j,key = best
            if self.motion(states[i],states[j]):
                del states[i+1:j]
            else:
                blocked.add(key)
        return states

    def simplify_max(self, path):
        states = [np.asarray(p,dtype=float) for p in distinct(path)]
        if len(states) < 3:
            return np.array(states)
        try_more = True
        while try_more:
            for _ in range(6):
                states,shortcut = self.partial_shortcut(states)
                if not shortcut:
                    break
            length = float(self.cumulative(np.array(states))[-1])
            states = self.smooth_bspline(states,3,length/100.)
            states,try_more = self.reduce_vertices(states)
            states = self.collapse_close_vertices(states)
            for _ in range(5):
                if not try_more:
                    break
                states,try_more = self.reduce_vertices(states)
        return np.array(states)


def ompl_path(points, grid, bounds, motion_step=None, mode='simplify_max'):
    """OMPL post-processing as the tracked waypoint polyline. mode: simplify_max,
    bspline_only (smoothBSpline defaults, 5 steps), reduce_then_bspline / partial_then_bspline
    (reduceVertices / partialShortcutPath defaults, then smoothBSpline defaults)."""
    raw = distinct(points)
    simplifier = OmplSimplifier(grid,bounds,zlib.crc32(raw.tobytes()),motion_step)
    if mode == 'simplify_max':
        states = simplifier.simplify_max(raw)
    elif mode in ('bspline_only','reduce_then_bspline','partial_then_bspline'):
        states = (list(raw) if mode == 'bspline_only' else simplifier.reduce_vertices(list(raw))[0]
                  if mode == 'reduce_then_bspline' else simplifier.partial_shortcut(list(raw))[0])
        states = simplifier.smooth_bspline(states,5,np.finfo(float).eps)
    else:
        raise ValueError(mode)
    states = distinct(states)
    u = np.r_[0.,np.cumsum(np.linalg.norm(np.diff(states,axis=0),axis=1))]
    curve = BSpline(np.r_[0.,u/u[-1],1.],states,1)
    return curve, states, float(u[-1]), states


FIT_RMS = (.1,.2,.3,.4)  # smoothing-spline residual rms [m]; Fig.4 reconstruction looked closest near 0.3


def run_corner(args):
    """RRT* on the production grid -> (margin shortcut -> corner-only spline | smoothing spline)
    -> check on the production grid, with retries. All smoothing parameters are assumptions."""
    args.output.mkdir(parents=True,exist_ok=True)
    _,static=core.load_octomap_points(base.MAP)
    def make_grid(inflation,boxes):
        grid=core.OccupancyGrid(base.RESOLUTION,inflation)
        grid.add_points(static)
        for center,half in boxes:
            grid.add_box(center.tolist(),half.tolist())
        return grid
    rows=[]
    examples={}
    for layout in range(5):
        scenario=paper_scenario(layout)
        check=make_grid(base.INFLATION,scenario.boxes)
        straight=float(np.linalg.norm(np.asarray(scenario.goal)-scenario.start))
        planning={margin:check if margin==0 else make_grid(base.INFLATION+margin,scenario.boxes)
                  for margin in sorted({m for m,_,_ in CORNER_VARIANTS})}
        cache={}
        def raw_for_seed(seed):
            if seed not in cache:
                t0=time.perf_counter()
                try:
                    cache[seed]=(RRTStar(check,base.BOUNDS,seed=seed).plan(scenario.start,scenario.goal),time.perf_counter()-t0)
                except JaxaPlanError as error:
                    cache[seed]=error
            if isinstance(cache[seed],Exception):
                raise cache[seed]
            return cache[seed]
        variants=[(variant_label(margin,radius,hybrid),dict(margin_m=margin,radius_m=radius),
                   lambda raw,g=planning[margin],r=radius,h=hybrid:
                       corner_spline(margin_shortcut(raw,g,base.BOUNDS,check if h else None),r))
                  for margin,radius,hybrid in CORNER_VARIANTS]
        variants+=[(f'smoothing_rms{rms:.1f}',dict(fit_rms_m=rms),lambda raw,rms=rms:smoothing_spline(raw,rms))
                   for rms in FIT_RMS]
        variants+=[('ompl_simplify_max',{},lambda raw:ompl_path(raw,check,base.BOUNDS)),
                   ('ompl_smooth_bspline_only',{},lambda raw:ompl_path(raw,check,base.BOUNDS,mode='bspline_only')),
                   ('ompl_reduce_then_bspline',{},lambda raw:ompl_path(raw,check,base.BOUNDS,mode='reduce_then_bspline'))]
        for label,meta,build in variants:
            for k in range(args.runs):
                seed=100*(k+1)
                result=plan_with_retries(raw_for_seed,seed,build,check,base.BOUNDS)
                row=dict(case=scenario.label,seed=seed,variant=label,**meta,
                         success=result['success'],reason=result['reason'],attempts=result['attempts'],
                         rrt_s=result['rrt_s'],smooth_s=result['smooth_s'])
                if result['success']:
                    metrics=curve_metrics(result['curve'],result['dense'],check,base.BOUNDS,scenario.boxes)
                    pts=result['points']
                    row.update(segments=len(pts)-1,polyline_over_straight=float(np.linalg.norm(np.diff(pts,axis=0),axis=1).sum())/straight,
                               length_over_straight=metrics['curve_length_m']/straight,
                               polyline_max_corner_deg=corners(pts),max_corner_deg=corners(result['coarse']),
                               waypoints=len(result['coarse']),
                               waypoint_spacing_min_m=float(np.linalg.norm(np.diff(result['coarse'],axis=0),axis=1).min()),
                               waypoint_spacing_max_m=float(np.linalg.norm(np.diff(result['coarse'],axis=0),axis=1).max()),
                               tracking_switch_max_m=tracking_switch(result['coarse'],result['dense']),**metrics)
                    if scenario.label=='layout0' and label not in examples:
                        examples[label]=(result['curve'],pts,result['dense'],True,result['raw'])
                rows.append(row)
    fields=list(dict.fromkeys(key for row in rows for key in row))
    with (args.output/'corner_metrics.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader();writer.writerows(rows)
    def median(values):
        return float(np.median(values)) if values else None
    summary=[]
    for label in ([variant_label(*v) for v in CORNER_VARIANTS]+[f'smoothing_rms{rms:.1f}' for rms in FIT_RMS]
                  +['ompl_simplify_max','ompl_smooth_bspline_only','ompl_reduce_then_bspline']):
        selected=[r for r in rows if r['variant']==label]
        ok=[r for r in selected if r['success']]
        item=dict(variant=label,runs=len(selected),success=len(ok),
                  success_by_layout={f'layout{i}':sum(r['success'] for r in selected if r['case']==f'layout{i}') for i in range(5)},
                  rrt_failures=sum(r['reason'].startswith('rrt') for r in selected),
                  smoothing_failures=sum(r['reason']=='smoothed path collided' for r in selected),
                  attempts_hist={str(a):sum(r['attempts']==a for r in ok) for a in range(1,MAX_ATTEMPTS+1)},
                  max_rrt_s=max([r['rrt_s'] for r in ok],default=None),
                  median_segments=median([r['segments'] for r in ok]),
                  median_length_over_straight=median([r['length_over_straight'] for r in ok]),
                  median_polyline_max_corner_deg=median([r['polyline_max_corner_deg'] for r in ok]),
                  median_max_corner_deg_waypoints=median([r['max_corner_deg'] for r in ok]),
                  median_waypoints=median([r['waypoints'] for r in ok]),
                  median_spacing_min_max_m=[median([r['waypoint_spacing_min_m'] for r in ok]),
                                            median([r['waypoint_spacing_max_m'] for r in ok])],
                  median_max_curvature_per_m=median([r['max_curvature_per_m'] for r in ok]),
                  min_box_clearance_m=min([r['min_box_clearance_m'] for r in ok],default=None),
                  median_tracking_switch_m=median([r['tracking_switch_max_m'] for r in ok]))
        summary.append(item)
        print(json.dumps(item),flush=True)
    case=fig4_paper_case()
    fig4={}
    for label,curve in (('shortcut+corner margin0 r0.3',corner_spline(margin_shortcut(case['raw'],case['grid'],case['bounds']),.3)[0]),
                        *[(f'smoothing rms{rms:.1f}',smoothing_spline(case['raw'],rms)[0]) for rms in FIT_RMS],
                        ('OMPL simplifyMax',ompl_path(case['raw'],case['grid'],case['bounds'])[0]),
                        ('OMPL smoothBSpline only',ompl_path(case['raw'],case['grid'],case['bounds'],mode='bspline_only')[0]),
                        ('OMPL reduceVertices+smoothBSpline',ompl_path(case['raw'],case['grid'],case['bounds'],mode='reduce_then_bspline')[0])):
        fig4[label]=dense_samples(curve,.005)[1]
    plot_fig4_cells(case,fig4,args.output/'fig4_paper_cells.png')
    if examples:
        examples={k:v for k,v in examples.items() if k in ('margin0.0_r0.2','ompl_simplify_max','ompl_smooth_bspline_only',
                                                            'ompl_reduce_then_bspline')}
        plot_case(dict(case='layout0',seed='first success',raw=None,boxes=paper_scenario(0).boxes),examples,
                  args.output/'layout0_corner_comparison.png',tuple(examples))
    report=dict(summary=summary,max_attempts=MAX_ATTEMPTS,attempt_seed='seed + attempt (production rule)',
                planning='RRT* and final check on production inflation grid; only the shortcut uses inflation+margin',
                acceptance=f'curve sampled at <={CHECK_STEP} m and the corner-sampled waypoint polyline both free in the check grid',
                collision_map='static OctoMap plus complete inflated boxes, not online depth',
                smoothing='splprep over all RRT* nodes, s=N*rms^2, end weight 100 then exact end pinning, waypoints <=0.2 m',
                methods='margin shortcut, corner-only spline and smoothing residual are reconstruction assumptions, not specified by paper')
    (args.output/'summary.json').write_text(json.dumps(report,indent=2)+'\n')


OMPL_SOLVE_TIMES = (.1,.3,1.)  # RRTstar runs to its time limit; IAC-22 gives none (1.0 s is our budget)
OMPL_MODES = ('reduce_then_bspline','simplify_max')


def run_ompl(args):
    """Real OMPL RRTstar + PathSimplifier (sobits_intball2_gnc_cpp.jaxa_ompl_plan), retried up to
    MAX_ATTEMPTS. Each attempt is scored by two final checks: the production half-voxel step
    and OMPL's own motion resolution; attempts continue until both rules have accepted once."""
    args.output.mkdir(parents=True,exist_ok=True)
    _,static=core.load_octomap_points(base.MAP)
    rows=[]
    for layout in range(5):
        scenario=paper_scenario(layout)
        grid=core.OccupancyGrid(base.RESOLUTION,base.INFLATION)
        grid.add_points(static)
        for center,half in scenario.boxes:
            grid.add_box(center.tolist(),half.tolist())
        straight=float(np.linalg.norm(np.asarray(scenario.goal)-scenario.start))
        ompl_check=OmplSimplifier(grid,base.BOUNDS,0)
        for k in range(args.runs):
            for solve_time in OMPL_SOLVE_TIMES:
                for mode in OMPL_MODES:
                    accepted={'fine':None,'ompl':None}
                    for attempt in range(1,MAX_ATTEMPTS+1):
                        row=dict(case=scenario.label,run=k,solve_time_s=solve_time,mode=mode,attempt=attempt)
                        try:
                            raw,path,solve_s,simplify_s=core.jaxa_ompl_plan(
                                grid,list(scenario.start),list(scenario.goal),*map(list,base.BOUNDS),solve_time,mode)
                        except JaxaPlanError as error:
                            row.update(error=str(error)); rows.append(row)
                            break
                        raw,path=distinct(raw),distinct(path)
                        fine=bool(path_is_free(dense_samples(ompl_path_curve(path),CHECK_STEP)[1],grid,base.BOUNDS))
                        coarse=all(ompl_check.motion(a,b) for a,b in zip(path[:-1],path[1:]))
                        row.update(error='',solve_s=solve_s,simplify_s=simplify_s,raw_points=len(raw),waypoints=len(path),
                                   raw_over_straight=float(np.linalg.norm(np.diff(raw,axis=0),axis=1).sum())/straight,
                                   length_over_straight=float(np.linalg.norm(np.diff(path,axis=0),axis=1).sum())/straight,
                                   max_corner_deg=corners(path),fine_free=fine,ompl_free=coarse,
                                   min_box_clearance_m=box_clearance(dense_samples(ompl_path_curve(path),.01)[1],scenario.boxes))
                        if mode=='reduce_then_bspline' and attempt==1:
                            port=ompl_path(raw,grid,base.BOUNDS,mode='reduce_then_bspline')[1]
                            row.update(port_waypoints=len(port),
                                       port_length_over_straight=float(np.linalg.norm(np.diff(port,axis=0),axis=1).sum())/straight)
                        rows.append(row)
                        for rule,ok in (('fine',fine),('ompl',coarse)):
                            if ok and accepted[rule] is None:
                                accepted[rule]=attempt
                        if all(v is not None for v in accepted.values()):
                            break
            print(json.dumps(dict(case=scenario.label,run=k)),flush=True)
    fields=list(dict.fromkeys(key for row in rows for key in row))
    with (args.output/'ompl_metrics.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader();writer.writerows(rows)
    def median(values):
        return float(np.median(values)) if values else None
    summary=[]
    for solve_time in OMPL_SOLVE_TIMES:
        for mode in OMPL_MODES:
            for rule in ('fine','ompl'):
                runs=[]
                for layout in range(5):
                    for k in range(args.runs):
                        tries=[r for r in rows if r['case']==f'layout{layout}' and r['run']==k
                               and r['solve_time_s']==solve_time and r['mode']==mode]
                        first=next((r for r in tries if r.get(f'{rule}_free')),None)
                        failed_rrt=any(r['error'] for r in tries if first is None or r['attempt']<first['attempt'])
                        runs.append((layout,first,failed_rrt))
                ok=[f for _,f,_ in runs if f is not None]
                item=dict(solve_time_s=solve_time,mode=mode,final_check=rule,runs=len(runs),success=len(ok),
                          success_by_layout={f'layout{i}':sum(f is not None for l,f,_ in runs if l==i) for i in range(5)},
                          rrt_failures=sum(f is None and e for _,f,e in runs),
                          attempts_hist={str(a):sum(f['attempt']==a for f in ok) for a in range(1,MAX_ATTEMPTS+1)},
                          median_raw_over_straight=median([f['raw_over_straight'] for f in ok]),
                          median_length_over_straight=median([f['length_over_straight'] for f in ok]),
                          median_waypoints=median([f['waypoints'] for f in ok]),
                          median_max_corner_deg=median([f['max_corner_deg'] for f in ok]),
                          min_box_clearance_m=min([f['min_box_clearance_m'] for f in ok],default=None))
                summary.append(item)
                print(json.dumps(item),flush=True)
    port=[r for r in rows if 'port_waypoints' in r]
    port_cmp=dict(pairs=len(port),median_waypoints_real=median([r['waypoints'] for r in port]),
                  median_waypoints_port=median([r['port_waypoints'] for r in port]),
                  median_length_real=median([r['length_over_straight'] for r in port]),
                  median_length_port=median([r['port_length_over_straight'] for r in port]))
    print(json.dumps(port_cmp),flush=True)
    report=dict(summary=summary,port_vs_real_reduce_then_bspline=port_cmp,max_attempts=MAX_ATTEMPTS,
                planner='OMPL 1.7.0 RRTstar via SimpleSetup, all OMPL defaults, terminated by solve time',
                final_checks={'fine':f'curve sampled at <={CHECK_STEP} m in the inflated grid',
                              'ompl':'OMPL DiscreteMotionValidator at 1% of the bounds diagonal'},
                collision_map='static OctoMap plus complete inflated boxes, not online depth',
                note='OMPL RNG is process-global; runs are not seed-reproducible')
    (args.output/'summary.json').write_text(json.dumps(report,indent=2)+'\n')


def run_ompl_figures(args):
    """Real OMPL shapes for visual comparison with IAC-22 Fig.4 and Fig.9 (not statistics)."""
    args.output.mkdir(parents=True,exist_ok=True)
    case=fig4_paper_case()
    lo,hi=(np.array(v,dtype=float) for v in case['bounds'])
    lo[2],hi[2]=-.01,.01  # Fig.4 is planar; a thin slab keeps OMPL from flying over the box
    bounds=(lo.tolist(),hi.tolist())
    dense=lambda states:dense_samples(ompl_path_curve(distinct(states)),.005)[1]
    curves={}
    for k in range(3):
        curves[f'real OMPL reduceVertices\n+smoothBSpline on (a) #{k+1}']=dense(
            core.jaxa_ompl_simplify(case['grid'],case['raw'].tolist(),*bounds,'reduce_then_bspline'))
    for k in range(3):
        curves[f'real OMPL partialShortcut\n+smoothBSpline on (a) #{k+1}']=dense(
            core.jaxa_ompl_simplify(case['grid'],case['raw'].tolist(),*bounds,'partial_then_bspline'))
    curves['real OMPL simplifyMax on (a)']=dense(core.jaxa_ompl_simplify(case['grid'],case['raw'].tolist(),*bounds,'simplify_max'))
    for k in range(2):
        raw,path,_,_=core.jaxa_ompl_plan(case['grid'],case['raw'][0].tolist(),case['raw'][-1].tolist(),*bounds,.3,
                                         'reduce_then_bspline')
        curves[f'real OMPL RRTstar 0.3 s\n+reduce+smooth #{k+1}']=dense(path)
    plot_fig4_cells(case,curves,args.output/'fig4_real_ompl.png')
    _,static=core.load_octomap_points(base.MAP)
    fig,axes=plt.subplots(5,3,figsize=(15,16),constrained_layout=True)
    for layout in range(5):
        scenario=paper_scenario(layout)
        grid=core.OccupancyGrid(base.RESOLUTION,base.INFLATION)
        grid.add_points(static)
        for center,half in scenario.boxes:
            grid.add_box(center.tolist(),half.tolist())
        for ax,mode in zip(axes[layout],('none','reduce_then_bspline','partial_then_bspline')):
            try:
                raw,path,_,_=core.jaxa_ompl_plan(grid,list(scenario.start),list(scenario.goal),*map(list,base.BOUNDS),.3,mode)
            except JaxaPlanError as error:
                ax.set_title(f'layout{layout} {mode}: {error}',fontsize=9); continue
            path=distinct(path)
            free=bool(path_is_free(dense_samples(ompl_path_curve(path),CHECK_STEP)[1],grid,base.BOUNDS))
            for center,half in scenario.boxes:
                ax.add_patch(Rectangle((center[1]-half[1],center[0]-half[0]),2*half[1],2*half[0],color='.3',alpha=.3))
            ax.plot(np.array(raw)[:,1],np.array(raw)[:,0],'--',color='.55',lw=1,label='RRTstar raw')
            ax.plot(path[:,1],path[:,0],color='tab:blue' if free else 'tab:red',lw=2,label='output')
            ax.set_title(f'layout{layout} {mode} ({len(path)} pts, final check {"free" if free else "touch"})',fontsize=9)
            ax.set_xlabel('y [m]');ax.set_ylabel('x [m]');ax.set_aspect('equal',adjustable='datalim');ax.grid(alpha=.2)
    axes[0][0].legend(fontsize=8)
    fig.suptitle('real OMPL 1.7.0, RRTstar 0.3 s (XY projection, one run each)',fontsize=11)
    fig.savefig(args.output/'layouts_real_ompl.png',dpi=110)
    plt.close(fig)


OMPL_INFLATIONS = (.2,.25,.3)


def run_ompl_inflation(args):
    """RRTstar args.solve_time + args.ompl_mode post-processing for several grid inflations; the same
    grid plans and checks. Each attempt is scored by OMPL's PathGeometric::check() and by the
    production half-voxel step; attempts continue until both rules have accepted once."""
    args.output.mkdir(parents=True,exist_ok=True)
    _,static=core.load_octomap_points(base.MAP)
    rows=[]
    for inflation in args.inflations:
        for layout in range(5):
            scenario=paper_scenario(layout)
            grid=core.OccupancyGrid(base.RESOLUTION,inflation)
            grid.add_points(static)
            for center,half in scenario.boxes:
                grid.add_box(center.tolist(),half.tolist())
            straight=float(np.linalg.norm(np.asarray(scenario.goal)-scenario.start))
            ompl_check=OmplSimplifier(grid,base.BOUNDS,0)
            for k in range(args.runs):
                accepted={'fine':None,'ompl':None}
                for attempt in range(1,MAX_ATTEMPTS+1):
                    row=dict(inflation_m=inflation,case=scenario.label,run=k,attempt=attempt)
                    try:
                        _,path,solve_s,_=core.jaxa_ompl_plan(grid,list(scenario.start),list(scenario.goal),
                                                             *map(list,base.BOUNDS),args.solve_time,args.ompl_mode)
                    except JaxaPlanError as error:
                        row.update(error=str(error)); rows.append(row)
                        break
                    path=distinct(path)
                    dense=dense_samples(ompl_path_curve(path),.01)[1]
                    fine=bool(path_is_free(dense_samples(ompl_path_curve(path),CHECK_STEP)[1],grid,base.BOUNDS))
                    coarse=all(ompl_check.motion(a,b) for a,b in zip(path[:-1],path[1:]))
                    row.update(error='',solve_s=solve_s,waypoints=len(path),max_corner_deg=corners(path),
                               length_over_straight=float(np.linalg.norm(np.diff(path,axis=0),axis=1).sum())/straight,
                               fine_free=fine,ompl_free=coarse,min_box_clearance_m=box_clearance(dense,scenario.boxes))
                    rows.append(row)
                    for rule,ok in (('fine',fine),('ompl',coarse)):
                        if ok and accepted[rule] is None:
                            accepted[rule]=attempt
                    if all(v is not None for v in accepted.values()):
                        break
        print(json.dumps(dict(inflation_m=inflation,done=True)),flush=True)
    fields=list(dict.fromkeys(key for row in rows for key in row))
    suffix=('' if args.ompl_mode=='reduce_then_bspline' else '_'+args.ompl_mode)+('' if args.solve_time==.1 else f'_{args.solve_time}s')
    with (args.output/f'ompl_inflation_metrics{suffix}.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader();writer.writerows(rows)
    def median(values):
        return float(np.median(values)) if values else None
    summary=[]
    for inflation in args.inflations:
        planned=[r for r in rows if r['inflation_m']==inflation and not r['error']]
        corner=np.array([r['max_corner_deg'] for r in planned])
        errors=[r['error'] for r in rows if r['inflation_m']==inflation and r['error']]
        item=dict(inflation_m=inflation,outputs=len(planned),
                  sharp_corner_over_45deg_share=float(np.mean(corner>45)) if len(corner) else None,
                  median_max_corner_deg=median(list(corner)),
                  errors={e:errors.count(e) for e in sorted(set(errors))})
        for rule in ('fine','ompl'):
            firsts=[]
            for layout in range(5):
                for k in range(args.runs):
                    tries=[r for r in rows if r['inflation_m']==inflation and r['case']==f'layout{layout}' and r['run']==k]
                    firsts.append((layout,next((r for r in tries if r.get(f'{rule}_free')),None)))
            ok=[f for _,f in firsts if f is not None]
            item[rule]=dict(success=len(ok),runs=len(firsts),
                            success_by_layout={f'layout{i}':sum(f is not None for l,f in firsts if l==i) for i in range(5)},
                            attempts_hist={str(a):sum(f['attempt']==a for f in ok) for a in range(1,MAX_ATTEMPTS+1)},
                            accepted_sharp_corner_share=float(np.mean([f['max_corner_deg']>45 for f in ok])) if ok else None,
                            median_length_over_straight=median([f['length_over_straight'] for f in ok]),
                            min_box_clearance_m=min([f['min_box_clearance_m'] for f in ok],default=None))
        summary.append(item)
        print(json.dumps(item),flush=True)
    report=dict(summary=summary,planner=f'OMPL 1.7.0 RRTstar {args.solve_time} s, {args.ompl_mode}, all OMPL defaults',
                final_checks={'fine':f'curve sampled at <={CHECK_STEP} m','ompl':'PathGeometric::check() semantics'},
                sharp_corner='max turn between consecutive output waypoints > 45 deg',
                collision_map='static OctoMap plus complete boxes inflated by inflation_m (same grid plans and checks)')
    suffix=('' if args.ompl_mode=='reduce_then_bspline' else '_'+args.ompl_mode)+('' if args.solve_time==.1 else f'_{args.solve_time}s')
    (args.output/f'inflation_summary{suffix}.json').write_text(json.dumps(report,indent=2)+'\n')


RETRY_LIMITS = (5,10,20)


def run_ompl_retry(args):
    """RRTstar + partialShortcutPath -> smoothBSpline, up to max(RETRY_LIMITS) attempts per plan,
    for several grid resolutions and solve times. An RRT* "no path" counts as a failed attempt
    (RRT* is random); an occupied start/goal ends the plan. The production-style check samples
    at half a voxel of each grid. Success for a smaller retry limit is read off the same attempts."""
    args.output.mkdir(parents=True,exist_ok=True)
    _,static=core.load_octomap_points(base.MAP)
    rows=[]
    limit=max(RETRY_LIMITS)
    for resolution in args.resolutions:
        for layout in range(5):
            scenario=paper_scenario(layout)
            grid=core.OccupancyGrid(resolution,base.INFLATION)
            grid.add_points(static)
            for center,half in scenario.boxes:
                grid.add_box(center.tolist(),half.tolist())
            straight=float(np.linalg.norm(np.asarray(scenario.goal)-scenario.start))
            ompl_check=OmplSimplifier(grid,base.BOUNDS,0)
            for solve_time in args.solve_times:
                for k in range(args.runs):
                    accepted={'fine':None,'ompl':None}
                    for attempt in range(1,limit+1):
                        row=dict(resolution_m=resolution,solve_time_s=solve_time,case=scenario.label,run=k,attempt=attempt)
                        try:
                            _,path,solve_s,_=core.jaxa_ompl_plan(grid,list(scenario.start),list(scenario.goal),
                                                                 *map(list,base.BOUNDS),solve_time,'partial_then_bspline')
                        except JaxaPlanError as error:
                            row.update(error=str(error)); rows.append(row)
                            if 'occupied' in str(error):
                                break
                            continue
                        path=distinct(path)
                        curve=ompl_path_curve(path)
                        fine=bool(path_is_free(dense_samples(curve,.5*resolution)[1],grid,base.BOUNDS))
                        coarse=all(ompl_check.motion(a,b) for a,b in zip(path[:-1],path[1:]))
                        row.update(error='',solve_s=solve_s,waypoints=len(path),max_corner_deg=corners(path),
                                   length_over_straight=float(np.linalg.norm(np.diff(path,axis=0),axis=1).sum())/straight,
                                   fine_free=fine,ompl_free=coarse,
                                   min_box_clearance_m=box_clearance(dense_samples(curve,.01)[1],scenario.boxes))
                        rows.append(row)
                        for rule,ok in (('fine',fine),('ompl',coarse)):
                            if ok and accepted[rule] is None:
                                accepted[rule]=attempt
                        if all(v is not None for v in accepted.values()):
                            break
            print(json.dumps(dict(resolution_m=resolution,case=scenario.label)),flush=True)
    fields=list(dict.fromkeys(key for row in rows for key in row))
    with (args.output/'ompl_retry_metrics.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader();writer.writerows(rows)
    def median(values):
        return float(np.median(values)) if values else None
    summary=[]
    for resolution in args.resolutions:
        for solve_time in args.solve_times:
            for rule in ('fine','ompl'):
                plans=[]
                for layout in range(5):
                    for k in range(args.runs):
                        tries=[r for r in rows if r['resolution_m']==resolution and r['solve_time_s']==solve_time
                               and r['case']==f'layout{layout}' and r['run']==k]
                        first=next((r for r in tries if r.get(f'{rule}_free')),None)
                        plans.append((layout,first,tries))
                for retries in RETRY_LIMITS:
                    ok=[(l,f,t) for l,f,t in plans if f is not None and f['attempt']<=retries]
                    # planning time until acceptance: every attempt solves for solve_time
                    times=[sum(r.get('solve_s',0.) for r in t if r['attempt']<=f['attempt']) for _,f,t in ok]
                    item=dict(resolution_m=resolution,solve_time_s=solve_time,final_check=rule,max_attempts=retries,
                              plans=len(plans),success=len(ok),
                              success_by_layout={f'layout{i}':sum(l==i for l,_,_ in ok) for i in range(5)},
                              median_attempts=median([f['attempt'] for _,f,_ in ok]),
                              median_plan_s=median(times),max_plan_s=max(times,default=None),
                              sharp_corner_share=float(np.mean([f['max_corner_deg']>45 for _,f,_ in ok])) if ok else None,
                              median_length_over_straight=median([f['length_over_straight'] for _,f,_ in ok]),
                              min_box_clearance_m=min([f['min_box_clearance_m'] for _,f,_ in ok],default=None))
                    summary.append(item)
                    print(json.dumps(item),flush=True)
    report=dict(summary=summary,planner='OMPL 1.7.0 RRTstar + partialShortcutPath -> smoothBSpline, all OMPL defaults',
                inflation_m=base.INFLATION,retry_rule='RRT* no-path counts as a failed attempt; occupied start/goal ends the plan',
                final_checks={'fine':'curve sampled at half a voxel of the grid','ompl':'PathGeometric::check() semantics'},
                plan_s='sum of RRTstar solve times up to the accepted attempt (simplify time is negligible)',
                collision_map='static OctoMap plus complete inflated boxes, not online depth')
    name='_'.join(f'res{r}' for r in args.resolutions)+'_'+'_'.join(f'{t}s' for t in args.solve_times)
    (args.output/f'retry_summary_{name}.json').write_text(json.dumps(report,indent=2)+'\n')
    (args.output/'ompl_retry_metrics.csv').rename(args.output/f'ompl_retry_metrics_{name}.csv')


def ompl_path_curve(states):
    u=np.r_[0.,np.cumsum(np.linalg.norm(np.diff(states,axis=0),axis=1))]
    return BSpline(np.r_[0.,u/u[-1],1.],states,1)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--runs',type=int,default=10)
    parser.add_argument('--retry',action='store_true',help='candidate methods with production-style retries')
    parser.add_argument('--corner',action='store_true',help='margin shortcut + corner-only spline with retries')
    parser.add_argument('--ompl',action='store_true',help='real OMPL RRTstar + PathSimplifier with retries')
    parser.add_argument('--ompl-figures',action='store_true',help='real OMPL shapes vs Fig.4 / Fig.9')
    parser.add_argument('--ompl-inflation',action='store_true',help='real OMPL at several grid inflations')
    parser.add_argument('--ompl-mode',default='reduce_then_bspline',
                        choices=('reduce_then_bspline','partial_then_bspline','bspline_only','simplify_max'))
    parser.add_argument('--inflations',type=float,nargs='+',default=list(OMPL_INFLATIONS))
    parser.add_argument('--solve-time',type=float,default=.1,help='RRTstar solve time for --ompl-inflation [s]')
    parser.add_argument('--ompl-retry',action='store_true',help='real OMPL: retry limit x grid resolution x solve time')
    parser.add_argument('--resolutions',type=float,nargs='+',default=[.1])
    parser.add_argument('--solve-times',type=float,nargs='+',default=[.1,.3])
    parser.add_argument('--check-step',type=float,default=None,
                        help='collision sampling step [m]; default 1 cm, production is half a voxel')
    parser.add_argument('--output',type=Path,default=None)
    args=parser.parse_args()
    if args.check_step is not None:
        if args.check_step<=0:
            parser.error('--check-step must be positive')
        CHECK_STEP=args.check_step
    if args.runs<1:
        parser.error('--runs must be positive')
    if args.output is None:
        args.output=Path('docs/results/'+('jaxa_smoothing_ompl_offline' if args.ompl or args.ompl_figures
                                          or args.ompl_inflation or args.ompl_retry else
                                          'jaxa_smoothing_corner_offline' if args.corner else
                                          'jaxa_smoothing_retry_offline' if args.retry else 'jaxa_smoothing_offline'))
    if args.ompl_retry:
        run_ompl_retry(args)
    elif args.ompl_inflation:
        run_ompl_inflation(args)
    elif args.ompl_figures:
        run_ompl_figures(args)
    elif args.ompl:
        run_ompl(args)
    elif args.corner:
        run_corner(args)
    elif args.retry:
        run_retry(args)
    else:
        run(args)
