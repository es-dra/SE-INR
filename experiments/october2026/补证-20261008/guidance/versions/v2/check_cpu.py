"""Small independent exact-matrix, adjoint and risk checks; no images/GPU."""
import json
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import numpy as np
import torch
from core import (Geometry, Operator, frame_params, sampling_points,
                  spectrum, gp_mean, random_field, calibrated_scale, NumericalFailure)


def run_checks():
    torch.set_num_threads(2)
    rng=np.random.default_rng(20261008)
    result={}
    # Dense risk identity independent of FFT implementation.
    a=rng.normal(size=(3,5))
    c0=rng.normal(size=(5,5)); c=c0@c0.T/5
    cj0=rng.normal(size=(5,5)); cj=cj0@cj0.T/5
    sigma=.2
    h=cj@a.T@np.linalg.inv(a@cj@a.T+sigma**2*np.eye(3))
    w=np.eye(5)[1:4]
    e=w@(h@a-np.eye(5))
    exact=np.trace(e@c@e.T)+sigma**2*np.sum((w@h)**2)
    n=200000
    x=rng.multivariate_normal(np.zeros(5),c,size=n)
    noise=rng.normal(size=(n,3))*sigma
    err=(x@a.T+noise)@h.T-x
    measured=np.mean(np.sum((err@w.T)**2,axis=1))
    rel=abs(measured-exact)/exact
    assert rel<.012,(exact,measured,rel)
    result['dense_risk_identity']=dict(exact=float(exact),monte_carlo=float(measured),relative_error=float(rel),draws=n)

    g=Geometry(hr=8,scale=4,frames=2,subsamples=2,field=16,shave=1)
    d,angles=frame_params('0801',g)
    op=Operator(g,sampling_points(d,angles,g,'cpu',torch.float64))
    generator=torch.Generator().manual_seed(17)
    field=torch.randn(3,g.field,g.field,generator=generator,dtype=torch.float64)
    y=torch.randn(g.frames,g.lr**2,3,generator=generator,dtype=torch.float64)
    lhs=(op.forward(field)*y).sum()
    rhs=(field*op.adjoint(y)).sum()
    adj_rel=float(abs(lhs-rhs)/max(abs(lhs),abs(rhs),torch.tensor(1e-15)))
    assert adj_rel<1e-12,adj_rel
    result['operator_adjoint']=dict(relative_error=adj_rel,geometry=g.__dict__)

    # Matrix A in one-channel basis; stationary covariance from FFT kernel.
    eye=torch.eye(g.field**2,dtype=torch.float64).reshape(g.field**2,g.field,g.field)
    amat=op.forward(eye).reshape(-1,g.field**2).numpy()
    p=spectrum('gauss_3',.7,g,'cpu',torch.float64)
    kernel=torch.fft.ifft2(p).real.numpy()
    rows,cols=np.unravel_index(np.arange(g.field**2),(g.field,g.field))
    cov=kernel[(rows[:,None]-rows[None,:])%g.field,(cols[:,None]-cols[None,:])%g.field]
    yv=rng.normal(size=amat.shape[0])
    yt=torch.from_numpy(np.repeat(yv.reshape(g.frames,g.lr**2,1),3,axis=2))
    sh,info=gp_mean(yt,op,p,.01,max_iters=600,tol=1e-10)
    mu=yv.mean()
    dense=cov@amat.T@np.linalg.solve(amat@cov@amat.T+.01**2*np.eye(len(yv)),yv-mu)+mu
    dense=dense.reshape(g.field,g.field)[:g.hr,:g.hr]
    cg_error=float(np.max(np.abs(sh[0].numpy()-dense)))
    assert cg_error<1e-7,cg_error
    result['cg_matches_dense_posterior']=dict(max_error=cg_error,iterations=info['iterations'],relative_residual=info['relative_residual'])
    assert info['true_residual_checks']>=1 and info['true_relative_residual']<1e-10
    result['convergence_certified_by_true_residual']=dict(checks=info['true_residual_checks'],
        recursive_relative_residual=info['recursive_relative_residual'],true_relative_residual=info['true_relative_residual'])
    try:
        gp_mean(yt,op,p,.01,max_iters=1,tol=1e-12)
        raise AssertionError('Nonconverged candidate was incorrectly accepted')
    except NumericalFailure as failure:
        assert 'true residual' in str(failure)
        result['nonconvergence_is_failure']=True
    try:
        gp_mean(yt.float(),op,p,.01)
        raise AssertionError('v2 silently accepted float32')
    except NumericalFailure as failure:
        assert 'float64' in str(failure)
        result['float32_input_rejected']=True
    constant,constant_info=gp_mean(torch.ones_like(yt)*.4,op,p,.01)
    assert torch.allclose(constant,torch.ones_like(constant)*.4)
    assert constant_info['iterations']==0
    result['zero_rhs_intercept']=dict(iterations=0,constant_output_preserved=True)

    zero,zi=gp_mean(yt,op,p,0.,max_iters=600,tol=1e-10)
    tiny,ti=gp_mean(yt,op,p,.001,max_iters=600,tol=1e-10)
    assert zi['solver_variance']==ti['solver_variance']==1e-5
    assert torch.equal(zero,tiny)
    hfloor=cov@amat.T@np.linalg.inv(amat@cov@amat.T+1e-5*np.eye(len(yv)))
    physical_zero_noise_term=0.*np.sum(hfloor**2)
    erroneously_using_floor=1e-5*np.sum(hfloor**2)
    assert physical_zero_noise_term==0 and erroneously_using_floor>0
    result['physical_noise_vs_solver_floor']=dict(solver_variance=zi['solver_variance'],
        zero_noise_contribution=float(physical_zero_noise_term),incorrect_floor_noise_contribution=float(erroneously_using_floor),
        zero_and_tiny_noise_same_fixed_y_output=True)

    unit=torch.ones(8,8,dtype=torch.float64);unit[0,0]=0;unit*=64/unit.sum()
    variances=[float(random_field(unit,9000+i,'cpu',torch.float64).square().mean()) for i in range(1000)]
    meanvar=float(np.mean(variances))
    assert abs(meanvar-1)<.02,meanvar
    result['reference_sampler_variance']=dict(measured=meanvar,expected=1.,draws=1000)

    refs=[torch.randn(2,4,3,generator=generator,dtype=torch.float64) for _ in range(4)]
    scale,details=calibrated_scale(torch.ones(2,4,3,dtype=torch.float64),refs,.02)
    assert scale==1e-8
    result['scale_floor']=dict(scale=scale,noise_correction=details['noise_correction'])
    result['status']='pass'
    result['device']='CPU only; no datasets or GPU loaded'
    return result


if __name__=='__main__':
    print(json.dumps(run_checks(),indent=2))
