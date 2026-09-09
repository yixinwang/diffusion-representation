"""Fabricated numerical checks only; no timing study or data access."""
import importlib.util,json
from pathlib import Path
import numpy as np
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'qalt/experiments/tilted_sampler_cost'))
from candidate import quantile_transport,inverse_transport,forward_logdet,log_density,heun

original=Path(__file__).parent/'pro8_artifacts/positive_class_reference.py'
spec=importlib.util.spec_from_file_location('frozen_reference',original)
reference=importlib.util.module_from_spec(spec);spec.loader.exec_module(reference)


def run():
    results={}
    boundary=np.array([np.nextafter(-5,-np.inf),-5,np.nextafter(-5,np.inf),np.nextafter(5,-np.inf),5,np.nextafter(5,np.inf),-50,50])
    z=np.unique(np.concatenate((np.linspace(-12,12,24001),boundary)))[:,None]
    e=np.linspace(-.6,.6,25)[None,:]
    for name,source,tilt in [('dense_boundary_extreme',z,e),
                            ('random_gaussian',np.random.default_rng(219).normal(size=(256,192)),np.linspace(-.6,.6,192))]:
        value=quantile_transport(source,tilt);expected=reference.quantile_transport(source,tilt)
        error=float(np.max(np.abs(value-expected)))
        assert error<1e-12
        back=inverse_transport(value,tilt);roundtrip=float(np.max(np.abs(back-source)))
        assert roundtrip<2e-12
        ld=forward_logdet(source,tilt)
        density_error=float(np.max(np.abs(log_density(value,tilt)+ld+.5*np.asarray(source)**2+.5*np.log(2*np.pi))))
        assert density_error<1e-12
        if name=='dense_boundary_extreme':
            # Adjacent binary64 inputs may round to equal outputs; no reversal beyond rounding scale.
            differences=np.diff(value,axis=0)
            assert differences.min()>-2e-14
            mask=np.diff(source[:,0])>1e-10
            assert (differences[mask]>0).all()
            results['minimum_adjacent_difference']=float(differences.min())
        results[name]={'max_reference_error':error,'max_roundtrip_error':roundtrip,'max_density_identity_error':density_error,'finite':bool(np.isfinite(value).all())}
    source=np.linspace(-4.9,4.9,2001)[:,None];tilt=np.linspace(-.6,.6,25)[None,:]
    step=1e-4
    derivative=(quantile_transport(source+step,tilt)-quantile_transport(source-step,tilt))/(2*step)
    error=float(np.max(np.abs(np.log(derivative)-forward_logdet(source,tilt))))
    assert error<1e-7
    results['central_difference_logdet_error']=error
    source=np.linspace(-12,12,1001)[:,None];tilt=np.linspace(-.6,.6,25)[None,:]
    results['heun']={}
    for nfe in (4,8,16,32,64):
        value,counts=heun(source,tilt,nfe)
        expected=reference.heun(np.broadcast_to(source,value.shape),tilt,nfe)
        error=float(np.max(np.abs(value-expected)));assert error<1e-12
        results['heun'][nfe]={**counts,'max_reference_error':error}
    return results


if __name__=='__main__':
    report=run();print(json.dumps(report,indent=2))
    # Print only; the archived JSON is immutable and never overwritten.
