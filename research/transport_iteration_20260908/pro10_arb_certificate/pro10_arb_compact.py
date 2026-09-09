"""Local scalar Arb feasibility check; no observations/fits/PSC. New output only."""
from pathlib import Path
import sys,json,time,traceback,argparse,hashlib
sys.path.insert(0,str(Path(__file__).resolve().parent/'pro10-arb-deps'))
from flint import arb,acb,ctx
import flint
p=argparse.ArgumentParser();p.add_argument('--calls',type=int,choices=(4,8,16,32,64),default=4);p.add_argument('--average',action='store_true');p.add_argument('--seconds',type=int,default=45);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
if args.output.exists():raise FileExistsError(args.output)
ctx.prec=96
SQRT2=arb(2).sqrt();C=1/(2*arb.pi()).sqrt();start=time.monotonic();counts={'integrand':0,'outer':0};parts=[]

def timeout():
    if time.monotonic()-start>args.seconds:raise TimeoutError('bounded local certificate attempt exhausted')

def phi(x):return (-x*x/2).exp()*C

def field(y,t,e):
    a=1-t;s=a*a+t*t;d=(2*a*a+t*t).sqrt();k=t/d;u=k*y
    density=phi(u);den=1+e*(u/SQRT2).erf()
    w=2*e*a/(s*d)*density/den
    return w,-k*w*(u+2*e*density/den)


def integrand(z,e,analytic):
    counts['integrand']+=1;timeout();y=z;j=acb(1);h=arb(2)/args.calls
    for i in range(args.calls//2):
        w,a=field(y,i*h,e);v,b=field(y+h*w,(i+1)*h,e)
        y=y+h*(w+v)/2;j=j*(1+h*(a+b*(1+h*a))/2)
    tilt=1+e*(y/SQRT2).erf()
    ell=-(y-z)*(y+z)/2+tilt.log(analytic=analytic)+j.log(analytic=analytic)
    return phi(z)*(ell.exp()*ell-ell.expm1())


def z_integral(e,outer_analytic=False,save=False):
    total=acb(0)
    for lo in range(-10,10):
        value=acb.integral(lambda z,a:integrand(z,e,a or outer_analytic),lo,lo+1,
           abs_tol=arb('1e-15'),rel_tol=arb('1e-10'),eval_limit=2000,depth_limit=20)
        total+=value
        if save:parts.append({'lo':lo,'hi':lo+1,'ball':str(value)})
    return total


def outer(e,analytic):
    counts['outer']+=1;timeout();return 5*z_integral(e,analytic)

report={'library':'python-flint','version':flint.__version__,'precision_bits':ctx.prec,
 'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
 'scope':'uniform e average' if args.average else 'fixed exact e=1/2 only',
 'calls':args.calls,'cutoff':10,'max_seconds':args.seconds,'formal_proof_assistant_verified':False}
try:
    compact=acb.integral(outer,acb(arb(2)/5),acb(arb(3)/5),abs_tol=arb('1e-12'),rel_tol=arb('1e-8'),eval_limit=300,depth_limit=12) if args.average else z_integral(acb(arb(1)/2),save=True)
    A=arb(17)/10;b=arb(7)/2;T=arb(10)
    bar=lambda x:(x/SQRT2).erfc()/2
    tail=2*(b+A*A/2).exp()*(A*phi(T-A)+(A*A+b+1)*bar(T-A))+2*bar(T)
    joint=compact.real*2880;upper=joint.upper()+tail.upper()*2880
    report.update(status='finite_enclosure' if compact.is_finite() else 'nonfinite_failure',
      compact_joint_ball=str(joint),imaginary_ball=str(compact.imag),joint_tail_upper_ball=str(tail.upper()*2880),
      full_joint_upper_ball=str(upper),upper_less_than_one=bool(upper<1),
      upper_less_than_0p00002=bool(upper<arb(1)/50000),
      conditional_certificate_scope='Arb enclosure under correct analytic callbacks and analytic tail proof; independent code review still required')
except BaseException as e:report.update(status='failed',error=repr(e),traceback=traceback.format_exc())
finally:
    report.update(elapsed_seconds=time.monotonic()-start,counts=counts,completed_segments=parts)
    args.output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
