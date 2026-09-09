"""Reproduce in a NEW directory without overwriting the delivered evidence."""
import argparse,os,shutil,subprocess,sys
from pathlib import Path

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',required=True,type=Path)
    args=ap.parse_args();out=args.output.resolve()
    if out.exists():raise SystemExit('Refusing existing output directory: '+str(out))
    out.mkdir(parents=True)
    here=Path(__file__).resolve().parent
    for p in here.iterdir():
        if p.suffix in {'.py','.c'} or p.name=='requirements.txt':shutil.copy2(p,out/p.name)
    env=os.environ.copy();env.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
    scripts=['check_math.py','standalone.py','benchmark_compiled.py','benchmark_symmetric.py','benchmark_outer_fit.py','precision_addendum.py','check_bijection.py']
    for name in scripts:
        with (out/(name+'.run.stdout.txt')).open('w') as so,(out/(name+'.run.stderr.txt')).open('w') as se:
            subprocess.run([sys.executable,name],cwd=out,env=env,stdout=so,stderr=se,check=True)
    print(out)

if __name__=='__main__':main()
