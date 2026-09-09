import sys,pathlib,json,math,numpy as np
from scipy.special import ndtri
here=pathlib.Path(__file__).resolve().parent;sys.path.insert(0,str(here/'replay'))
from copula import Model,expected_kl,copula_kl
src=pathlib.Path(sys.argv[1]);old=json.loads((src/'results.json').read_text());new=json.loads((here/'replay/results.json').read_text());n=4096;B=8;K=32;a=.85;rho=.9
AB=a*a/2*(1-(math.sin(math.pi/B)/(math.pi/B))**2);upper=K/(54*(1-rho))*((1+B/n)*AB+27*B/(n*K));floor=K*a*a/108;conservative=floor-K/(54*(1-rho))*((1+B/n)*a*a*10/(6*B*B)+27*B/(n*K))
rows=[]
for o,r in zip(old['scenarios'],new['scenarios']):
 rows.append({'scenario':r['scenario'],'training_hash_equal':r['observed_train_sha256_little_endian_float64']==o['observed_train_sha256_little_endian_float64'],'maximum_parameter_difference':max(float(np.max(np.abs(np.asarray(r['models'][k])-np.asarray(o['models'][k])))) for k in r['models']),'maximum_risk_difference':max(abs(v-o['population_kl_nats_per_full_observation'][k]) for k,v in r['population_kl_nats_per_full_observation'].items()),'risks':r['population_kl_nats_per_full_observation'],'sample_copy_difference':r['copy_max_sample_difference'],'logprob_copy_difference':r['copy_max_log_prob_difference']})
# Additional bounded negative API probes, no fit or stress regeneration.
model=Model(np.array([[.2],[.7]]),16,32);z=np.zeros((1,113));z[0,1]=np.nan
nan_dec=model.decode(z)
bad_model=Model(np.array([[np.nan]]),16,32)
# Demonstrate discontinuity at root c=.5 with all other coordinates fixed.
z0=np.zeros((1,113));z0[0,17:20]=[.5,.7,.2];left=z0.copy();right=z0.copy();left[0,0]=ndtri(.5-1e-10);right[0,0]=ndtri(.5+1e-10)
gap=float(abs(model.decode(left)[0,19]-model.decode(right)[0,19]))
# Failure identity: average KL of opposing parameters minus product risk is KL(p0||pe).
grid=np.linspace(-.85,.85,41);e=np.linspace(-.9,.9,41);identity=np.max(np.abs(.5*(copula_kl(grid,e)+copula_kl(-grid,e))-copula_kl(grid,0)-copula_kl(0,e)))
d={'n':n,'sites':K,'AB':AB,'expected_KL_upper':upper,'product_floor':floor,'expected_gain_lower':floor-upper,'conservative_gain_lower':conservative,'constant_agreement':abs(upper-json.loads((src/'FINITE_SAMPLE_CERTIFICATE.json').read_text())['expected_KL_upper_shared'])<1e-15,'local_vs_original':rows,'initial_rng_exact':old['rng_initial']==new['rng_initial'],'final_rng_exact':old['rng_final']==new['rng_final'],'versions':{'original':[old['python'],old['numpy'],old['scipy']],'local':[new['python'],new['numpy'],new['scipy']]},'additional_limitations':{'decode_accepts_nan_anchor_and_returns_nonfinite':bool(np.isnan(nan_dec[0,1])),'constructor_accepts_nan_theta_and_logprob_returns_nan':bool(np.isnan(bad_model.log_prob(np.zeros((1,113)))[0])),'piecewise_context_map_jump_example':gap},'alternating_sign_identity_max_roundoff':float(identity)}
assert d['constant_agreement'] and conservative>.09 and identity<1e-14
(here/'independent_check.json').write_text(json.dumps(d,indent=2,allow_nan=False)+'\n');print(json.dumps(d,indent=2))
