import unittest,tempfile,os,ast,json
from pathlib import Path
import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.special import ndtr
from model import (A,KAPPA,Config,Model,ExactCopyDecoder,psi,integral_psi,pair_decode,
                   pair_encode,design,project_simplex_floor)
from discover import fit,isolated_edges
from fixture import make_truth,observe,theta
from evaluate import entropy_and_derivative,context_rule,population,direct_population
from bounds import radius,required_graph_n,matching_fano,all_bounds

class Checks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.loc=Path(os.environ.get('PRO15_TEST_OUTPUT','results/test_fixtures'))
        cls.loc.mkdir(parents=True,exist_ok=False)
        rng=np.random.default_rng(151111)
        cls.u=rng.random(2048);cls.p=rng.random(2048);cls.t=rng.uniform(-KAPPA,KAPPA,2048)
        cls.v,cls.pld=pair_decode(cls.u,cls.p,cls.t)
        cls.cfg=Config(root_dim=2,blocks=1,block_size=4,graph_arrays=32,parameter_arrays=32)
        cls.truth=make_truth(151112,cls.cfg,'harmonic')
        cls.z=rng.standard_normal((128,cls.cfg.dimension));cls.x=observe(cls.z,cls.truth,cls.cfg)
        cls.model=Model(cls.cfg,cls.truth['root'],cls.truth['pairs'],[[.01,.02,-.03]])
        cls.out,cls.ld=cls.model.sample(cls.z)
        cls.model.save(cls.loc/'model.json')
        (cls.loc/'truth.json').write_text(json.dumps(cls.truth,indent=2)+'\n')
        np.savez_compressed(cls.loc/'arrays.npz',u=cls.u,p=cls.p,theta=cls.t,v=cls.v,pair_logdet=cls.pld,
            source=cls.z,observed=cls.x,output=cls.out,forward_logdet=cls.ld)
    def test_01_feature_mean(self):
        z,w=leggauss(12);pts=[];ww=[]
        from model import KNOTS
        for l,r in zip(KNOTS[:-1],KNOTS[1:]):pts.extend(l+(z+1)*(r-l)/2);ww.extend(w*(r-l)/2)
        p=np.asarray(pts);w=np.asarray(ww)
        np.savez(self.loc/'quadrature.npz',points=p,weights=w)
        self.assertLess(abs(w@psi(p)),1e-14)
    def test_02_feature_second(self):
        a=np.load(self.loc/'quadrature.npz');self.assertAlmostEqual(float(a['weights']@psi(a['points'])**2),1.,places=13)
    def test_03_feature_third(self):
        a=np.load(self.loc/'quadrature.npz');self.assertLess(abs(a['weights']@psi(a['points'])**3),1e-14)
    def test_04_ordinary_covariance_zero(self):
        a=np.load(self.loc/'quadrature.npz');self.assertLess(abs(a['weights']@(a['points']*psi(a['points']))),1e-14)
    def test_05_integral_endpoints(self):
        self.assertTrue(np.array_equal(integral_psi(np.array([0.,1.])),np.array([0.,0.])))
    def test_06_inverse(self):
        p,ld=pair_encode(self.u,self.v,self.t);self.assertLess(np.max(abs(p-self.p)),1e-14)
    def test_07_pair_logdet(self):
        _,ld=pair_encode(self.u,self.v,self.t);self.assertLess(np.max(abs(ld+self.pld)),1e-14)
    def test_08_endpoints(self):
        u=np.array([0.,.2,.5,.8,1.])[:,None,None];p=np.array([0.,np.nextafter(0.,1.),np.nextafter(1.,0.),1.])[None,:,None];t=np.array([-.45,0,.45])[None,None,:]
        v,ld=pair_decode(u,p,t);pp,_=pair_encode(u,v,t)
        np.savez(self.loc/'endpoints.npz',u=u,p=p,theta=t,v=v,roundtrip=pp)
        self.assertTrue(np.all(v[:,0,:]==0) and np.all(v[:,-1,:]==1));self.assertLess(np.max(abs(pp-p)),1e-14)
    def test_09_gaussian_roundtrip(self):
        z,ld=self.model.encode(self.out);np.savez(self.loc/'inverse.npz',recovered=z,inverse_logdet=ld)
        self.assertLess(np.max(abs(z-self.z)),1e-11)
    def test_10_gaussian_logdet(self):
        _,ild=self.model.encode(self.out);self.assertLess(np.max(abs(self.ld+ild)),1e-10)
    def test_11_density(self):
        lp=self.model.log_prob(self.out)
        self.assertLess(np.max(abs(lp+self.ld-(-.5*self.z**2-.5*np.log(2*np.pi)).sum(1))),1e-10)
    def test_12_exact_copy(self):
        x,ld=ExactCopyDecoder(self.model).sample(self.z)
        self.assertTrue(np.array_equal(x,self.out) and np.array_equal(ld,self.ld))
    def test_13_state_replay(self):
        x,ld=Model.load(self.loc/'model.json').sample(self.z);self.assertTrue(np.array_equal(x,self.out))
    def test_14_finite_difference_jacobian(self):
        z=np.full((1,6),.13);x,ld=self.model.sample(z);J=np.zeros((6,6));eps=1e-5
        for k in range(6):
            dz=np.zeros_like(z);dz[0,k]=eps;J[:,k]=((self.model.sample(z+dz)[0]-self.model.sample(z-dz)[0])/(2*eps))[0]
        np.savez(self.loc/'jacobian.npz',source=z,output=x,J=J,forward_logdet=ld)
        self.assertLess(abs(np.linalg.slogdet(J)[1]-ld[0]),1e-7)
    def test_15_saturation_rejected(self):
        with self.assertRaises(ValueError):self.model.sample(np.full((1,6),50.))
    def test_16_nonfinite_rejected(self):
        with self.assertRaises(ValueError):self.model.sample(np.full((1,6),np.nan))
    def test_17_bad_theta_rejected(self):
        with self.assertRaises(ValueError):pair_decode(.2,.3,.451)
    def test_18_overlapping_edges_rejected(self):
        with self.assertRaises(ValueError):Model(self.cfg,self.model.root,[[[0,1],[1,2]]],[[0,0,0]])
    def test_19_noninteger_edges_rejected(self):
        with self.assertRaises(ValueError):Model(self.cfg,self.model.root,[[[0.,1.]]],[[0,0,0]])
    def test_20_first_root_assumption(self):
        p=self.model.root.copy();p[0,0]+=.01;p[0,1]-=.01
        with self.assertRaises(ValueError):Model(self.cfg,p,self.model.pairs,self.model.coef)
    def test_21_immutable_state(self):
        with self.assertRaises(ValueError):self.model.coef[0,0]=0
    def test_22_simplex_projection(self):
        p=project_simplex_floor(np.array([.8,.1,.1,0]),.5)
        self.assertAlmostEqual(p.sum(),1,places=14);self.assertTrue(np.all(p>=.125))
    def test_23_isolation_filter(self):
        a=np.zeros((4,4),bool);a[0,1]=a[1,0]=True;a[1,2]=a[2,1]=True
        self.assertEqual(len(isolated_edges(a)),0)
    def test_24_null_population_signal(self):
        c,w=context_rule(8,64);tr=make_truth(151112,self.cfg,'null')
        self.assertTrue(np.array_equal(theta(c,tr),np.zeros((len(c),1))))
    def test_25_off_basis_projection(self):
        c,w=context_rule(8,64);tr=make_truth(151112,self.cfg,'off_basis');v=design(c).T@(w[:,None]*theta(c,tr))
        self.assertLess(np.max(abs(v)),1e-12)
    def test_26_unknown_phase_signal(self):
        c,w=context_rule(8,64);v=design(c).T@(w[:,None]*theta(c,self.truth))
        self.assertAlmostEqual(np.linalg.norm(v[1:]),.075/np.sqrt(2),places=12)
    def test_27_population_crosscheck(self):
        p=population(self.truth,self.model);d=direct_population(self.truth,self.model)
        self.assertLess(abs(p['joint_kl']-d),1e-11)
    def test_28_deterministic_kl_bound(self):
        p=population(self.truth,self.model);self.assertLessEqual(p['joint_kl'],p['deterministic_joint_upper']+1e-12)
    def test_29_fano_lower_bound(self):
        self.assertGreater(matching_fano(2000)['error_probability_lower'],.49)
    def test_30_4000_not_excluded_by_fano(self):
        self.assertEqual(matching_fano(4000)['error_probability_lower'],0.)
    def test_31_recovery_threshold(self):
        M=1035360;n=required_graph_n(M)
        self.assertLess(2*radius(n,M),.075/np.sqrt(2));self.assertGreaterEqual(2*radius(n-1,M),.075/np.sqrt(2))
    def test_32_contract_no_truth_import(self):
        tree=ast.parse((Path(__file__).resolve().parents[1]/'src/discover.py').read_text())
        imports=[n.module for n in ast.walk(tree) if isinstance(n,ast.ImportFrom)]
        self.assertNotIn('fixture',imports);self.assertNotIn('evaluate',imports)
        fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='fit')
        self.assertEqual([a.arg for a in fn.args.args],['x','cfg','arm'])
    def test_33_small_fit_preserved(self):
        m,s=fit(self.x[:64],self.cfg,'spectral');m.save(self.loc/'small_fit.json');np.savez_compressed(self.loc/'small_fit_arrays.npz',**s)
        self.assertTrue(np.array_equal(m.root[0],np.full(8,.125)))
    def test_34_wrong_budget_rejected(self):
        with self.assertRaises(ValueError):fit(self.x[:63],self.cfg,'spectral')
    def test_35_histogram_normalized(self):
        m=Model(self.cfg,self.model.root,self.model.pairs,np.full((1,8),.45),'histogram');m.save(self.loc/'histogram_model.json')
        x,ld=m.sample(self.z);r,ild=m.encode(x);np.savez(self.loc/'histogram_outputs.npz',output=x,forward_logdet=ld,recovered=r,inverse_logdet=ild)
        self.assertLess(np.max(abs(r-self.z)),1e-10)
    def test_36_invalid_config(self):
        with self.assertRaises(ValueError):Config(block_size=3)

if __name__=='__main__':unittest.main(verbosity=2)
