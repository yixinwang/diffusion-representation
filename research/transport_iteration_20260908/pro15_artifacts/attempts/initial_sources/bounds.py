"""Explicit elementary bounds; no data, fitted phase, or fitted graph inputs."""
import math

A2 = 1.5
KAPPA = .45

def radius(n, edges, delta=.01, directions=16, frequencies=1):
    if n <= 0 or edges < 1 or not 0 < delta < 1 or directions < 4:
        raise ValueError('invalid bound arguments')
    t = math.log(2*edges*directions*frequencies/delta)
    R = 2*math.sqrt(2)*A2
    eps = math.sqrt(2*t/n) + 2*R*t/(3*n)
    return eps/math.cos(math.pi/directions)

def required_graph_n(edges, amplitude=.075, delta=.01):
    signal = amplitude/math.sqrt(2)
    lo, hi = 1, 2
    while 2*radius(hi, edges, delta) >= signal:
        hi *= 2
    while lo < hi:
        mid = (lo+hi)//2
        if 2*radius(mid,edges,delta) < signal: hi = mid
        else: lo = mid+1
    return lo

def sine_pair_information(amplitude=.075, terms=16):
    """KL to product, with a rigorous positive-series remainder upper bound."""
    s=0.
    for k in range(1,terms+1):
        moment = A2**k*(k+1)/(2*k+1)
        context_moment = amplitude**(2*k)*math.comb(2*k,k)/4**k
        s += context_moment*moment**2/((2*k)*(2*k-1))
    rho = A2*amplitude
    tail = rho**(2*terms+2)/((2*terms+2)*(2*terms+1)*(1-rho*rho))
    return s,tail

def matching_fano(n, blocks=4, block_size=720, amplitude=.075):
    m=block_size; k=m//2
    entropy=blocks*(math.lgamma(m+1)-k*math.log(2)-math.lgamma(k+1))
    info,tail=sine_pair_information(amplitude)
    per_array=blocks*k*(info+tail)
    return dict(log_number_matchings=entropy,pair_kl=info,pair_kl_tail_bound=tail,
                per_array_kl_upper=per_array,
                error_probability_lower=max(0.,1-(n*per_array+math.log(2))/entropy),
                necessary_n_for_99pct=math.ceil((.99*entropy-math.log(2))/per_array))

def root_bound(n, root_dim, bins=8, floor=.5, delta=.01):
    r=root_dim-1
    if r==0: return 0.
    return r*bins/floor*(1+math.sqrt(math.log(r/delta)))**2/n

def parameter_bound(n, pairs_per_block, blocks=4, dim=3, lam=.5,
                    kappa=KAPPA, delta=.01, eta_l2_sq=0., eta_sup=0.):
    """Conditional on min eigenvalue >= lam and a correct graph.
    Fourier design is population orthonormal; squared row norm is dim.
    Residual approximation is the population L2 projection error.
    """
    t=math.log(2*blocks*dim/delta)
    R=math.sqrt(dim/lam)*(A2+kappa)
    tau=math.sqrt(2*t/(n*pairs_per_block))+2*R*t/(3*n*pairs_per_block)
    noise=dim*tau*tau/lam
    mse=noise if eta_l2_sq==0 and eta_sup==0 else eta_l2_sq+2*eta_sup**2/lam+2*noise
    return dict(coefficient_mse_bound=mse,
                residual_joint_kl_bound=blocks*pairs_per_block*mse/(2*(1-A2*kappa)),
                conditional_noise_coordinate_radius=tau,
                design_failure_bound=min(1.,2*dim*dim*math.exp(-n*(1-lam)**2/(8*dim*dim))))

def all_bounds():
    M=4*720*719//2
    n=required_graph_n(M)
    return dict(full_dimension=3072,root_dimension=192,blocks=4,block_size=720,
      candidate_edges=M,amplitude=.075,signal_norm=.075/math.sqrt(2),delta_graph=.01,
      radius_2000=radius(2000,M),radius_4000=radius(4000,M),
      sufficient_graph_arrays=n,sufficient_radius=radius(n,M),
      fano_2000=matching_fano(2000),fano_4000=matching_fano(4000),
      root_bound_total_arrays=root_bound(n+4000,192),
      parameter_bound_4000=parameter_bound(4000,360),
      graph_dense_multiplications=2*n*4*720**2,
      graph_dense_additions=2*(n-1)*4*720**2,
      graph_dense_accumulator_bytes=2*4*720**2*8,
      input_bytes_float64=(n+4000)*3072*8,
      small_required_graph_arrays=required_graph_n(2*8*7//2),
      scope='Sufficient bound is conservative, not a necessary rate. Fano is average error over uniformly random full matchings, even with phase known.')

if __name__=='__main__':
    import json
    print(json.dumps(all_bounds(),indent=2))
