#!/usr/bin/env python3
"""Exact rational checks for constants used by the scoped Pro8 interval audit.

Uses Machin's identity pi=16 atan(1/5)-4 atan(1/239) and the alternating
arctangent-series remainder. No floating transcendental evaluation is used.
This verifies constants; it does not replace the interval Heun propagation.
"""
from fractions import Fraction as F
import json
import math


def arctan_bounds(denominator, terms=64):
    x=F(1,denominator)
    total=sum(((-1)**j*x**(2*j+1)/F(2*j+1) for j in range(terms)),F(0))
    following=(-1)**terms*x**(2*terms+1)/F(2*terms+1)
    return min(total,total+following),max(total,total+following)


def check():
    a,b=arctan_bounds(5);c,d=arctan_bounds(239)
    pi_lo,pi_hi=16*a-4*d,16*b-4*c
    lower=F('0.39894228040143267793');upper=F('0.39894228040143267795')
    exp_remainder=F(1,2)**19/math.factorial(19)
    integral_remainder=exp_remainder/39
    e_lower=sum((F(1,math.factorial(j)) for j in range(21)),F(0))
    exp018_lower=sum((F(18,100)**j/math.factorial(j) for j in range(4)),F(0))
    checks={
        'pi_bracket_order':F(3)<pi_lo<pi_hi<F(22,7),
        'normal_constant_lower':2*pi_hi*lower**2<1,
        'normal_constant_upper':2*pi_lo*upper**2>1,
        'exponential_taylor_remainder_below_1e_minus22':exp_remainder<F(1,10**22),
        'integrated_taylor_remainder_below_1e_minus22':integral_remainder<F(1,10**22),
        'exp_negative_point18_less_than_point84':exp018_lower>1/F('0.84'),
        'positive_derivative_lemma_margin':F('2.4')**2<6,
        'global_field_derivative_upper_less_than_point544435':F('0.544435')**2*2*pi_lo*e_lower>F('2.25')**2,
    }
    if not all(checks.values()):raise AssertionError(checks)
    return {'scope':'Exact rational constant checks; not the Heun interval propagation or arbitrary floating implementation.',
            'arctan_terms':64,'checks':checks,'all_passed':True}


if __name__=='__main__':print(json.dumps(check(),indent=2))
