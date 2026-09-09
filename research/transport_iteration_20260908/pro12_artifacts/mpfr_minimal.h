#pragma once
// Minimal declarations of the public MPFR C ABI, for systems with the runtime
// library but without development headers. Prefer the vendor header when present.
#if __has_include(<mpfr.h>)
#include <mpfr.h>
#else
#include <gmp.h>
extern "C" {
typedef long mpfr_prec_t;
typedef long mpfr_exp_t;
typedef int mpfr_sign_t;
typedef struct {mpfr_prec_t _mpfr_prec; mpfr_sign_t _mpfr_sign; mpfr_exp_t _mpfr_exp; mp_limb_t *_mpfr_d;} __mpfr_struct;
typedef __mpfr_struct mpfr_t[1];
typedef __mpfr_struct *mpfr_ptr;
typedef const __mpfr_struct *mpfr_srcptr;
typedef enum {MPFR_RNDN=0,MPFR_RNDZ=1,MPFR_RNDU=2,MPFR_RNDD=3,MPFR_RNDA=4,MPFR_RNDF=5} mpfr_rnd_t;
void mpfr_init2(mpfr_ptr,mpfr_prec_t);
void mpfr_clear(mpfr_ptr);
int mpfr_set(mpfr_ptr,mpfr_srcptr,mpfr_rnd_t);
int mpfr_set_d(mpfr_ptr,double,mpfr_rnd_t);
int mpfr_set_si(mpfr_ptr,long,mpfr_rnd_t);
int mpfr_set_str(mpfr_ptr,const char*,int,mpfr_rnd_t);
int mpfr_add(mpfr_ptr,mpfr_srcptr,mpfr_srcptr,mpfr_rnd_t);
int mpfr_sub(mpfr_ptr,mpfr_srcptr,mpfr_srcptr,mpfr_rnd_t);
int mpfr_mul(mpfr_ptr,mpfr_srcptr,mpfr_srcptr,mpfr_rnd_t);
int mpfr_div(mpfr_ptr,mpfr_srcptr,mpfr_srcptr,mpfr_rnd_t);
int mpfr_neg(mpfr_ptr,mpfr_srcptr,mpfr_rnd_t);
int mpfr_abs(mpfr_ptr,mpfr_srcptr,mpfr_rnd_t);
int mpfr_sqrt(mpfr_ptr,mpfr_srcptr,mpfr_rnd_t);
int mpfr_exp(mpfr_ptr,mpfr_srcptr,mpfr_rnd_t);
int mpfr_expm1(mpfr_ptr,mpfr_srcptr,mpfr_rnd_t);
int mpfr_log(mpfr_ptr,mpfr_srcptr,mpfr_rnd_t);
int mpfr_erf(mpfr_ptr,mpfr_srcptr,mpfr_rnd_t);
int mpfr_erfc(mpfr_ptr,mpfr_srcptr,mpfr_rnd_t);
int mpfr_const_pi(mpfr_ptr,mpfr_rnd_t);
int mpfr_cmp(mpfr_srcptr,mpfr_srcptr);
int mpfr_cmp_si(mpfr_srcptr,long);
int mpfr_number_p(mpfr_srcptr);
int mpfr_zero_p(mpfr_srcptr);
double mpfr_get_d(mpfr_srcptr,mpfr_rnd_t);
const char *mpfr_get_version(void);
}
#endif
