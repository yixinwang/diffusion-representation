// Pro12: real interval Taylor / composite Gauss certificate. No complex branches.
// Every interval endpoint operation, including transcendentals, is MPFR-directed.
#include "mpfr_minimal.h"
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>
static int PREC=128;
struct Real {
    mpfr_t x;
    Real(){mpfr_init2(x,PREC);mpfr_set_si(x,0,MPFR_RNDN);}
    Real(const Real&o){mpfr_init2(x,PREC);mpfr_set(x,o.x,MPFR_RNDN);}
    Real&operator=(const Real&o){if(this!=&o)mpfr_set(x,o.x,MPFR_RNDN);return *this;}
    ~Real(){mpfr_clear(x);}
};
struct I {
    Real l,u;
    I(){}
    I(int a){mpfr_set_si(l.x,a,MPFR_RNDN);mpfr_set_si(u.x,a,MPFR_RNDN);}
    explicit I(double a){mpfr_set_d(l.x,a,MPFR_RNDD);mpfr_set_d(u.x,a,MPFR_RNDU);}
    explicit I(const char*a){mpfr_set_str(l.x,a,10,MPFR_RNDD);mpfr_set_str(u.x,a,10,MPFR_RNDU);}
    I(const Real&a,const Real&b):l(a),u(b){}
    bool positive()const{return mpfr_cmp_si(l.x,0)>0;}
    bool finite()const{return mpfr_number_p(l.x)&&mpfr_number_p(u.x);}
    double lower()const{return std::nextafter(mpfr_get_d(l.x,MPFR_RNDD),-std::numeric_limits<double>::infinity());}
    double upper()const{return std::nextafter(mpfr_get_d(u.x,MPFR_RNDU),std::numeric_limits<double>::infinity());}
};
I operator+(const I&a,const I&b){I c;mpfr_add(c.l.x,a.l.x,b.l.x,MPFR_RNDD);mpfr_add(c.u.x,a.u.x,b.u.x,MPFR_RNDU);return c;}
I operator-(const I&a){I c;mpfr_neg(c.l.x,a.u.x,MPFR_RNDD);mpfr_neg(c.u.x,a.l.x,MPFR_RNDU);return c;}
I operator-(const I&a,const I&b){return a+(-b);}
I operator*(const I&a,const I&b){
    I c;Real p,q;
    mpfr_mul(c.l.x,a.l.x,b.l.x,MPFR_RNDD);mpfr_mul(c.u.x,a.l.x,b.l.x,MPFR_RNDU);
    mpfr_srcptr aa[2]={a.l.x,a.u.x},bb[2]={b.l.x,b.u.x};
    for(int i=0;i<2;i++)for(int j=0;j<2;j++){
        mpfr_mul(p.x,aa[i],bb[j],MPFR_RNDD);if(mpfr_cmp(p.x,c.l.x)<0)c.l=p;
        mpfr_mul(q.x,aa[i],bb[j],MPFR_RNDU);if(mpfr_cmp(q.x,c.u.x)>0)c.u=q;
    }return c;
}
I inv(const I&a){if(mpfr_cmp_si(a.l.x,0)<=0&&mpfr_cmp_si(a.u.x,0)>=0)throw std::runtime_error("division enclosure meets zero");I c;I one(1);mpfr_div(c.l.x,one.l.x,a.u.x,MPFR_RNDD);mpfr_div(c.u.x,one.u.x,a.l.x,MPFR_RNDU);return c;}
I operator/(const I&a,const I&b){return a*inv(b);}
I sqr(const I&a){I c=a*a;if(mpfr_cmp_si(a.l.x,0)<=0&&mpfr_cmp_si(a.u.x,0)>=0)mpfr_set_si(c.l.x,0,MPFR_RNDN);return c;}
I hull(const I&a,const I&b){return I(mpfr_cmp(a.l.x,b.l.x)<0?a.l:b.l,mpfr_cmp(a.u.x,b.u.x)>0?a.u:b.u);}
I absbound(const I&a){I c;Real p,q;mpfr_abs(p.x,a.l.x,MPFR_RNDU);mpfr_abs(q.x,a.u.x,MPFR_RNDU);c.u=mpfr_cmp(p.x,q.x)>0?p:q;c.l=c.u;return c;}
I upperpoint(const I&a){return I(a.u,a.u);}
I lowerpoint(const I&a){return I(a.l,a.l);}
I midpoint(const I&a){I c=(lowerpoint(a)+upperpoint(a))/I(2);return c;}
#define MONO(NAME,MP) I NAME(const I&a){I c;MP(c.l.x,a.l.x,MPFR_RNDD);MP(c.u.x,a.u.x,MPFR_RNDU);if(!c.finite())throw std::runtime_error(#NAME " nonfinite");return c;}
MONO(expI,mpfr_exp) MONO(expm1I,mpfr_expm1) MONO(erfI,mpfr_erf)
I logI(const I&a){if(!a.positive())throw std::runtime_error("nonpositive log argument");I c;mpfr_log(c.l.x,a.l.x,MPFR_RNDD);mpfr_log(c.u.x,a.u.x,MPFR_RNDU);return c;}
I sqrtI(const I&a){if(mpfr_cmp_si(a.l.x,0)<0)throw std::runtime_error("negative square root");I c;mpfr_sqrt(c.l.x,a.l.x,MPFR_RNDD);mpfr_sqrt(c.u.x,a.u.x,MPFR_RNDU);return c;}
I erfcI(const I&a){I c;mpfr_erfc(c.l.x,a.u.x,MPFR_RNDD);mpfr_erfc(c.u.x,a.l.x,MPFR_RNDU);return c;}
I powI(I a,int n){I c(1);for(;n;n>>=1,a=a*a)if(n&1)c=c*a;return c;}
I piI(){I c;mpfr_const_pi(c.l.x,MPFR_RNDD);mpfr_const_pi(c.u.x,MPFR_RNDU);return c;}
I factorial(int n){I v(1);for(int j=2;j<=n;j++)v=v*I(j);return v;}
std::ostream&operator<<(std::ostream&os,const I&a){return os<<"["<<std::setprecision(17)<<a.lower()<<","<<a.upper()<<"]";}
// Univariate normalized derivatives: a[k] encloses f^(k)/k! everywhere
// on the interval represented by a[0]. Operations are finite algebraic recurrences.
static int ORDER=12;
struct Jet {
    std::vector<I>a;
    Jet():a(ORDER+1){}
    Jet(int v):a(ORDER+1){a[0]=I(v);}
    Jet(const I&v):a(ORDER+1){a[0]=v;}
};
Jet operator+(const Jet&a,const Jet&b){Jet c;for(int k=0;k<=ORDER;k++)c.a[k]=a.a[k]+b.a[k];return c;}
Jet operator-(const Jet&a){Jet c;for(int k=0;k<=ORDER;k++)c.a[k]=-a.a[k];return c;}
Jet operator-(const Jet&a,const Jet&b){return a+(-b);}
Jet operator*(const Jet&a,const Jet&b){Jet c;for(int k=0;k<=ORDER;k++)for(int j=0;j<=k;j++)c.a[k]=c.a[k]+a.a[j]*b.a[k-j];return c;}
Jet operator*(const Jet&a,const I&b){Jet c;for(int k=0;k<=ORDER;k++)c.a[k]=a.a[k]*b;return c;}
Jet operator*(const I&a,const Jet&b){return b*a;}
Jet operator/(const Jet&a,const Jet&b){Jet c;I ib=inv(b.a[0]);for(int k=0;k<=ORDER;k++){I s=a.a[k];for(int j=1;j<=k;j++)s=s-b.a[j]*c.a[k-j];c.a[k]=s*ib;}return c;}
Jet expJ(const Jet&a){Jet c;c.a[0]=expI(a.a[0]);for(int k=1;k<=ORDER;k++){I s;for(int j=1;j<=k;j++)s=s+I(j)*a.a[j]*c.a[k-j];c.a[k]=s/I(k);}return c;}
Jet logJ(const Jet&a){Jet da;for(int j=0;j<ORDER;j++)da.a[j]=I(j+1)*a.a[j+1];Jet q=da/a;Jet c;c.a[0]=logI(a.a[0]);for(int j=1;j<=ORDER;j++)c.a[j]=q.a[j-1]/I(j);return c;}
Jet erfJ(const Jet&a){Jet b=expJ(-(a*a));Jet c;c.a[0]=erfI(a.a[0]);I two_sqrtpi=I(2)/sqrtI(piI());for(int k=1;k<=ORDER;k++){I s;for(int j=1;j<=k;j++)s=s+I(j)*a.a[j]*b.a[k-j];c.a[k]=s*two_sqrtpi/I(k);}return c;}
Jet FJ(const Jet&ell){
    Jet ex=expJ(ell), product=ell*ex, der;
    Jet f; f.a[0]=ell.a[0]*ex.a[0]-expm1I(ell.a[0]);
    // F'=ell*exp(ell); use chain rule to avoid cancellation in higher coefficients.
    for(int k=1;k<=ORDER;k++){I s;for(int j=1;j<=k;j++)s=s+I(j)*ell.a[j]*product.a[k-j];f.a[k]=s/I(k);}return f;
}
struct Plan {int N;I h, invsqrtpi, invsqrt2pi, invsqrt2;std::vector<I>k,A;
    Plan(int n):N(n),h(I(2)/I(n)),invsqrtpi(inv(sqrtI(piI()))),invsqrt2pi(inv(sqrtI(I(2)*piI()))),invsqrt2(inv(sqrtI(I(2)))){
        for(int i=0;i<=n/2;i++){I t=I(i)*h,a=I(1)-t,s=a*a+t*t,d=sqrtI(I(2)*a*a+t*t);k.push_back(t/d);A.push_back(I(2)*a/(s*d)*invsqrt2pi);}
    }
};
std::pair<Jet,Jet> field(const Jet&y,const Jet&e,const Plan&p,int t){
    Jet q=y*p.k[t];Jet ph=expJ(-(q*q)*I("0.5"))*p.invsqrt2pi;
    Jet den=Jet(1)+e*erfJ(q*p.invsqrt2);
    Jet w=(e*expJ(-(q*q)*I("0.5")))*p.A[t]/den;
    Jet dy=-(w*(q+(e*ph)*I(2)/den))*p.k[t];
    return {w,dy};
}
Jet integrand(Jet z,Jet e,const Plan&p){
    Jet y=z,J(1);
    for(int i=0;i<p.N/2;i++){
        Jet v,dy;
        if(i==0){v=e*p.invsqrtpi;}else{auto f=field(y,e,p,i);v=f.first;dy=f.second;}
        Jet fac;
        if(i+1==p.N/2){fac=Jet(1)+dy*(p.h/I(2));y=y+v*(p.h/I(2));}
        else {auto f2=field(y+v*p.h,e,p,i+1);fac=Jet(1)+(dy+f2.second*(Jet(1)+dy*p.h))*(p.h/I(2));y=y+(v+f2.first)*(p.h/I(2));}
        if(!fac.a[0].positive())throw std::runtime_error("step Jacobian not certified positive");
        J=J*fac;
    }
    Jet ell=-((y-z)*(y+z))*I("0.5")+logJ(Jet(1)+e*erfJ(y*p.invsqrt2))+logJ(J);
    return expJ(-(z*z)*I("0.5"))*p.invsqrt2pi*FJ(ell);
}
// Legendre recurrence and verified brackets: sign changes isolate n disjoint roots.
std::pair<I,I> legendre(int n,const I&x){I p0(1),p1=x;if(n==0)return{p0,I(0)};for(int k=2;k<=n;k++){I pn=(I(2*k-1)*x*p1-I(k-1)*p0)/I(k);p0=p1;p1=pn;}I dp=I(n)*(x*p1-p0)/(x*x-I(1));return{p1,dp};}
std::vector<std::pair<I,I>> gauss(int n){
    std::vector<std::pair<I,I>> out;
    for(int k=1;k<=n;k++){
        // Double approximation is only a proposal. Actual sign brackets and
        // disjointness prove completeness, independent of the proposal accuracy.
        double x=std::cos(3.14159265358979323846*(k-.25)/(n+.5));
        for(int j=0;j<12;j++){double p0=1,p1=x;for(int m=2;m<=n;m++){double p=((2*m-1)*x*p1-(m-1)*p0)/m;p0=p1;p1=p;}double dp=n*(x*p1-p0)/(x*x-1);x-=p1/dp;}
        I a=I(x)-I("1e-10"),b=I(x)+I("1e-10");a=lowerpoint(a);b=upperpoint(b);
        auto sign=[&](const I&v){if(v.lower()>0)return 1;if(v.upper()<0)return -1;return 0;};
        int sa=sign(legendre(n,a).first),sb=sign(legendre(n,b).first);if(!sa||!sb||sa==sb)throw std::runtime_error("root bracket not certified");
        for(int j=0;j<80;j++){I m=midpoint(hull(a,b));I pm=legendre(n,m).first;int sm=sign(pm);if(!sm)break;if(sm==sa)a=m;else b=m;}
        I root=hull(a,b);I der=legendre(n,root).second;I weight=I(2)/((I(1)-root*root)*der*der);
        if(!weight.positive())throw std::runtime_error("Gauss weight nonpositive");out.push_back({root,weight});
    }
    std::sort(out.begin(),out.end(),[](const auto&a,const auto&b){return a.first.lower()<b.first.lower();});
    for(int k=1;k<n;k++)if(out[k-1].first.upper()>=out[k].first.lower())throw std::runtime_error("root brackets overlap");
    I wsum;for(auto&v:out)wsum=wsum+v.second;if(wsum.lower()>2||wsum.upper()<2)throw std::runtime_error("weight sum missing 2");
    return out;
}
struct Cell {I zl,zu,el,eu;int depth=0;};
I tail(const Plan&p){I B("1.7"),A("3.5"),T(12),x=T-B;I phi=expI(-x*x/I(2))*p.invsqrt2pi;
 return I(2880)*(I(2)*expI(A+B*B/I(2))*(B*phi+(B*B+A+I(1))*erfcI(x*p.invsqrt2)/I(2))+erfcI(T*p.invsqrt2));}
I correction(const Plan&p){I B("1.7"),c=B/I(2);I M=B*(c*(I(1)+erfI(c*p.invsqrt2))/I(2)+expI(-c*c/I(2))*p.invsqrt2pi)+I("0.12")+I("1.1");
 I pg=expI(-I(32)*I(192)/I(72)),rho=I(1)-I(9)/I(16)*(I(1)-expI(-I(2880)*I("0.1")*I("0.1")*I("0.5")*I("0.5")/(I(24)*I("1.6")))),pj=I(15)*powI(rho,256);
 std::cout<<"\"M_wrong\":"<<M<<",\"pg\":"<<pg<<",\"pj\":"<<pj<<",\"exact_expected_upper\":"<<I(96)*pj+I(224)*pg<<",";
 return I(2880)*M*pj+(I(2880)*M+I(128))*pg;
}
int main(int argc,char**argv){
 try {
 int N=argc>1?std::stoi(argv[1]):4,gn=argc>2?std::stoi(argv[2]):6;
 int nz=argc>3?std::stoi(argv[3]):48,ne=argc>4?std::stoi(argv[4]):4;
 std::string tol=argc>5?argv[5]:"5e-9";
 if(argc>6)PREC=std::stoi(argv[6]);
 if(N!=4&&N!=8&&N!=16&&N!=32&&N!=64)throw std::runtime_error("fixed N only");
 auto start=std::chrono::steady_clock::now();Plan p(N);auto nodes=gauss(gn);
 I cn=powI(factorial(gn),4)/(I(2*gn+1)*powI(factorial(2*gn),2));
 std::vector<Cell> todo;
 for(int i=0;i<nz;i++)for(int j=0;j<ne;j++)todo.push_back({I(-12)+I(24)*I(i)/I(nz),I(-12)+I(24)*I(i+1)/I(nz),I(2)/I(5)+I(j)/(I(5)*I(ne)),I(2)/I(5)+I(j+1)/(I(5)*I(ne)),0});
 I total,errtotal;int good=0,split=0,failed=0,maxdepth=0;long evals=0;
 while(!todo.empty()){
  Cell c=todo.back();todo.pop_back();I dz=c.zu-c.zl,de=c.eu-c.el;
  I error;bool valid=true;
  try{
   ORDER=2*gn;Jet z(hull(c.zl,c.zu)),e(hull(c.el,c.eu));z.a[1]=I(1);Jet fz=integrand(z,e,p);
   z.a[1]=I(0);e.a[1]=I(1);Jet fe=integrand(z,e,p);
   error=cn*(powI(dz,2*gn+1)*de*absbound(fz.a[2*gn])+dz*powI(de,2*gn+1)*absbound(fe.a[2*gn]));
   if(!error.finite())valid=false;
  }catch(const std::exception&){valid=false;failed++;}
  // Allocate absolute integral error proportional to rectangle area; global area=24/5.
  // Convert full-array budget by multiplier 14400.
  I allowance=I(tol.c_str())*dz*de/I(69120);
  if(!valid||error.upper()>allowance.lower()){
    if(c.depth>=20)throw std::runtime_error("subdivision cap reached");
    // Balanced in scaled coordinates; take widths against nominal scales .25,.025.
    Cell d=c;c.depth++;d.depth=c.depth;
    if(dz.upper()/0.25>=de.upper()/0.025){I m=(c.zl+c.zu)/I(2);c.zu=m;d.zl=m;}else{I m=(c.el+c.eu)/I(2);c.eu=m;d.el=m;}
    todo.push_back(c);todo.push_back(d);split++;continue;
  }
  ORDER=0;I zm=(c.zl+c.zu)/I(2),em=(c.el+c.eu)/I(2),q;
  for(auto &az:nodes)for(auto &ae:nodes){Jet z(zm+dz*az.first/I(2)),e(em+de*ae.first/I(2));q=q+az.second*ae.second*integrand(z,e,p).a[0];evals++;}
  q=q*dz*de/I(4);total=total+q;errtotal=errtotal+upperpoint(error);good++;maxdepth=std::max(maxdepth,c.depth);
  if(good%100==0)std::cerr<<"N="<<N<<" cells="<<good<<" split="<<split<<" todo="<<todo.size()<<" seconds="<<std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count()<<"\n";
 }
 I globalerr=I(14400)*errtotal,base=I(14400)*total;
 I cert=hull(lowerpoint(base-globalerr),upperpoint(base+globalerr)+upperpoint(tail(p)));
 std::cout<<std::setprecision(17)<<"{\"N\":"<<N<<",\"method\":\"real MPFR interval jets / Gauss remainder\",\"mpfr\":\""<<mpfr_get_version()<<"\",\"precision_bits\":"<<PREC<<",\"gauss_order\":"<<gn<<",\"initial_nz\":"<<nz<<",\"initial_ne\":"<<ne<<",\"tolerance\":\""<<tol<<"\",\"conditional_KL\":"<<cert<<",\"quadrature_value_enclosure\":"<<base<<",\"quadrature_error_bound\":"<<globalerr<<",\"tail_upper\":"<<tail(p)<<",\"cells\":"<<good<<",\"splits\":"<<split<<",\"rejected_interval_boxes\":"<<failed<<",\"maxdepth\":"<<maxdepth<<",\"point_evaluations\":"<<evals<<",";
 I corr=correction(p);std::cout<<"\"learning_correction\":"<<corr<<",\"unconditional_upper\":"<<upperpoint(cert)+corr<<",\"seconds\":"<<std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count()<<"}\n";
 return 0;
 }catch(const std::exception&e){std::cerr<<"FAIL: "<<e.what()<<"\n";return 2;}
}
