#define main certificate_program_main
#include "certify.cpp"
#undef main
static int tests=0;
void covers(const I&b,const I&x,const char*msg){
    // For nonexact transcendental expected values, overlap is an independent
    // algebraic consistency check, not a proof of enclosure correctness.
    if(mpfr_cmp(b.u.x,x.l.x)<0||mpfr_cmp(x.u.x,b.l.x)<0)throw std::runtime_error(msg);tests++;
}
std::pair<Jet,Jet> state(const I&z0,const I&e0,const Plan&p){
    Jet z(z0);z.a[1]=I(1);Jet e(e0),y=z,J(1);
    for(int i=0;i<p.N/2;i++){
        Jet v,dy;if(i==0)v=e*p.invsqrtpi;else{auto a=field(y,e,p,i);v=a.first;dy=a.second;}
        Jet fac;if(i+1==p.N/2){fac=Jet(1)+dy*(p.h/I(2));y=y+v*(p.h/I(2));}
        else{auto b=field(y+v*p.h,e,p,i+1);fac=Jet(1)+(dy+b.second*(Jet(1)+dy*p.h))*(p.h/I(2));y=y+(v+b.first)*(p.h/I(2));}
        J=J*fac;
    }return{y,J};
}
int main(){try{
 PREC=192;ORDER=12;
 for(int n:{4,6,8}){auto g=gauss(n);for(int k=0;k<2*n;k++){I v;for(auto&w:g)v=v+w.second*powI(w.first,k);covers(v,k%2?I(0):I(2)/I(k+1),"Gauss polynomial moment");}}
 Jet x;x.a[1]=I(1);Jet ex=expJ(x),lg=logJ(Jet(1)+x),ef=erfJ(x),ff=FJ(x),rc=Jet(1)/(Jet(1)-x);
 for(int k=0;k<=ORDER;k++){
  covers(ex.a[k],inv(factorial(k)),"exp Taylor");
  covers(rc.a[k],I(1),"reciprocal Taylor");
  covers(lg.a[k],k?I(k%2?1:-1)/I(k):I(0),"log Taylor");
  I expected;if(k%2){int m=(k-1)/2;expected=I(2)*I(m%2?-1:1)/(sqrtI(piI())*factorial(m)*I(k));}
  covers(ef.a[k],expected,"erf Taylor");covers(ff.a[k],k<2?I(0):I(k-1)/factorial(k),"F cancellation Taylor");
 }
 // Gaussian error bound tested on exp(x) over [-1,1].
 for(int n:{4,6,8}){auto g=gauss(n);I q;for(auto&v:g)q=q+v.second*expI(v.first);
  I cn=powI(factorial(n),4)/(I(2*n+1)*powI(factorial(2*n),2));I err=cn*powI(I(2),2*n+1)*expI(I(1))/factorial(2*n);
  I cert=hull(q-err,q+err);covers(cert,expI(I(1))-expI(I(-1)),"Gauss exponential remainder");
 }
 // Explicit product Jacobian versus independently propagated dual derivative.
 ORDER=1;
 for(int n:{4,8,16,32,64}){Plan p(n);for(int z:{-12,-3,0,3,12})for(auto e:{"0.4","0.5","0.6"}){
  auto a=state(I(z),I(e),p);covers(a.first.a[1],a.second.a[0],"discrete Jacobian dual parity");
  if(!a.second.a[0].positive())throw std::runtime_error("positive J point test");tests++;
 }}
 // Fail closed for illegal real domains.
 bool bad=false;try{inv(hull(I(-1),I(1)));}catch(...){bad=true;}if(!bad)throw std::runtime_error("division must reject");tests++;
 bad=false;try{logI(hull(I(-1),I(1)));}catch(...){bad=true;}if(!bad)throw std::runtime_error("log must reject");tests++;
 std::cout<<"{\"status\":\"pass\",\"checks\":"<<tests<<",\"precision_bits\":"<<PREC<<",\"mpfr\":\""<<mpfr_get_version()<<"\",\"scope\":\"algebraic and independent derivative consistency tests; not a proof assistant\"}\n";
 }catch(const std::exception&e){std::cerr<<"FAIL: "<<e.what()<<"\n";return 2;}return 0;}
