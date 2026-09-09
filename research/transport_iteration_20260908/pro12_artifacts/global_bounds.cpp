#define main certificate_program_main
#include "certify.cpp"
#undef main
// Real all-domain lemmas reduced to explicit constant inequalities.
// The polynomial proof of lambda*k <= 3/2 and sign cases are in MATH.md.
int main(){try{
 PREC=192; Plan p(4); I pi=piI(),ee=expI(I(1)),B("1.7"),L("0.6"),h("0.5");
 I psi1=erfI(p.invsqrt2);
 I Apos=I("0.9")*(inv(sqrtI(I(2)*pi*ee))+I("1.2")/(I(2)*pi));
 I Aneg=I("0.9")/(I("0.4")*sqrtI(I(2)*pi*ee));
 I Bnear=I("0.9")*I("1.2")/(I(2)*pi*powI(I(1)-I("0.6")*psi1,2));
 I Bfar=I("0.9")*I("1.2")/(I(2)*pi*ee*I("0.16"));
 I speed=I(3)/sqrtI(pi);
 I r=h*L+powI(h*L,2)/I(2),sumr=L+h*L*L/I(2);
 I logabs=sumr/(I(1)-r),ellconstant=B*B/I(2)-logI(I("0.4"))+I("1.1");
 for(auto a:{Apos,Aneg,Bnear,Bfar})if(!(a.upper()<L.lower()))throw std::runtime_error("global derivative bound failed");
 if(!(speed.upper()<B.lower()&&r.upper()<1&&logabs.upper()<1.1&&ellconstant.upper()<3.5))throw std::runtime_error("global tail lemma failed");
 for(int n:{4,6,8})for(auto a:gauss(n))if(!(a.first.lower()>-1&&a.first.upper()<1))throw std::runtime_error("node escaped (-1,1)");
 std::cout<<"{\"status\":\"pass\",\"precision_bits\":192,\"positive_q_derivative_bound\":"<<Apos<<",\"negative_q_first_term_bound\":"<<Aneg<<",\"negative_near_second_term_bound\":"<<Bnear<<",\"negative_far_second_term_bound\":"<<Bfar<<",\"field_upper\":"<<speed<<",\"step_perturbation_upper\":"<<r<<",\"logJ_absolute_upper\":"<<logabs<<",\"ell_constant_upper\":"<<ellconstant<<",\"tail_upper\":"<<tail(p)<<"}\n";
 }catch(const std::exception&e){std::cerr<<e.what()<<"\n";return 2;}return 0;}
