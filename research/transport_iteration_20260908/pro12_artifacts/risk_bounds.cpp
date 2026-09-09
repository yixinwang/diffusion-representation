#define main certificate_program_main
#include "certify.cpp"
#undef main
// Directed postprocessing of a certified interval [a,b].
// The primary source of a,b is the corresponding completed certificate JSON.
int main(int argc,char**argv){try{
 if(argc!=4)throw std::runtime_error("usage: risk_bounds N conditional_lower conditional_upper");
 PREC=192;int n=std::stoi(argv[1]);Plan p(n);I a(argv[2]),b(argv[3]);
 if(!a.positive()||b.lower()<a.upper())throw std::runtime_error("invalid supplied interval");
 I pg=expI(-I(32)*I(192)/I(72));
 I rho=I(1)-I(9)/I(16)*(I(1)-expI(-I(2880)*I("0.1")*I("0.1")*I("0.5")*I("0.5")/(I(24)*I("1.6"))));
 I pj=I(15)*powI(rho,256);
 I B("1.7"),c=B/I(2);
 I M=B*(c*(I(1)+erfI(c*p.invsqrt2))/I(2)+expI(-c*c/I(2))*p.invsqrt2pi)+I("0.12")+I("1.1");
 I W=I(96)+I("1.5")*b+I("1.5")*sqrtI(I(1440)*b);
 I penalty=pj*W+pg*(I(2880)*M+I(128));
 I lo=(I(1)-pj-pg)*a,hi=b+penalty;
 I risk=hull(lowerpoint(lo),upperpoint(hi));
 std::cout<<"{\"N\":"<<n<<",\"root_correct_wrong_head_bound\":"<<W<<",\"learning_correction\":"<<penalty<<",\"unconditional_KL\":"<<risk<<",\"unconditional_width_upper\":"<<upperpoint(hi)-lowerpoint(lo)<<",\"exact_expected_upper\":"<<I(96)*pj+I(224)*pg<<",\"precision_bits\":192,\"scope\":\"expected training risk for each fixed truth; not KL of mixture over fitted generators\"}\n";
 }catch(const std::exception&e){std::cerr<<e.what()<<"\n";return 2;}return 0;}
