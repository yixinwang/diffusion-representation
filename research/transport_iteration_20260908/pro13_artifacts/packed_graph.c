/* Exact threshold graph from packed observed signs. Single-thread C kernel.
 * No data access, RNG, known matching, or context/teacher inputs.
 * Inputs are d columns, each nw uint64 words; unused final bits must be zero.
 */
#include <stdint.h>
#include <stddef.h>
#include <limits.h>

int pro13_graph(const uint64_t *x, size_t d, size_t nw, size_t n,
                int64_t tau_num, int64_t tau_den,
                int32_t *degree, int32_t *partner, int64_t *edge_count) {
    if (!x || !degree || !partner || !edge_count || d<2 || d>INT32_MAX ||
        n==0 || nw!=(n+63)/64 || n>INT64_MAX/2000 ||
        tau_num<0 || tau_num>tau_den || tau_den>1000 || tau_den<=0) return -1;
    for(size_t i=0;i<d;++i) { degree[i]=0; partner[i]=-1; }
    *edge_count=0;
    for(size_t i=0;i<d;++i) {
        const uint64_t *a=x+i*nw;
        for(size_t j=i+1;j<d;++j) {
            const uint64_t *b=x+j*nw;
            int64_t h=0;
            for(size_t k=0;k<nw;++k)
                h += __builtin_popcountll(a[k]^b[k]);
            int64_t s=(int64_t)n-2*h;
            if(s<0) s=-s;
            if(s*tau_den>=tau_num*(int64_t)n) {
                degree[i]++; degree[j]++; partner[i]=(int32_t)j; partner[j]=(int32_t)i;
                (*edge_count)++;
            }
        }
    }
    return 0;
}
