#include "PackedReactionCopy.hpp"
#include <vector>
#include <cstdio>
#include <cstdlib>

template<class T> void check(){
    for(int ni:{1,3}) for(int nj:{1,5}) for(int nk:{1,2,3,16})
    for(int bc:{0,1,2}) for(int width:{1,2,9,13,18,26}){
        size_t length=size_t(ni+2*bc)*(nj+2*bc)*(nk+2*bc)*width;
        std::vector<T> source(size_t(ni)*nj*nk*width);
        for(size_t n=0;n<source.size();++n) source[n]=T(n+1);
        std::vector<T> actual(length,T(-7)),expected=actual;
        for(int i=0;i<ni;++i) for(int j=0;j<nj;++j)
        for(int k=0;k<nk;++k) for(int s=0;s<width;++s)
            expected[((size_t(i+bc)*(nj+2*bc)+j+bc)*(nk+2*bc)+k+bc)*width+s]=
                source[((size_t(i)*nj+j)*nk+k)*width+s];
        copy_packed_interior(actual.data(),source.data(),ni,nj,nk,bc,width,true);
        if(std::memcmp(actual.data(),expected.data(),length*sizeof(T))){
            std::fprintf(stderr,"Packed copy failed: %d %d %d %d %d\n",ni,nj,nk,bc,width);
            std::exit(1);
        }
    }
}
int main(){check<double>();check<int>();std::puts("Packed copy: Passed, specialized and fallback rows, halos checked");}
