#include "FlowfieldCuda.cuh"
#include <cuda_runtime.h>
#include <vector>
#include <array>
#include <cstdio>
#include <cstring>
#include <cmath>
#include <cstdlib>
int main(){
    int devices=0;
    if(cudaGetDeviceCount(&devices)!=cudaSuccess || devices<1){std::fprintf(stderr,"GPU unavailable\n");return 77;}
    int sweeps=0;
    for(int shape=0;shape<3;++shape){
        int nx=shape==1?13:11, ny=9, nz=8, ns=3, bc=2;
        size_t n=size_t(nx)*ny*nz, nf=n*(ns+4);
        std::array<std::vector<double>,7> fields;
        for(auto& f:fields) f.resize(n);
        std::vector<double> yi(n*ns);
        auto run=[&](int dir,std::vector<double>& flux){
            muscl_ausm_host(fields[0].data(),fields[1].data(),fields[2].data(),fields[3].data(),fields[4].data(),fields[5].data(),fields[6].data(),yi.data(),flux.data(),nx,ny,nz,ns,bc,dir,true);
        };
        for(int step=0;step<3;++step){
            for(size_t i=0;i<n;++i){
                double f=double((i*17+step*11)%101)/101;
                fields[0][i]=100000+100*f; fields[1][i]=1+.1*f;
                fields[2][i]=3*f;fields[3][i]=2*f;fields[4][i]=f;
                fields[5][i]=300000+100*f;fields[6][i]=1.4;
                yi[i*ns]=.2+.01*f;yi[i*ns+1]=.3;yi[i*ns+2]=.5-.01*f;
            }
            std::array<std::vector<double>,3> expected;
            begin_muscl_upload_scope(false);
            for(int d=1;d<=3;++d){expected[d-1].assign(nf,7.0);run(d,expected[d-1]);}
            end_muscl_upload_scope();
            begin_muscl_upload_scope(true);
            for(int d=1;d<=3;++d){
                std::vector<double> actual(nf,7.0);run(d,actual);
                for(size_t j=0;j<nf;++j) if(!std::isfinite(actual[j]) || !std::isfinite(expected[d-1][j])) return 2;
                if(std::memcmp(actual.data(),expected[d-1].data(),nf*sizeof(double))){
                    std::fprintf(stderr,"Failed shape=%d step=%d direction=%d\n",shape,step,d);return 1;
                }
            }
            end_muscl_upload_scope();++sweeps;
            // Standalone call after scope must freshly upload changed data.
            fields[0][n/2]+=300;
            std::vector<double> fresh(nf,7.0),scoped(nf,7.0);
            run(2,fresh);
            begin_muscl_upload_scope(true);run(2,scoped);end_muscl_upload_scope();
            if(std::memcmp(fresh.data(),scoped.data(),nf*sizeof(double))) return 3;
        }
    }
    std::printf("Passed: %d sweeps, all directions, changed input at same addresses, resize, standalone calls; bitwise flux comparison.\n",sweeps);
}
