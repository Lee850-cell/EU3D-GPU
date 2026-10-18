#pragma once
#include <cstddef>
#include <cstring>
// A fixed-size memcpy lets the compiler inline short rows. The source and
// destination are separate allocations; halos are deliberately left untouched.
template<size_t Elements, class T>
inline void copy_packed_fixed_rows(T* destination, const T* source,
                                  int ni, int nj, int nk, int bc, int width)
{
    const size_t ny=static_cast<size_t>(nj)+2*bc;
    const size_t row_stride=(static_cast<size_t>(nk)+2*bc)*width;
    for(int i=0;i<ni;++i){
        T* row=destination+((static_cast<size_t>(i+bc)*ny+bc)*row_stride)+bc*width;
        for(int j=0;j<nj;++j){
            std::memcpy(row,source,Elements*sizeof(T));
            source+=Elements;
            row+=row_stride;
        }
    }
}
// Source contains only internal cells, with components contiguous per cell.
// Destination has bc ghost cells on both ends of all three dimensions.
template<class T>
inline void copy_packed_interior(T* destination, const T* source,
                                int ni, int nj, int nk, int bc, int width,
                                bool fast_rows=false)
{
    const size_t ny=static_cast<size_t>(nj)+2*bc;
    const size_t nz=static_cast<size_t>(nk)+2*bc;
    const size_t row_elements=static_cast<size_t>(nk)*width;
    if(fast_rows){
        // Common thin-grid scalar/species/conserved rows (NS=9). Other
        // shapes still use the general implementation below.
        switch(row_elements){
        case 2: return copy_packed_fixed_rows<2>(destination,source,ni,nj,nk,bc,width);
        case 18: return copy_packed_fixed_rows<18>(destination,source,ni,nj,nk,bc,width);
        case 26: return copy_packed_fixed_rows<26>(destination,source,ni,nj,nk,bc,width);
        }
    }
    for(int i=0;i<ni;++i) for(int j=0;j<nj;++j){
        const size_t input=(static_cast<size_t>(i)*nj+j)*row_elements;
        const size_t output=((static_cast<size_t>(i+bc)*ny+(j+bc))*nz+bc)*width;
        std::memcpy(destination+output,source+input,row_elements*sizeof(T));
    }
}
