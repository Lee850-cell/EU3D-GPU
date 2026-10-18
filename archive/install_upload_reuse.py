from pathlib import Path
import shutil

def once(s,a,b):
    if s.count(a)!=1: raise SystemExit('Unexpected source; no files changed: '+a[:100])
    return s.replace(a,b)
changes={}
p=Path('include/FlowfieldCuda.cuh');s=p.read_text()
if 'begin_muscl_upload_scope' in s: raise SystemExit('Already installed')
changes[p]=s+'\n// Scope is one uninterrupted x/y/z sweep over unchanged host fields.\nvoid begin_muscl_upload_scope(bool enabled);\nvoid end_muscl_upload_scope();\n'
p=Path('src/FlowfieldCuda.cu');s=p.read_text()
s=once(s,'struct MusclAusmBuffers\n{','''struct MusclAusmBuffers
{
    double* cached_fields = nullptr;
    bool upload_scope = false;
    bool uploaded = false;
    const double* host_identity[8] = {};''')
s=once(s,'        if (input) cudaFree(input);','''        if (cached_fields) cudaFree(cached_fields);
        cached_fields = nullptr;
        uploaded = false;
        if (input) cudaFree(input);''')
anchor='void muscl_ausm_host('
s=once(s,anchor,'''void begin_muscl_upload_scope(bool enabled)
{
    auto& b = get_muscl_ausm_buffers();
    b.upload_scope = enabled;
    b.uploaded = false;
}
void end_muscl_upload_scope()
{
    auto& b = get_muscl_ausm_buffers();
    b.upload_scope = false;
    b.uploaded = false;
}

'''+anchor)
anchor='    double* d_input = buffers.input;'
s=once(s,anchor,'''    const bool reuse_upload = buffers.upload_scope;
    const bool upload_now = !reuse_upload || !buffers.uploaded;
    if (reuse_upload && !buffers.cached_fields)
        check(cudaMalloc(&buffers.cached_fields, 7 * scalar_bytes));
    if (reuse_upload && buffers.uploaded) {
        bool same = buffers.host_identity[7] == h_yi;
        for (int n=0;n<7;++n) same = same && buffers.host_identity[n] == host_fields[n];
        if (!same) {
            std::fprintf(stderr,"MUSCL upload scope error: input pointers changed inside sweep\\n");
            std::exit(EXIT_FAILURE);
        }
    }

'''+anchor)
anchor='''        check(cudaMemcpy(
            d_input, host_fields[n], scalar_bytes,
            cudaMemcpyHostToDevice));'''
s=once(s,anchor,'''        double* field_input = reuse_upload ? buffers.cached_fields + count*n : d_input;
        if (upload_now) {
            check(cudaMemcpy(field_input, host_fields[n], scalar_bytes, cudaMemcpyHostToDevice));
        }''')
# Only change the reconstruction in this host function, not unrelated helpers.
start=s.index('void muscl_ausm_host(')
a=s[:start];b=s[start:]
b=once(b,'            d_input, d_lr[n],','            field_input, d_lr[n],')
b=once(b,'''    check(cudaMemcpy(
        d_yi, h_yi, species_bytes,
        cudaMemcpyHostToDevice));''','''    if (upload_now) {
        check(cudaMemcpy(d_yi, h_yi, species_bytes, cudaMemcpyHostToDevice));
    }
    if (reuse_upload) {
        for (int n=0;n<7;++n) buffers.host_identity[n] = host_fields[n];
        buffers.host_identity[7] = h_yi;
        buffers.uploaded = true;
        std::printf("MUSCL upload reuse: direction=%d, host_to_device_bytes=%zu\\n",
            direction, upload_now ? 7*scalar_bytes+species_bytes : size_t(0));
    }''')
changes[p]=a+b
p=Path('src/Flowfield.cpp');s=p.read_text()
a='\tAUSM(1, F, diff, keep_flux_on_gpu);'
s=once(s,a,'''    const char* reuse_flag = std::getenv("EU3D_REUSE_UPLOAD");
    const bool reuse_upload = use_cuda && !validate_gpu && diff == MUSCL_2
        && reuse_flag && std::strcmp(reuse_flag,"1")==0;
    begin_muscl_upload_scope(reuse_upload);
'''+a)
a='\tAUSM(3, Q, diff, keep_flux_on_gpu);'
s=once(s,a,a+'\n    end_muscl_upload_scope();')
changes[p]='#include <cstdlib>\n#include <cstring>\n'+s
for p in changes:
    if Path(str(p)+'.before_upload_reuse').exists(): raise SystemExit('Backup exists; no files changed')
for p,s in changes.items():
    shutil.copy2(p,str(p)+'.before_upload_reuse');p.write_text(s)
print('Installed. Opt in with EU3D_REUSE_UPLOAD=1. No math kernels or thermo code changed.')
