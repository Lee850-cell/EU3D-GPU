#!/usr/bin/env python3
"""Compare every distributed interior-node copy against a single-rank GPU reference."""
import argparse
from array import array
import math
from pathlib import Path
import sys


def read(path):
    with Path(path).open('rb') as f:
        if f.readline() != b'EU3D_STATE_V1_LE\n':
            raise ValueError(f'{path}: wrong dump format')
        tokens = f.readline().decode().split()
        if len(tokens) != 19:
            raise ValueError(f'{path}: wrong metadata length')
        meta = list(map(int, tokens[:17]))
        times = tuple(map(float, tokens[17:]))
        fields = f.readline().decode().split()
        ns = meta[0]
        expected = 'X Y Z D T U V W E Wav Rgas Cp P H Gamma C Ma'.split()
        expected += [f'{name}{s}' for name in ('Di', 'Yi', 'Mc') for s in range(ns)]
        expected += [f'CS{s}' for s in range(ns+4)] + [f'CMS{s}' for s in range(ns)] + ['Nchem']
        if ns < 1 or fields != expected:
            raise ValueError(f'{path}: wrong field schema')
        g, n, c, b = meta[1:4], meta[4:7], meta[7:10], meta[10:13]
        if any(x < 2 for x in g+n) or any(x < 1 for x in b):
            raise ValueError(f'{path}: invalid grid dimensions')
        if any(not 0 <= c[d] < b[d] or (n[d]-1)*b[d]+1 != g[d] for d in range(3)):
            raise ValueError(f'{path}: inconsistent partition')
        if meta[13] != c[0]+b[0]*(c[1]+b[1]*c[2]) or meta[14] != math.prod(b):
            raise ValueError(f'{path}: inconsistent rank metadata')
        if meta[15] < 1 or any(not math.isfinite(t) or t <= 0 for t in times):
            raise ValueError(f'{path}: invalid iteration/time metadata')
        data = array('d')
        data.frombytes(f.read())
        if sys.byteorder != 'little':
            data.byteswap()
        if len(data) != math.prod(n)*len(fields):
            raise ValueError(f'{path}: truncated or oversized dump')
    return meta, times, fields, data


def compare(reference, parts, atol=1e-12, rtol=1e-10):
    rm, rt, names, ref = read(reference)
    if rm[14] != 1:
        raise ValueError('Reference must contain the entire grid from one rank')
    if any(not math.isfinite(v) for v in ref):
        raise ValueError('Reference contains nonfinite data')
    width = len(names)
    total = math.prod(rm[1:4])
    seen = bytearray(total)
    maxima = [0.0]*width
    counts = [0]*width
    worst = [0.0]*width
    ranks = set()
    layout = None
    examples = 0
    for path in parts:
        m, t, fields, values = read(path)
        if (m[:4], m[15:17], t, fields) != (rm[:4], rm[15:17], rt, names):
            raise ValueError(f'{path}: different grid/species/iteration/output/time/fields')
        if layout is None:
            layout = m[10:13]
        if m[10:13] != layout or m[13] in ranks:
            raise ValueError(f'{path}: mixed partitions or repeated rank')
        ranks.add(m[13])
        ni, nj, nk = m[4:7]
        ox, oy, oz = (m[7+d]*(m[4+d]-1) for d in range(3))
        for i in range(ni):
            for j in range(nj):
                for k in range(nk):
                    global_idx = ((ox+i)*rm[2]+oy+j)*rm[3]+oz+k
                    local_idx = (i*nj+j)*nk+k
                    seen[global_idx] += 1
                    for f in range(width):
                        a, b = ref[global_idx*width+f], values[local_idx*width+f]
                        err = abs(a-b) if math.isfinite(b) else math.inf
                        tolerance = 0.0 if names[f] == 'Nchem' else atol+rtol*abs(a)
                        maxima[f] = max(maxima[f], err)
                        if tolerance > 0:
                            worst[f] = max(worst[f], err/tolerance)
                        if err > tolerance:
                            counts[f] += 1
                            if examples < 12:
                                print(f'Mismatch rank={m[13]} global=({ox+i},{oy+j},{oz+k}) '
                                      f'field={names[f]} ref={a:.17g} actual={b:.17g} '
                                      f'error={err:.6e} tolerance={tolerance:.6e}')
                                examples += 1
    if layout is None or ranks != set(range(math.prod(layout))) or 0 in seen:
        raise ValueError('Missing ranks or incomplete global-grid coverage')
    print(f'Iterations={rm[15]}, global_grid={tuple(rm[1:4])}, unique_nodes={total}, '
          f'compared_node_copies={sum(seen)}, shared_nodes={sum(v>1 for v in seen)}')
    print(f'Tolerance: abs(error) <= {atol} + {rtol} * abs(reference); Nchem exact')
    for name, error, count, scaled in zip(names, maxima, counts, worst):
        print(f'{name}: max_abs={error:.6e}, max_error_over_tolerance={scaled:.3g}, mismatches={count}')
    mismatches = sum(counts)
    print(f'Multi-GPU state validation: {"Passed" if not mismatches else "Failed"}, mismatches={mismatches}')
    return mismatches


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('reference')
    p.add_argument('parts', nargs='+')
    args = p.parse_args()
    try:
        bad = compare(args.reference, args.parts)
    except (OSError, ValueError, OverflowError) as e:
        p.exit(2, f'Validation input error: {e}\n')
    sys.exit(1 if bad else 0)


if __name__ == '__main__':
    main()
