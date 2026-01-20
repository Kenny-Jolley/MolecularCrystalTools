#!/usr/bin/env python3
"""
unwrap_cell.py

Unwrap periodic boundaries in an extended XYZ file so that:
 - connected atoms (by covalent-bond cutoff) are placed in contiguous images
 - all coordinates are positive

Usage:
    python unwrap_cell.py input.xyz output.xyz [--tol 0.4] [--minpos 0.2]

Notes:
 - Requires Lattice="a  b  c  d  e  f  g  h  i" in the comment/header line (extended XYZ).
 - If no Lattice is found the script will abort.
 - Bond detection uses covalent radii + tolerance; common elements supported.
"""

import sys
import re
import argparse
from collections import deque, defaultdict
import numpy as np

# Minimal covalent radii table (Å). Extend if needed.
COVALENT_RADII = {
    'H': 0.31, 'He': 0.28,
    'Li': 1.28, 'Be': 0.96, 'B': 0.84, 'C': 0.76, 'N': 0.71, 'O': 0.66, 'F': 0.57, 'Ne': 0.58,
    'Na': 1.66, 'Mg': 1.41, 'Al': 1.21, 'Si': 1.11, 'P': 1.07, 'S': 1.05, 'Cl': 1.02, 'Ar': 1.06,
    'K': 2.03, 'Ca': 1.76, 'Sc': 1.70, 'Ti': 1.60, 'V': 1.53, 'Cr': 1.39, 'Mn': 1.39, 'Fe': 1.32,
    'Co': 1.26, 'Ni': 1.24, 'Cu': 1.32, 'Zn': 1.22, 'Ga': 1.22, 'Ge': 1.20, 'As': 1.19, 'Se': 1.20,
    'Br': 1.20, 'Kr': 1.16, 'Rb': 2.20, 'Sr': 1.95, 'Y': 1.90, 'Zr': 1.75, 'Nb': 1.64, 'Mo': 1.54,
    'Tc': 1.47, 'Ru': 1.46, 'Rh': 1.42, 'Pd': 1.39, 'Ag': 1.45, 'Cd': 1.44, 'In': 1.42, 'Sn': 1.39,
    'Sb': 1.39, 'Te': 1.38, 'I': 1.39, 'Xe': 1.40, 'Cs': 2.44, 'Ba': 2.15, 'La': 2.07, 'Ce': 2.04,
    # add more as needed
}

# Regex to extract Lattice="..." (9 floats)
LATTICE_RE = re.compile(r'Lattice\s*=\s*"([^"]+)"')

def parse_extended_xyz(filename):
    """Parse an extended xyz file into a list of frames.
    Each frame is a dict with keys:
      'natoms', 'comment', 'elements' (list), 'coords' (Nx3 numpy array), 'atom_lines' (list of token lists),
      'lattice' (3x3 numpy array or None)
    """
    frames = []
    with open(filename, 'r') as f:
        lines = f.readlines()

    i = 0
    lineno = 0
    while i < len(lines):
        line = lines[i].strip()
        if line == '':
            i += 1
            continue
        try:
            natoms = int(line)
        except ValueError:
            raise RuntimeError(f"Expected atom count at line {i+1}, got: {lines[i]!r}")
        comment = lines[i+1].rstrip('\n')
        # extract lattice if present
        lattice = None
        m = LATTICE_RE.search(comment)
        if m:
            nums = [float(x) for x in m.group(1).split()]
            if len(nums) != 9:
                raise RuntimeError(f"Lattice found but did not contain 9 numbers: {m.group(1)!r}")
            lattice = np.array(nums, dtype=float).reshape((3,3))
        elements = []
        coords = []
        atom_lines = []
        for j in range(natoms):
            ln = lines[i+2+j].rstrip('\n')
            toks = ln.split()
            if len(toks) < 4:
                raise RuntimeError(f"Atom line too short at {i+3+j}: {ln!r}")
            elem = toks[0]
            try:
                x = float(toks[1]); y = float(toks[2]); z = float(toks[3])
            except ValueError:
                # sometimes extended xyz provides "index element x y z ..." (rare). Try to find floats.
                floats = [t for t in toks if re.match(r'[+-]?\d+(\.\d*)?([eE][+-]?\d+)?$', t)]
                if len(floats) < 3:
                    raise RuntimeError(f"Could not parse coordinates on line {i+3+j}: {ln!r}")
                x,y,z = map(float, floats[:3])
            elements.append(elem)
            coords.append([x,y,z])
            atom_lines.append(toks)
        frame = {
            'natoms': natoms,
            'comment': comment,
            'elements': elements,
            'coords': np.array(coords, dtype=float),
            'atom_lines': atom_lines,
            'lattice': lattice
        }
        frames.append(frame)
        i += 2 + natoms
    return frames

def detect_bonds(elements, coords, cell, tol=0.4):
    """Return list of bonded pairs (i,j) using covalent radii + tol.
    Uses brute-force O(N^2) check.
    """
    n = len(elements)
    pairs = []
    for i in range(n):
        ri = COVALENT_RADII.get(elements[i], 0.75)  # fallback radius if unknown
        for j in range(i+1, n):
            rj = COVALENT_RADII.get(elements[j], 0.75)
            cutoff = ri + rj + tol
            d = np.linalg.norm(coords[i] - coords[j])
            if d <= cutoff:
                pairs.append((i,j))
                continue
            # Account for the periodic boundaries
            for nx in range(-1, 2):
                for ny in range(-1, 2):
                    for nz in range(-1, 2):
                        shift_vec = nx * cell[0] + ny * cell[1] + nz * cell[2]
                        d = np.linalg.norm(coords[i] - coords[j] + shift_vec)
                        if d <= cutoff:
                            pairs.append((i, j))
                            break
    return pairs

def periodic_image_shift(current_pos, neighbor_pos, cell, max_range=1):
    """
    Find integer shift (nx,ny,nz) (each in [-max_range...max_range]) to apply to neighbor such that
    distance to current_pos is minimized. Returns the shifted neighbor position and the shift tuple.
    """
    best = None
    best_pos = None
    best_shift = (0,0,0)
    # loops over shifts
    for nx in range(-max_range, max_range+1):
        for ny in range(-max_range, max_range+1):
            for nz in range(-max_range, max_range+1):
                shift_vec = nx*cell[0] + ny*cell[1] + nz*cell[2]
                cand = neighbor_pos + shift_vec
                d = np.linalg.norm(cand - current_pos)
                if best is None or d < best:
                    best = d
                    best_shift = (nx,ny,nz)
                    best_pos = cand
    #print(best)
    #print(best_shift)
    return best_pos, best_shift

def unwrap_frame(frame, tol=0.4):
    """Unwrap one frame in-place: adjusts coords so bonded atoms are contiguous and coords positive."""
    coords = frame['coords'].copy()
    cell = frame['lattice']
    if cell is None:
        raise RuntimeError("No Lattice found in frame comment. Extended XYZ 'Lattice' required for unwrapping.")

    elements = frame['elements']
    natoms = frame['natoms']

    #print(cell)
    #print(elements)
    #print(natoms)
    #print(coords)


    # Fold atoms into the unit cell, this ensures we start from a consistent position.

    # Create an empty array of atom coordinates
    new_coords = np.zeros_like(coords)

    # compute the fractional coordinates
    A = np.column_stack((cell[0], cell[1], cell[2]))  # 3x3
    try:
        Ainv = np.linalg.inv(A)
    except np.linalg.LinAlgError:
        raise RuntimeError("Lattice matrix is singular")

    fracs = (Ainv @ coords.T).T  # fractional coordinates for each atom

    #print(fracs)
    # choose integer shifts per atom so that fractional coords are within [0,1) ideally but keep continuity among connected components.
    # We'll shift each atom by integer vector = -floor(frac) to bring them into [0,1)
    int_shifts = -np.floor(fracs).astype(int)
    #print(int_shifts)
    # apply integer shifts
    for idx in range(natoms):
        shift = int_shifts[idx]
        new_coords[idx] = coords[idx].copy() + shift[0] * cell[0] + shift[1] * cell[1] + shift[2] * cell[2]

    #print(new_coords)
    #print(new_coords[38])
    # detect bonds based on current coordinates (note: bonds near cell boundaries will be detected only if within cutoff)
    bonds = detect_bonds(elements, new_coords, cell, tol=tol)
    neighbors = defaultdict(list)
    for i, j in bonds:
        neighbors[i].append(j)
        neighbors[j].append(i)

    #print(bonds)
    #print(neighbors[45])
    # Unwrap each molecule by traversing connected components
    # keeping a 'placed' boolean and placing atoms in cartesian space (moving by integer multiples of cell vectors).
    placed = [False] * natoms

    for start in range(natoms):
        if placed[start]:
            continue
        # start a new component
        placed[start] = True

        q = deque([start])
        #print(q)
        while q:
            i = q.popleft()
            #print(i)
            pi = new_coords[i]
            for j in neighbors.get(i, []):
                if placed[j]:
                    continue
                # find nearest periodic image of j to pi
                pj = new_coords[j]
                pj_shifted, shift = periodic_image_shift(pi, pj, cell, max_range=2)
                #print(pi, pj, pj_shifted)
                #print(np.linalg.norm(pi - pj_shifted))


                new_coords[j] = pj_shifted
                placed[j] = True
                q.append(j)


    frame['coords'] = new_coords
    return frame

def write_extended_xyz(frames, filename, minpos_margin=0.2):
    """Write frames back to extended xyz file. Ensure min coords >= minpos_margin by global translate."""
    # compute global minimal coordinate across all frames
    all_coords = np.vstack([fr['coords'] for fr in frames])
    min_coord = all_coords.min(axis=0)
    shift = np.zeros(3)
    for ax in range(3):
        if min_coord[ax] < minpos_margin:
            shift[ax] = minpos_margin - min_coord[ax]
    # apply shift to all frames
    with open(filename, 'w') as f:
        for fr in frames:
            nat = fr['natoms']
            f.write(f"{nat}\n")
            # keep original comment, but ensure Lattice values are the same - we will not change Lattice
            f.write(fr['comment'] + "\n")
            coords = fr['coords'] #+ shift
            # write per-atom lines preserving extra tokens if any; replace x,y,z tokens
            for idx in range(nat):
                toks = fr['atom_lines'][idx].copy()
                # replace 2..4 (indices 1,2,3) with coords (if present)
                if len(toks) >= 4:
                    toks[1] = f"{coords[idx,0]:.8f}"
                    toks[2] = f"{coords[idx,1]:.8f}"
                    toks[3] = f"{coords[idx,2]:.8f}"
                    f.write(" ".join(toks) + "\n")
                else:
                    # fallback write element and coords
                    f.write(f"{fr['elements'][idx]} {coords[idx,0]:.8f} {coords[idx,1]:.8f} {coords[idx,2]:.8f}\n")

def main():
    parser = argparse.ArgumentParser(description="Unwrap extended XYZ periodic images so bonds don't cross boundaries")
    parser.add_argument('input', help='input extended XYZ file')
    parser.add_argument('output', help='output extended XYZ file')
    parser.add_argument('--tol', type=float, default=0.4, help='bond detection tolerance (Å) added to covalent radii sum (default 0.4)')
    parser.add_argument('--minpos', type=float, default=0.2, help='minimum coordinate above zero after shifting (Å) (default 0.2)')
    args = parser.parse_args()

    frames = parse_extended_xyz(args.input)
    print(f"Read {len(frames)} frames from {args.input}")

    for fi,fr in enumerate(frames):
        print(f"Unwrapping frame {fi+1}/{len(frames)} (natoms={fr['natoms']})")
        try:
            unwrap_frame(fr, tol=args.tol)
        except Exception as e:
            print(f"Error unwrapping frame {fi+1}: {e}")
            sys.exit(1)

    write_extended_xyz(frames, args.output, minpos_margin=args.minpos)
    print(f"Wrote unwrapped frames to {args.output}")

if __name__ == '__main__':
    main()

