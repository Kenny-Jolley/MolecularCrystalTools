#!/usr/bin/env python3
"""
overlay_molecule.py

Takes an input unit cell (intended as a known crystal structure) and a predicted unit cell
(intended as an output of a crystal structure prediction) and overlays the predicted cell on top of the known one.
The script will translate and rotate the predicted cell to find the best fit

Usage:
    python overlay_molecule.py exp_cell.xyz predicted_cell.xyz

Notes:
 - Requires input structures in extended xyz format.
"""

import re
import argparse
from collections import deque, defaultdict
import numpy as np
from scipy.optimize import minimize

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
            #print(coords)
            # write per-atom lines preserving extra tokens if any; replace x,y,z tokens
            for idx in range(nat):
                toks = fr['atom_lines'][idx].copy()
                print(toks)
                # replace 2..4 (indices 1,2,3) with coords (if present)
                if len(toks) >= 4:
                    print(idx)
                    toks[1] = f"{coords[idx,0]:.8f}"
                    toks[2] = f"{coords[idx,1]:.8f}"
                    toks[3] = f"{coords[idx,2]:.8f}"
                    f.write(" ".join(toks) + "\n")
                else:
                    # fallback write element and coords
                    f.write(f"{fr['elements'][idx]} {coords[idx,0]:.8f} {coords[idx,1]:.8f} {coords[idx,2]:.8f}\n")

def write_extended_xyz_new_data_only(frames, filename):
    """Write frames back to extended xyz file. """
    with open(filename, 'w') as f:
        for fr in frames:
            nat = fr['natoms']
            f.write(f"{nat}\n")
            # keep original comment, but ensure Lattice values are the same - we will not change Lattice
            f.write(f'Lattice="{fr['lattice'][0][0]} {fr['lattice'][0][1]} {fr['lattice'][0][2]} '
                    f'{fr['lattice'][1][0]} {fr['lattice'][1][1]} {fr['lattice'][1][2]} '
                    f'{fr['lattice'][2][0]} {fr['lattice'][2][1]} {fr['lattice'][2][2]} " Properties=species:S:1:pos:R:3' + "\n")

            coords = fr['coords']

            # write per-atom lines preserving extra tokens if any; replace x,y,z tokens
            for idx in range(nat):
                f.write(f"{fr['elements'][idx]} {coords[idx,0]:.8f} {coords[idx,1]:.8f} {coords[idx,2]:.8f}\n")


def replicate_frame(frame, nx, ny, nz):
    """Replicate one frame in-place: duplicate cell in positive direction only"""
    print('Periodic replication of structure called...')
    coords = frame['coords'].copy()
    cell = frame['lattice']
    if cell is None:
        raise RuntimeError("No Lattice found in frame comment. Extended XYZ 'Lattice' required for replication.")
    try:
        nx = int(nx)
        ny = int(ny)
        nz = int(nz)
        print(f"Periodic replication is {nx}x{ny}x{nz}")
    except:
        raise RuntimeError("nx, ny, nz must be integers")

    # update number of atoms
    tot_rep = nx*ny*nz
    print(f'Old number of atoms is {frame['natoms']}')
    frame['natoms'] = tot_rep*frame['natoms']
    print(f'New number of atoms is {frame['natoms']}')

    # element list should be replicated
    elements = frame['elements'].copy()
    #print(elements)
    frame['elements'] = frame['elements']*tot_rep
    #print(frame['elements'] )

    frame['atom_lines'] = frame['atom_lines'] * tot_rep

    rep_coords = []
    # coordinates should be replicated periodically
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                #print(i,j,k)
                for c in coords:
                    #print(c)
                    rep_coords.append(list(c + i * cell[0] + j * cell[1] + k * cell[2]))
    #print(rep_coords)
    frame['coords'] = np.array(rep_coords, dtype=float)

    # cell should also be expanded now
    frame['lattice'][0] = frame['lattice'][0]* nx
    frame['lattice'][1] = frame['lattice'][1] * ny
    frame['lattice'][2] = frame['lattice'][2] * nz

    #print(cell)
    #print(frame['natoms'])
    #print(elements)
    #print(frame['atom_lines'])

    #print(coords)




    return frame


def make_overlayed_frame(exp_frame, predicted_frame, set_element_colour=False, rot_steps = 3):
    """Overlays the predicted unit cell on top of the experimental cell.  Translates and rotates predicted cell to find the best fit"""

    # total atoms is the sum
    natoms = exp_frame['natoms'] + predicted_frame['natoms']
    # Use comment from the experimental, this will be the larger system
    comment = exp_frame['comment']

    # error func
    def error_func(exp_coords, predicted_coords):
        error = 0
        for i, c in enumerate(predicted_coords):
            #print(i,c)
            best = None
            for j, d in enumerate(exp_coords):
                #print(j, exp_frame['elements'][j], d, predicted_frame['elements'][i])
                if exp_frame['elements'][j] == predicted_frame['elements'][i]:
                    n = np.linalg.norm(c - d)
                    if best is None or n < best:
                        best = n
            error += best

        return error

    print(error_func(exp_frame['coords'], predicted_frame['coords']))




    def opt_fun(x):
        predicted_coords = np.zeros_like(predicted_frame['coords'])

        # compute average coords
        avx = 0
        avy = 0
        avz = 0
        for i in range(len(predicted_coords)):
            avx += predicted_frame['coords'][i][0]
            avy += predicted_frame['coords'][i][1]
            avz += predicted_frame['coords'][i][2]
        avx /= len(predicted_coords)
        avy /= len(predicted_coords)
        avz /= len(predicted_coords)
        # Translates all atoms to origin av
        for i in range(len(predicted_coords)):
            predicted_coords[i] = predicted_frame['coords'][i] + [-avx, 0, 0] + [0, -avy, 0] + [0, 0, -avz]

        # rotate all atoms about 100 around origin
        k = np.array([1,0,0], dtype=float)
        #k = k / np.linalg.norm(k)  # normalize axis

        # Rodrigues' rotation formula:
        # p_rot = p*cosθ + (k × p)*sinθ + k*(k·p)*(1 - cosθ)
        cos_t = np.cos(x[3])
        sin_t = np.sin(x[3])
        for i in range(len(predicted_coords)):
            p = predicted_coords[i]
            term1 = p * cos_t
            term2 = np.cross(k, p) * sin_t
            term3 = k * np.dot(k, p) * (1 - cos_t)
            predicted_coords[i] = term1 + term2 + term3

        # rotate all atoms about 010 around origin
        k = np.array([0, 1, 0], dtype=float)
        # k = k / np.linalg.norm(k)  # normalize axis

        # Rodrigues' rotation formula:
        # p_rot = p*cosθ + (k × p)*sinθ + k*(k·p)*(1 - cosθ)
        cos_t = np.cos(x[4])
        sin_t = np.sin(x[4])
        for i in range(len(predicted_coords)):
            p = predicted_coords[i]
            term1 = p * cos_t
            term2 = np.cross(k, p) * sin_t
            term3 = k * np.dot(k, p) * (1 - cos_t)
            predicted_coords[i] = term1 + term2 + term3

        # rotate all atoms about 010 around origin
        k = np.array([0, 0, 1], dtype=float)
        # k = k / np.linalg.norm(k)  # normalize axis

        # Rodrigues' rotation formula:
        # p_rot = p*cosθ + (k × p)*sinθ + k*(k·p)*(1 - cosθ)
        cos_t = np.cos(x[5])
        sin_t = np.sin(x[5])
        for i in range(len(predicted_coords)):
            p = predicted_coords[i]
            term1 = p * cos_t
            term2 = np.cross(k, p) * sin_t
            term3 = k * np.dot(k, p) * (1 - cos_t)
            predicted_coords[i] = term1 + term2 + term3


                # Translates all atoms back
        for i in range(len(predicted_coords)):
            predicted_coords[i] = predicted_coords[i] + [avx, 0, 0] + [0, avy, 0] + [0, 0, avz]


        # Translates all atoms in cell
        for i in range(len(predicted_coords)):
            predicted_coords[i] = predicted_coords[i] + [x[0],0,0] + [0, x[1],0] + [0, 0, x[2]]



        return error_func(exp_frame['coords'], predicted_coords)


    # some initial displacement
    init_pos = np.array([0,0,0])  # position in cell to translate overlaid structure before optimisation
    # rotation steps within 2pi rads
    n = rot_steps
    best = None
    best_res = None
    for i in range(1,n+1):
        for j in range(1, n+1):
            for k in range(1, n+1):
                x0 = np.concatenate( (init_pos, [i * 2 * np.pi / n, j * 2 * np.pi / n, k * 2 * np.pi / n]), axis=0)
                # x0 = np.array([10, 10, 0, i*2*np.pi/n, j*2*np.pi/n, k*2*np.pi/n])
                res = minimize(opt_fun, x0, method='powell', options={'disp': True})
                print(i,j,k)
                print(res.x)
                print(opt_fun(res.x))
                if best is None or opt_fun(res.x) < best:
                    best = opt_fun(res.x)
                    best_res = res.x

    res.x = best_res

    # Apply rotation and translation
    predicted_coords = np.zeros_like(predicted_frame['coords'])
    # compute average coords
    avx = 0
    avy = 0
    avz = 0
    for i in range(len(predicted_coords)):
        avx += predicted_frame['coords'][i][0]
        avy += predicted_frame['coords'][i][1]
        avz += predicted_frame['coords'][i][2]
    avx /= len(predicted_coords)
    avy /= len(predicted_coords)
    avz /= len(predicted_coords)
    # Translates all atoms to origin av
    for i in range(len(predicted_coords)):
        predicted_coords[i] = predicted_frame['coords'][i] + [-avx, 0, 0] + [0, -avy, 0] + [0, 0, -avz]

    # rotate all atoms about 100 around origin
    k = np.array([1, 0, 0], dtype=float)
    # k = k / np.linalg.norm(k)  # normalize axis

    # Rodrigues' rotation formula:
    # p_rot = p*cosθ + (k × p)*sinθ + k*(k·p)*(1 - cosθ)
    cos_t = np.cos(res.x[3])
    sin_t = np.sin(res.x[3])
    for i in range(len(predicted_coords)):
        p = predicted_coords[i]
        term1 = p * cos_t
        term2 = np.cross(k, p) * sin_t
        term3 = k * np.dot(k, p) * (1 - cos_t)
        predicted_coords[i] = term1 + term2 + term3

    # rotate all atoms about 010 around origin
    k = np.array([0, 1, 0], dtype=float)
    # k = k / np.linalg.norm(k)  # normalize axis

    # Rodrigues' rotation formula:
    # p_rot = p*cosθ + (k × p)*sinθ + k*(k·p)*(1 - cosθ)
    cos_t = np.cos(res.x[4])
    sin_t = np.sin(res.x[4])
    for i in range(len(predicted_coords)):
        p = predicted_coords[i]
        term1 = p * cos_t
        term2 = np.cross(k, p) * sin_t
        term3 = k * np.dot(k, p) * (1 - cos_t)
        predicted_coords[i] = term1 + term2 + term3

    # rotate all atoms about 010 around origin
    k = np.array([0, 0, 1], dtype=float)
    # k = k / np.linalg.norm(k)  # normalize axis

    # Rodrigues' rotation formula:
    # p_rot = p*cosθ + (k × p)*sinθ + k*(k·p)*(1 - cosθ)
    cos_t = np.cos(res.x[5])
    sin_t = np.sin(res.x[5])
    for i in range(len(predicted_coords)):
        p = predicted_coords[i]
        term1 = p * cos_t
        term2 = np.cross(k, p) * sin_t
        term3 = k * np.dot(k, p) * (1 - cos_t)
        predicted_coords[i] = term1 + term2 + term3

    # Translates all atoms back
    for i in range(len(predicted_coords)):
        predicted_coords[i] = predicted_coords[i] + [avx, 0, 0] + [0, avy, 0] + [0, 0, avz]

    # Translates all atoms in cell
    for i in range(len(predicted_coords)):
        predicted_coords[i] = predicted_coords[i] + [res.x[0], 0, 0] + [0, res.x[1], 0] + [0, 0, res.x[2]]





    # Elements, we can either keep names as they are, or set to predicted ones to element 3 so they are easy to see in ovito
    predicted_elements = predicted_frame['elements']
    if set_element_colour:
        for i in range(len(predicted_elements)):
            predicted_elements[i] = 'Li'
    # Sum element lists together
    elements = exp_frame['elements'] + predicted_elements


    # coords lists are summed too
    # print(type(exp_frame['coords']))
    #coords = np.concatenate((exp_frame['coords'], predicted_frame['coords']))
    coords = np.concatenate((exp_frame['coords'], predicted_coords))

    lattice = exp_frame['lattice']

    frame = {
        'natoms': natoms,
        'comment': comment,
        'elements': elements,
        'coords': np.array(coords, dtype=float),
        'lattice': lattice
    }
    return frame

def main():
    parser = argparse.ArgumentParser(description="""Takes an input unit cell (intended as a known crystal structure) and a predicted unit cell 
(intended as an output of a crystal structure prediction) and overlays the predicted cell on top of the known one.
The script will translate and rotate the predicted cell to find the best fit""")


    parser.add_argument('input_exp', help='input extended XYZ file (experimental structure)')
    parser.add_argument('input_predicted', help='input extended XYZ file (predicted structure)')
    parser.add_argument('--rot_steps', type=int, default=3, help='number of angles within 360 to try')
    parser.add_argument('--set_element_colour', type=bool, default=True, help='set overlaid atoms to Li so they are easily visible in ovito')
    #parser.add_argument('--tol', type=float, default=0.4, help='bond detection tolerance (Å) added to covalent radii sum (default 0.4)')
    #parser.add_argument('--minpos', type=float, default=0.2, help='minimum coordinate above zero after shifting (Å) (default 0.2)')
    args = parser.parse_args()

    exp_frames = parse_extended_xyz(args.input_exp)
    print(f"Read {len(exp_frames)} frame(s) from {args.input_exp}")
    # The exp structure should only be one frame, but we take the last frame anyway to be sure
    print(f"Experimental unit cell has {exp_frames[-1]['natoms']} atoms")

    predicted_frames = parse_extended_xyz(args.input_predicted)
    print(f"Read {len(predicted_frames)} frame(s) from {args.input_predicted}")
    # The predicted structure should only be one frame, but we take the last frame anyway to be sure
    print(f"Predicted unit cell has {predicted_frames[-1]['natoms']} atoms")

    # Replicate the experimental structure to 2x2x2 to make sure that a fit can be found if the predicted molecules follow a different arrangement
    exp_frame_rep = [replicate_frame(exp_frames[-1], 1, 1, 1)]

    #print(exp_frame_rep[-1]['natoms'])
    #print(exp_frame_rep[-1]['coords'])



    write_extended_xyz_new_data_only(exp_frame_rep, 'replicated_exp.xyz')


    # overlay structure
    overlayed_cell = [make_overlayed_frame(exp_frame_rep[-1], predicted_frames[-1], args.set_element_colour, args.rot_steps)]

    write_extended_xyz_new_data_only(overlayed_cell, 'overlayed_cell.xyz')


    '''for fi,fr in enumerate(frames):
        print(f"Unwrapping frame {fi+1}/{len(frames)} (natoms={fr['natoms']})")
        try:
            unwrap_frame(fr, tol=args.tol)
        except Exception as e:
            print(f"Error unwrapping frame {fi+1}: {e}")
            sys.exit(1)

    write_extended_xyz(frames, args.output, minpos_margin=args.minpos)
    print(f"Wrote unwrapped frames to {args.output}")'''

if __name__ == '__main__':
    main()

