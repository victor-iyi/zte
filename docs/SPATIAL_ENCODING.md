# Electrode spatial positional encoding

ZTE's sequence positional encodings (RoPE, sinusoidal, ALiBi — see [ARCHITECTURE.md](ARCHITECTURE.md)) answer *when* a word occurred: position along the one‑dimensional token axis. They say nothing about *where on the scalp* each of the 105 EEG channels sits. Until now the channel axis carried **no** geometry at all: `BandPowerMLP` flattens `(n_bp_features, n_channels)` into one vector and `RawConformer` feeds channels straight into a `Conv1d`, so the electrode's scalp position is only implicit in an arbitrary column index that the weights must memorise. The coarse 8‑band `RegionMap` is the only spatial notion, and it is explicitly approximate.

`model.spatial_encoding = 'spherical_harmonics'` adds the mathematically correct spatial analogue.

## Why spherical harmonics

Sinusoidal position encoding works because $\sin/\cos$ at geometric frequencies are the Fourier eigenbasis of the **line** under translation — which is exactly why relative distance falls out of an inner product (and why RoPE works). The scalp is not a line; it is (topologically) a **sphere** $S^2$. The correct generalisation of that Fourier basis to the sphere is the family of **real spherical harmonics** $Y_\ell^m(\theta,\phi)$ — the eigenfunctions of the Laplace–Beltrami operator $\Delta_{S^2}$ on $S^2$ and a complete orthonormal basis for functions on the sphere:

$$
\Delta_{S^2}\,Y_\ell^m = -\ell(\ell+1)\,Y_\ell^m
$$

Where sinusoidal encoding lays a geometric‑frequency ladder along the one‑dimensional token line,

$$
PE_{p,2i} = \sin\!\Big(\frac{p}{10000^{2i/d}}\Big),
$$

spherical harmonics lay the analogous frequency ladder over the scalp sphere. They are to the sphere what sines are to the line:

- **Multi‑resolution frequency ladder.** Degree $\ell$ is angular frequency. $\ell=0$ is constant; $\ell=1$ are the three dipolar left–right / front–back / up–down gradients; higher $\ell$ resolves progressively finer scalp patterns. This mirrors the $10000^{2i/d}$ frequency ladder of sinusoidal encoding, but on the sphere. `spatial_harmonic_degree` sets $\ell_{\max}$; the basis has $(\ell_{\max}+1)^2$ functions.
- **Rotation structure.** A rotation of the head, an element of $\mathrm{SO}(3)$, mixes harmonics *within* a degree via the Wigner‑D matrices and never across degrees — the spherical analogue of "translation acts by a phase shift", which makes the encoding a faithful, equivariant position code.
- **Geodesic locality (addition theorem).** The inner product between two electrodes' harmonic feature vectors is a function of the geodesic distance between them:

$$
\sum_{m=-\ell}^{\ell} Y_\ell^m(a)\,Y_\ell^m(b) = \frac{2\ell + 1}{4\pi}\,P_\ell(\cos\gamma)
$$

where $\gamma = \arccos(a^\top b)$ is the great‑circle angle between electrodes $a$ and $b$. So nearby electrodes get similar encodings, and the module's **learnable per‑degree gains** turn this into a learnable, rotation‑invariant kernel of scalp distance.

The harmonics are *exact* for any electrode coordinates. All approximation lives in the coordinates (see below), never in the encoding.

## Why not a graph network, and why not 2-D interpolation

The three standard ways to give a model electrode geometry are a graph neural network over an adjacency matrix, a
projection onto a 2-D scalp image, and a positional code on the sphere. They differ in what they assume and in what
survives a cap that sits two centimetres off.

| | what it needs | what a head rotation does to it | what an electrode shift does to it | is it exact? |
| --- | --- | --- | --- | --- |
| **GNN over an electrode graph** | a hand-chosen adjacency or $k$-NN threshold | nothing, but only because the graph already discarded orientation | changes the edge set, so the learned filters index different neighbours | no -- the graph is a discretisation of the geometry, chosen before training |
| **2-D interpolation / topographic image** | an azimuthal projection to a plane | changes the image, because the projection has a pole | moves a sample between grid cells, and interpolation invents values between electrodes | no -- the projection distorts area and angle away from its centre |
| **Spherical harmonics** | electrode coordinates on $S^2$ | mixes harmonics within a degree by a Wigner-D matrix, never across degrees | moves a point on the sphere; the code is a continuous function of that point | yes -- exact for any coordinates |

Three consequences follow, and they are the reason the spatial encoder is a contribution rather than a preprocessing
step.

**A graph throws away the quantity the problem is about.** An adjacency matrix records *which* electrodes are
neighbours, not *how far apart* they are or *in which direction*. Two caps whose $k$-NN graphs are isomorphic but
whose inter-electrode distances differ by a centimetre are indistinguishable to a GNN, and the threshold that built
the graph is a hyperparameter chosen before any data is seen. The harmonic code has no threshold: the addition
theorem above makes the inner product between two electrodes' codes an exact function of their geodesic separation,
so distance is in the representation rather than in a preprocessing decision.

**A 2-D projection has a pole, and the scalp does not.** Every azimuthal or stereographic projection of a sphere onto
a plane distorts, and the distortion grows with angular distance from the projection centre. On a 105-channel EGI net
that means occipital and frontal electrodes are represented on a different effective scale than central ones, and a
convolution kernel that is isotropic in the image is anisotropic on the head. Interpolating onto a grid compounds it
by manufacturing values at positions where nothing was measured -- a smoothness prior imposed silently, before the
model gets a vote. The harmonics are defined on the sphere itself, so there is no pole and no interpolation.

**Electrode shift is a continuity property, not a robustness trick.** The clinical case this project exists for is a
person whose cap is placed by a carer, differently each day. Under a shift, a GNN's edge set changes discretely and a
grid sample jumps cells; the harmonic code moves continuously, because $Y_\ell^m$ is a smooth function of position
and degree $\ell$ bounds how fast it can vary. The learnable per-degree gains are what make that a *learnable*
kernel of scalp distance: the model chooses how much weight to give coarse dipolar structure ($\ell = 1$) against
fine local structure ($\ell = 6$), rather than having that choice frozen into an adjacency threshold or a grid
resolution.

The cost is that the harmonics need coordinates, which a graph does not. That is a real dependency and the next
section is about what happens when it is unmet.

## What the shipped runs actually used

**Every live configuration in this repository runs on the approximate geometry.** No config under
`experiments/flagship/`, `experiments/decoder/`, `experiments/parallax/` or `experiments/alignment/` sets
`dataset.montage_csv`, so `resolve_geometry` falls back to the Fibonacci cap, logs it, and sets `approximate=True`.
That flag is a **persistent** buffer beside the harmonic basis, so it travels inside the checkpoint and reports the
geometry the numbers were computed under rather than whatever the loading machine can find. Read
`SphericalHarmonicEncoding.approximate_geometry`; never re-derive it from the config.

What this costs, precisely:

- The mathematical properties above **still hold** -- the Fibonacci cap is a set of genuine, well-separated points on
  $S^2$, so rotation structure, geodesic locality and the addition theorem are exact for *those* points.
- What is lost is the **correspondence to a real head**. Harmonic column $c$ describes a point on a sphere that is not
  where electrode $c$ was. So the encoding is a usable, geometry-shaped inductive bias, and **any per-channel scalp
  claim from such a run is positionally meaningless** -- a topographic map of channel importance from an
  `approximate=True` checkpoint shows which array indices mattered, not which brain regions.

A scalp topography is therefore only interpretable from a run trained with `dataset.montage_csv` pointed at a real
montage, exported as in *Getting exact coordinates* below. `resolve_geometry` also swallows a malformed CSV -- it
catches `ValueError`, `KeyError` and `FileNotFoundError`, warns, and falls back -- so a run can look configured for a
real montage and silently be on the cap. Check the checkpoint's flag, not the YAML.

## How it is wired in

`zte.models.spatial` provides:

- `real_spherical_harmonics(theta, phi, l_max)` — the validated real (tesseral, Condon–Shortley) basis.
- `ScalpGeometry` — electrode positions on the unit sphere, with constructors `from_csv` (`channel,x,y,z` or `channel,theta,phi`), `from_mne` (`GSN-HydroCel-128`), `from_xyz`, and the coordinate‑free `fibonacci_fallback`.
- `SphericalHarmonicEncoding` — precomputes the harmonic matrix (one row per electrode, $(\ell_{\max}+1)^2$ columns) as a fixed buffer, applies learnable per‑degree gains, and projects to the frontend width.
- `SpatialChannelMixer` — adds that per‑electrode encoding to the channel features and (when `spatial_mix=True`) runs one self‑attention layer over the electrodes‑as‑tokens, so each electrode is contextualised by geometrically related electrodes before the frontend consumes the channel axis.

Both coordinate forms describe the same unit‑sphere point under the convention

$$
x=\sin\theta\cos\phi,\quad y=\sin\theta\sin\phi,\quad z=\cos\theta,
$$

and the geodesic angle between two electrodes is $\gamma=\arccos(a^\top b)$.

It is injected at the **channel** level, orthogonal to the word‑sequence `pos_encoding`:

- **Band power:** the `n_bp_features × n_channels` block is reshaped to electrode tokens `(n_channels, n_bp_features)`, encoded/mixed, and re‑flattened; appended eye‑tracking scalars pass through untouched.
- **Raw:** the mixer is applied to `(n_channels, time_steps)` windows *before* the temporal convolution mixes channels.

Everything is opt‑in: with the default `spatial_encoding='none'` the frontends are byte‑identical to before. Geometry (`n_channels`, `montage_csv`) is persisted in the checkpoint so inference rebuilds the identical modules.

## Configuration

```yaml
model:
  spatial_encoding: spherical_harmonics   # default 'none'
  spatial_harmonic_degree: 6              # l_max; (l_max+1)^2 harmonics
  spatial_mix: true                       # electrodes-as-tokens self-attention on top of the additive code
  spatial_encoding_learnable: true        # trainable per-degree gains + projection
dataset:
  montage_csv: res/montage_gsn105.csv     # channel,x,y,z for exact geometry (see below)
```

## Getting exact coordinates (recommended)

Without a montage the module uses a smooth, well‑separated **Fibonacci‑cap** placeholder and logs `approximate=True` (mirroring `RegionMap`). This makes the encoding *usable* but not geometrically true. For real accuracy, supply the electrode coordinates of the ZuCo EGI net:

```sh
# Requires `mne` (uv sync --group spatial). ZuCo uses standard EGI ordering, so --zuco105
# reproduces the retained 105-channel montage with no manual channel list:
python scripts/export_montage.py --out res/montage_gsn105.csv --zuco105
```

then point `dataset.montage_csv` at the CSV. **The CSV's channel index must line up with the channel axis of your EEG tensors** — the 23 dropped outer
electrodes must be excluded in the correct positions, otherwise harmonic column `c` describes the wrong electrode.
