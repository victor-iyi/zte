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
PE_{p,2i} = \sin\!\left(\frac{p}{10000^{2i/d}}\right),
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
# Requires `mne`. Provide the retained 105 electrode labels in channel-axis order.
# uv sync --group montage
python scripts/export_montage.py --out res/montage_gsn105.csv --keep-file res/zuco_channel_labels.txt
```

then point `dataset.montage_csv` at the CSV. **The CSV's channel index must line up with the channel axis of your EEG tensors** — the 23 dropped outer
electrodes must be excluded in the correct positions, otherwise harmonic column `c` describes the wrong electrode.
