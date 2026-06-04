# %%
# ============================================================
# Squid chromatophore self-organization toy model
# ------------------------------------------------------------
# Goal:
# 1. Model developmental "interstitial growth" of chromatophores
# 2. Show local inhibition can generate a globally uniform pigment-cell network
# 3. Add radial muscle fibers and a simple neural activation layer
# 4. Quantify emergence by nearest-neighbor CV and pair-correlation-like curve
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

# -----------------------------
# 1. Global parameters
# -----------------------------

SEED = 42
rng = np.random.default_rng(SEED)

L = 1.0                       # skin patch size: [0, L] x [0, L]
n_steps = 900                 # developmental simulation steps
initial_cells = 8             # initial chromatophores
max_cells = 320               # stop when reaching this number

candidate_per_step = 80       # candidate birth sites sampled per step
inhibition_radius = 0.055     # local exclusion length scale
birth_sharpness = 8.0         # larger -> stronger preference for "empty gaps"

mature_y_to_r = 80            # yellow -> red maturation time
mature_r_to_b = 180           # red -> black maturation time

muscle_min = 10               # each chromatophore has 10-20 radial muscle fibers
muscle_max = 20
muscle_length_base = 0.030
muscle_length_jitter = 0.008

snapshot_steps = [0, 100, 250, 500, 899]


# -----------------------------
# 2. Helper functions
# -----------------------------

def pairwise_dist_periodic(A, B, L=1.0):
    """
    Pairwise distance under periodic boundary condition.
    This avoids edge artifacts in the toy skin patch.
    """
    diff = A[:, None, :] - B[None, :, :]
    diff = diff - L * np.round(diff / L)
    return np.sqrt((diff ** 2).sum(axis=2))


def min_distance_to_existing(candidates, points, L=1.0):
    if len(points) == 0:
        return np.full(len(candidates), np.inf)
    D = pairwise_dist_periodic(candidates, points, L=L)
    return D.min(axis=1)


def nearest_neighbor_dist(points, L=1.0):
    """
    Nearest neighbor distance for each point.
    """
    if len(points) < 2:
        return np.array([])
    D = pairwise_dist_periodic(points, points, L=L)
    np.fill_diagonal(D, np.inf)
    return D.min(axis=1)


def nnd_metrics(points, L=1.0):
    nnd = nearest_neighbor_dist(points, L=L)
    if len(nnd) == 0:
        return dict(n=len(points), nnd_mean=np.nan, nnd_std=np.nan, nnd_cv=np.nan)
    return dict(
        n=len(points),
        nnd_mean=float(np.mean(nnd)),
        nnd_std=float(np.std(nnd)),
        nnd_cv=float(np.std(nnd) / (np.mean(nnd) + 1e-12))
    )


def maturation_state(age):
    """
    0: yellow, 1: red, 2: black
    """
    if age < mature_y_to_r:
        return 0
    elif age < mature_r_to_b:
        return 1
    else:
        return 2


def state_colors(states):
    """
    Use default matplotlib color cycle names instead of customized RGB.
    """
    color_map = {
        0: "gold",       # yellow-like
        1: "tab:red",   # red-like
        2: "black"      # black
    }
    return [color_map[int(s)] for s in states]


def generate_muscle_fibers(points, rng, L=1.0):
    """
    Generate local radial muscle fibers.
    Each pigment cell gets 10-20 nearly evenly spaced fibers.
    This is not the main self-organization mechanism;
    it is a simplified developmental readout.
    """
    segments = []
    for p in points:
        k = int(rng.integers(muscle_min, muscle_max + 1))
        angles = np.linspace(0, 2*np.pi, k, endpoint=False)
        angles += rng.normal(0, 0.08, size=k)
        lengths = muscle_length_base + rng.normal(0, muscle_length_jitter, size=k)
        lengths = np.clip(lengths, 0.01, 0.06)

        for theta, length in zip(angles, lengths):
            q = p + length * np.array([np.cos(theta), np.sin(theta)])
            # do not wrap for visualization; just keep local fibers
            if 0 <= q[0] <= L and 0 <= q[1] <= L:
                segments.append([p, q])
    return segments


def pair_correlation_like(points, L=1.0, bins=40, r_max=0.25):
    """
    Simple radial pair count curve.
    Not a fully normalized Ripley's K/g(r), but useful enough for this assignment.
    A strong dip near zero indicates local exclusion/self-organization.
    """
    if len(points) < 2:
        return np.array([]), np.array([])

    D = pairwise_dist_periodic(points, points, L=L)
    upper = D[np.triu_indices(len(points), k=1)]
    hist, edges = np.histogram(upper, bins=bins, range=(0, r_max))
    centers = 0.5 * (edges[:-1] + edges[1:])
    shell_area = np.pi * (edges[1:]**2 - edges[:-1]**2)
    density = len(points) / (L * L)

    # expected pair count in shell under spatial randomness, rough normalization
    expected = 0.5 * len(points) * density * shell_area
    g = hist / (expected + 1e-12)
    return centers, g


# -----------------------------
# 3. Developmental simulation
# -----------------------------

# initial random chromatophores
points = rng.uniform(0, L, size=(initial_cells, 2))
ages = np.zeros(initial_cells, dtype=int)

history = []
snapshots = {}

for t in range(n_steps):
    # record snapshot
    if t in snapshot_steps:
        snapshots[t] = (points.copy(), ages.copy())

    # update ages
    ages += 1

    # stop adding cells if saturated
    if len(points) < max_cells:
        candidates = rng.uniform(0, L, size=(candidate_per_step, 2))
        dmin = min_distance_to_existing(candidates, points, L=L)

        # local inhibition:
        # existing chromatophores suppress nearby births.
        # candidates in larger gaps get larger scores.
        score = 1 / (1 + np.exp(-birth_sharpness * (dmin - inhibition_radius) / inhibition_radius))

        # choose one birth site probabilistically
        prob = score / (score.sum() + 1e-12)
        new_idx = rng.choice(len(candidates), p=prob)
        new_point = candidates[new_idx:new_idx+1]

        points = np.vstack([points, new_point])
        ages = np.concatenate([ages, np.array([0])])

    # collect metrics
    m = nnd_metrics(points, L=L)
    m["step"] = t
    history.append(m)

history = pd.DataFrame(history)

# final snapshot
snapshots[n_steps - 1] = (points.copy(), ages.copy())

print(history.tail())



# %%
# -----------------------------
# 4. Plot developmental snapshots
# -----------------------------

fig, axes = plt.subplots(1, len(snapshot_steps), figsize=(4 * len(snapshot_steps), 4))

for ax, step in zip(axes, snapshot_steps):
    pts, ag = snapshots[step]
    states = np.array([maturation_state(a) for a in ag])
    ax.scatter(
        pts[:, 0], pts[:, 1],
        s=18,
        c=state_colors(states),
        alpha=0.85,
        edgecolors="none"
    )
    ax.set_title(f"Step {step}\nN={len(pts)}")
    ax.set_xlim(0, L)
    ax.set_ylim(0, L)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])

plt.suptitle("Developmental self-organization of chromatophore positions", y=1.03)
plt.tight_layout()
plt.show()

# %%
# -----------------------------
# 5. Quantify emergence:
# nearest-neighbor distance CV over development
# -----------------------------

plt.figure(figsize=(7, 4))
plt.plot(history["step"], history["nnd_cv"])
plt.xlabel("Developmental step")
plt.ylabel("CV of nearest-neighbor distance")
plt.title("Emergence of spatial regularity: lower CV means more uniform spacing")
plt.grid(alpha=0.3)
plt.show()

plt.figure(figsize=(7, 4))
plt.plot(history["step"], history["nnd_mean"])
plt.xlabel("Developmental step")
plt.ylabel("Mean nearest-neighbor distance")
plt.title("Mean spacing decreases as chromatophore density increases")
plt.grid(alpha=0.3)
plt.show()

# %%
# -----------------------------
# 6. Compare against random distribution
# -----------------------------

final_points = points.copy()
random_points = rng.uniform(0, L, size=final_points.shape)

final_nnd = nearest_neighbor_dist(final_points, L=L)
random_nnd = nearest_neighbor_dist(random_points, L=L)

summary = pd.DataFrame({
    "model": ["self_organized", "random"],
    "N": [len(final_points), len(random_points)],
    "mean_NND": [final_nnd.mean(), random_nnd.mean()],
    "std_NND": [final_nnd.std(), random_nnd.std()],
    "CV_NND": [
        final_nnd.std() / final_nnd.mean(),
        random_nnd.std() / random_nnd.mean()
    ]
})

print(summary)

plt.figure(figsize=(7, 4))
plt.hist(random_nnd, bins=30, alpha=0.6, label="random")
plt.hist(final_nnd, bins=30, alpha=0.6, label="self-organized")
plt.xlabel("Nearest-neighbor distance")
plt.ylabel("Count")
plt.title("Self-organized distribution is more regular than random")
plt.legend()
plt.grid(alpha=0.3)
plt.show()

# %%
# -----------------------------
# 7. Pair-correlation-like curve
# -----------------------------

r_self, g_self = pair_correlation_like(final_points, L=L)
r_rand, g_rand = pair_correlation_like(random_points, L=L)

plt.figure(figsize=(7, 4))
plt.plot(r_rand, g_rand, label="random")
plt.plot(r_self, g_self, label="self-organized")
plt.axhline(1.0, linestyle="--", linewidth=1)
plt.xlabel("Distance r")
plt.ylabel("Pair-correlation-like g(r)")
plt.title("Local exclusion creates a near-zero dip at short distances")
plt.legend()
plt.grid(alpha=0.3)
plt.show()

# %%
# -----------------------------
# 8. Add radial muscle fibers
# -----------------------------

segments = generate_muscle_fibers(final_points, rng, L=L)
states = np.array([maturation_state(a) for a in ages])

fig, ax = plt.subplots(figsize=(7, 7))

lc = LineCollection(segments, linewidths=0.4, alpha=0.35)
ax.add_collection(lc)

ax.scatter(
    final_points[:, 0], final_points[:, 1],
    s=20,
    c=state_colors(states),
    alpha=0.9,
    edgecolors="none"
)

ax.set_xlim(0, L)
ax.set_ylim(0, L)
ax.set_aspect("equal")
ax.set_xticks([])
ax.set_yticks([])
ax.set_title("Chromatophore-muscle local network")
plt.show()

# %%
# -----------------------------
# 9. Simple neural activation and emergent skin pattern
# -----------------------------

def neural_field(points, mode="stripe"):
    """
    A toy neural command field.
    This does not model true cephalopod vision.
    It only shows how a regular chromatophore array can serve as a dynamic pixel sheet.
    """
    x = points[:, 0]
    y = points[:, 1]

    if mode == "stripe":
        signal = 0.5 + 0.5 * np.sin(8 * np.pi * x)
    elif mode == "spots":
        signal = 0.5 + 0.5 * np.sin(10 * np.pi * x) * np.sin(10 * np.pi * y)
    elif mode == "gradient":
        signal = x
    else:
        signal = rng.uniform(0, 1, size=len(points))

    return np.clip(signal, 0, 1)


def plot_activation(mode):
    activation = neural_field(final_points, mode=mode)

    # chromatophore radius increases with neural activation
    radius = 8 + 80 * activation

    plt.figure(figsize=(6, 6))
    plt.scatter(
        final_points[:, 0], final_points[:, 1],
        s=radius,
        c=activation,
        alpha=0.75,
        edgecolors="none"
    )
    plt.xlim(0, L)
    plt.ylim(0, L)
    plt.gca().set_aspect("equal")
    plt.xticks([])
    plt.yticks([])
    plt.title(f"Emergent dynamic skin pattern: {mode}")
    plt.colorbar(label="neural activation / expansion")
    plt.show()


plot_activation("stripe")
plot_activation("spots")
plot_activation("gradient")

# %%
# -----------------------------
# 10. Parameter scan:
# stronger local inhibition should produce more regular spacing
# -----------------------------

def run_one_sim(inhibition_radius, birth_sharpness, seed=0, max_cells=250, n_steps=600):
    local_rng = np.random.default_rng(seed)
    pts = local_rng.uniform(0, L, size=(initial_cells, 2))
    ag = np.zeros(initial_cells, dtype=int)

    for t in range(n_steps):
        ag += 1
        if len(pts) >= max_cells:
            break

        cand = local_rng.uniform(0, L, size=(candidate_per_step, 2))
        dmin = min_distance_to_existing(cand, pts, L=L)
        score = 1 / (1 + np.exp(-birth_sharpness * (dmin - inhibition_radius) / inhibition_radius))
        prob = score / (score.sum() + 1e-12)
        idx = local_rng.choice(len(cand), p=prob)

        pts = np.vstack([pts, cand[idx:idx+1]])
        ag = np.concatenate([ag, np.array([0])])

    return nnd_metrics(pts, L=L)


scan_rows = []

radii = [0.025, 0.035, 0.045, 0.055, 0.070]
sharpness_values = [2.0, 4.0, 8.0, 12.0]

for rad in radii:
    for sharp in sharpness_values:
        cvs = []
        for rep in range(5):
            out = run_one_sim(rad, sharp, seed=1000 + rep, max_cells=250, n_steps=600)
            cvs.append(out["nnd_cv"])

        scan_rows.append({
            "inhibition_radius": rad,
            "birth_sharpness": sharp,
            "CV_NND_mean": np.mean(cvs),
            "CV_NND_std": np.std(cvs)
        })

scan = pd.DataFrame(scan_rows)
print(scan)

plt.figure(figsize=(7, 4))
for sharp in sharpness_values:
    sub = scan[scan["birth_sharpness"] == sharp]
    plt.plot(
        sub["inhibition_radius"],
        sub["CV_NND_mean"],
        marker="o",
        label=f"sharpness={sharp}"
    )

plt.xlabel("Local inhibition radius")
plt.ylabel("Mean CV of nearest-neighbor distance")
plt.title("Parameter scan: local inhibition controls spatial self-organization")
plt.legend()
plt.grid(alpha=0.3)
plt.show()


