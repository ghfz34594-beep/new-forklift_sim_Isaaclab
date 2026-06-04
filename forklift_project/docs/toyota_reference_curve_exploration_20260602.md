# Toyota Reference Curve Exploration

Date: 2026-06-02

Paper: <https://arxiv.org/abs/2412.11503>

## Paper Facts

The Toyota paper exposes only a high-level reward-shaping reference:

- The positive reward uses deviation from a reference trajectory.
- The start is the forklift location at the start of the task.
- The terminal position is the pallet.
- The trajectory is based on an approximation of a clothoid curve.
- The trajectory remains fixed throughout the task.
- Reward terms query fork-center distance to the pallet, fork-center distance to the curve, and fork orientation error relative to the curve tangent.

The paper does not publish the actual curve generator, number of segments, boundary yaw selection, curvature constraints, solver, minimum turning radius, or reward weights.

The most consistent interpretation is: one reference curve is generated at each episode reset from that episode's start pose and pallet pose, then cached and kept frozen during the episode. It is not a teacher demonstration and not a globally shared path across all randomized starts.

## Candidate Proxies

The exploration script compares four reproducible Toyota-like proxies:

- `poly3`: cubic lateral polynomial in the pallet frame. It is the minimal clothoid-like approximation because, for small slopes, curvature is approximately linear along the path.
- `line_poly3`: short initial straight + cubic transition + terminal straight. This is closer to the sketch in Fig.3 but has sharper curvature behavior at the transition.
- `line_g2_quintic`: short initial straight + quintic transition with zero second derivative at both joins + terminal straight. This is the recommended proxy for reward shaping.
- `hermite_direct`: direct cubic Hermite connector from start pose to pallet pose. Smooth and low curvature, but it lacks the explicit terminal pallet-axis approach corridor implied by insertion.

These are proxy curves, not recovered author code.

## Current Artifacts

Generated with:

```bash
cd /data/jianshi/projects/forklift_sim_exp9
python scripts/toyota_pipeline/explore_toyota_reference_curve.py \
  --no-stamp \
  --output-dir outputs/toyota_reference_curve_exploration/current
```

Outputs:

- `outputs/toyota_reference_curve_exploration/current/summary.md`
- `outputs/toyota_reference_curve_exploration/current/manifest.json`
- `outputs/toyota_reference_curve_exploration/current/overlay_all_cases.svg`
- `outputs/toyota_reference_curve_exploration/current/case_*.svg`

Current default grid: x in `[-4.0, -3.0]`, y in `[-0.6, 0.6]`, yaw in `[-14.3239 deg, 14.3239 deg]`, 27 cases.

Aggregate result:

| model | endpoint ok | monotone | mean length m | max curvature 1/m |
| --- | ---: | ---: | ---: | ---: |
| `poly3` | 27/27 | 27/27 | 2.528 | 5.495 |
| `line_poly3` | 27/27 | 27/27 | 2.562 | 13.007 |
| `line_g2_quintic` | 27/27 | 27/27 | 2.581 | 10.419 |
| `hermite_direct` | 27/27 | 27/27 | 2.503 | 1.293 |

## Recommendation For Reward Experiments

Use `line_g2_quintic` as the Toyota-like reference curve:

- generated once at reset;
- fixed throughout the episode;
- queried by nearest sampled point for `r_cd`;
- queried by nearest sampled tangent for `r_cpsi`;
- terminal segment aligned with the pallet insertion axis.

Recommended ablation:

- `none`: no reference curve reward;
- `poly3`: minimal clothoid-like approximation;
- `line_g2_quintic`: Toyota-like recommended proxy;
- `hermite_direct`: smooth direct-connector contrast;
- keep RS/Dubins as a separate vehicle-kinematic baseline, not as the Toyota curve.

