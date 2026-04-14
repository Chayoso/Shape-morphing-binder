# Legacy Experiment Results Summary

## ablation_bob (32³, PPC=4, bob, 60ep)

| Method | Best α | Final α | Final Physics | Fp_dev | Chamfer mean |
|--------|--------|---------|---------------|--------|-------------|
| physics_only | 0.0738 | 0.0936 | 701.8 | 0.0000 | - |
| F_injection_no_plasticity | 0.0747 | 0.0969 | 2117.4 | 0.0000 | - |
| position_injection | 0.0763 | 0.0961 | 2199.6 | 0.0000 | - |
| full_method | 0.0747 | 0.0970 | 2095.0 | 0.0554 | 0.3714 |

## ablation_bob_ppc5 (32³, PPC=5, bob, 60ep)

| Method | Best α | Final α | Final Physics | Fp_dev | Chamfer mean |
|--------|--------|---------|---------------|--------|-------------|
| physics_only | 0.0854 | 0.0948 | 615.1 | 0.0000 | - |
| F_injection_no_plasticity | 0.0783 | 0.0904 | 946.2 | 0.0000 | - |
| full_method | 0.0783 | 0.0909 | 930.2 | 0.0084 | 0.1208 |

## morph_from_isosphere (full method, various targets)

| Target | Eps | Best α | Final α | Final Physics | Fp_dev | Chamfer mean |
|--------|-----|--------|---------|---------------|--------|-------------|
| bunny | 60 | 0.0852 | 0.0852 | 1402.9 | 0.0169 | 0.0665 |
| bob | 74 | 0.0378 | 0.0737 | 2083.0 | 0.0213 | 0.0880 |
| spot | 60 | 0.0730 | 0.0750 | 850.0 | 0.0326 | 0.0801 |
| bunny_90ep | 19 | 0.1583 | 0.1603 | 2541.6 | 0.0000 | - |

## Key Observations

### ablation_bob (PPC=4)
- PO physics converges to 701.8, others to ~2117-2200
- Best alpha: PO=0.0738, F_inj=0.0747, pos_inj=0.0763, full=0.0747
- F injection and position injection hurt physics (3x worse) with marginal alpha gain
- full_method (F_inj + plasticity): similar alpha to F_inj alone, Fp_dev=0.0554

### ablation_bob_ppc5
- PPC=5 improves PO physics (615.1 vs PPC=4 701.8)
- F_injection also better physics (946.2 vs 2117.4)
- full_method Fp_dev much smaller (0.0084 vs 0.0554)

### morph_from_isosphere
- Bob best alpha=0.0378 (74ep), but rebound to 0.0737
- Bunny best alpha=0.0852 at final ep (no rebound)
- Spot best alpha=0.0730, slight rebound
- All use chamfer-guided plasticity (chamfer_mean tracked)
