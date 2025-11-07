# PCGrad Quick Reference

## 🎯 TL;DR

### Enable PCGrad (Recommended):
```yaml
optimization:
  use_session_mode: false
  # That's it! PCGrad enabled by default
```

### Disable PCGrad:
```yaml
optimization:
  use_session_mode: false
  use_pcgrad: false
```

---

## 📺 What You'll See in Terminal

### ✅ PCGrad Working (With Conflict):
```
├─ [PCGrad Status]
│  ├─ Config: use_pcgrad = True
│  ├─ 🎯 GRADIENT SIMILARITY:
│  │   ├─ Cosine: -0.2345 ⚠️ CONFLICT
│  │   │  └─ ⚠️  Mild conflict (gradients diverge)
│  └─ Action: ✅ APPLYING PCGrad

🔥 [PCGrad] Conflict detected! Projecting render gradients...
    ✅ PCGrad projection complete

├─ PCGrad:
│  ├─ Applied: ✅ YES
│  ├─ Cosine: -0.2345
│  └─ Projection scale: 0.123
```

### ✅ PCGrad Enabled (No Conflict):
```
├─ [PCGrad Status]
│  ├─ Config: use_pcgrad = True
│  ├─ 🎯 GRADIENT SIMILARITY:
│  │   ├─ Cosine: +0.4567 ✓ aligned
│  │   │  └─ ✅ Aligned (gradients cooperate)
│  └─ Action: ⏭️  Skipping (no conflict)

├─ PCGrad:
│  ├─ Applied: ❌ NO
│  └─ Reason: No conflict detected
```

### ❌ PCGrad Disabled:
```
├─ [PCGrad Status]
│  ├─ Config: use_pcgrad = False
│  ├─ 🎯 GRADIENT SIMILARITY:
│  │   ├─ Cosine: -0.2345 ⚠️ CONFLICT
│  └─ Action: ❌ DISABLED
    ⚠️  PCGrad disabled in config

├─ PCGrad:
│  ├─ Applied: ❌ NO
│  └─ Reason: Disabled in config
```

---

## 🎯 Cosine Similarity Guide

```
 +1.0  ✅ Perfect alignment (physics & render want same thing)
  ↑
 +0.3  ━━━━━━━━━━━━━━━ ✅ Aligned (cooperate)
  ↑
  0.0  ~ Orthogonal (independent)
  ↓
 -0.1  ━━━━━━━━━━━━━━━ ⚠️  PCGrad Threshold (mild conflict)
  ↓
 -0.3  ━━━━━━━━━━━━━━━ ⚠️  Strong conflict (oppose)
  ↓
 -1.0  ❌ Perfect opposition (physics & render fight)
```

**PCGrad activates when:** Cosine < -0.1 (default threshold)

---

## 🔍 Quick Checks

### 1. Is PCGrad Available?
**Look for:** `✅ [LEGACY MODE]` at episode start
- ✅ GOOD: PCGrad available
- ❌ BAD: `⚠️ [SESSION MODE]` → Add `use_session_mode: false`

### 2. Is PCGrad Enabled?
**Look for:** `Config: use_pcgrad = True`
- ✅ YES: PCGrad enabled
- ❌ NO: Set `use_pcgrad: true` in config

### 3. Is PCGrad Working?
**Look for:** `🎯 GRADIENT SIMILARITY: Cosine: ...`
- ✅ Always visible in every pass
- Shows conflict status and interpretation

### 4. Was PCGrad Applied?
**Look for:** `├─ PCGrad: Applied: ✅ YES`
- ✅ YES: Conflict resolved
- ❌ NO: Check reason (disabled or no conflict)

---

## 📝 Config Options

```yaml
optimization:
  # Required for PCGrad
  use_session_mode: false

  # Enable/disable PCGrad (default: true)
  use_pcgrad: true

  # Conflict threshold (default: -0.1)
  # Lower = more aggressive (e.g., -0.2)
  # Higher = less aggressive (e.g., -0.05)
  pcgrad_threshold: -0.1
```

---

## 🚨 Troubleshooting

| Problem | Solution |
|---------|----------|
| "⚠️ [SESSION MODE]" in logs | Add `use_session_mode: false` |
| "Config: use_pcgrad = False" | Set `use_pcgrad: true` or remove it |
| "Action: ❌ DISABLED" | Enable PCGrad in config |
| Never see conflict messages | Normal if gradients always aligned (check cosine values) |
| Cosine not visible | Update to latest code (similarity now always shown) |

---

## 📚 Documentation

- **Quick Guide:** `docs/HOW_TO_USE_PCGRAD.md`
- **Config Examples:** `configs/sp_to_by/PCGRAD_CONFIG_EXAMPLES.yaml`
- **Refactoring Details:** `docs/PCGRAD_REFACTORING_SUMMARY.md`
- **Critical Bug Info:** `docs/PCGRAD_CRITICAL_BUG.md`

---

## ✅ Checklist

Before running experiments, verify:

- [ ] Config has `use_session_mode: false`
- [ ] See `✅ [LEGACY MODE]` in logs
- [ ] See `Config: use_pcgrad = True`
- [ ] See `🎯 GRADIENT SIMILARITY` in every pass
- [ ] Cosine values are visible and make sense

**All good? You're ready to train with PCGrad!** 🚀
