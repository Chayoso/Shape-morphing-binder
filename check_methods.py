import sys
sys.path.insert(0, '/home/chayo/Desktop/Shape-morphing-binder')
import diffmpm_bindings

print("CompGraph methods:")
methods = [m for m in dir(diffmpm_bindings.CompGraph) if not m.startswith('_')]
for m in sorted(methods):
    print(f"  - {m}")

print(f"\nhas set_render_gradients: {'set_render_gradients' in methods}")
print(f"has run_e2e_pass_batched: {'run_e2e_pass_batched' in methods}")
print(f"has has_render_gradients: {'has_render_gradients' in methods}")
