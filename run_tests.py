import importlib
import sys
import traceback

MODULES = [
    "tests.test_part1_single_head",
    "tests.test_part2_causal_mask",
    "tests.test_part3_multi_head",
    "tests.test_bonus_kv_cache",
    "tests.test_bonus_weight_tying",
    "tests.test_model_generation",
]

def run_module(name):
    mod = importlib.import_module(name)
    test_fns = [getattr(mod, n) for n in dir(mod) if n.startswith("test_")]
    for fn in test_fns:
        fn()
    print(f"  [OK] {name}  ({len(test_fns)} tests)")


def main():
    print("Running Multi-Head Attention test suite\n" + "=" * 50)
    failures = 0
    for name in MODULES:
        try:
            run_module(name)
        except Exception:
            failures += 1
            print(f"  [FAIL] {name}")
            traceback.print_exc()
    print("=" * 50)
    if failures:
        print(f"{failures} module(s) failed.")
        sys.exit(1)
    print("All modules passed.")


if __name__ == "__main__":
    main()
