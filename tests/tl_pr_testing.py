import os
import sys
import subprocess
from pathlib import Path



def main():
    SCRIPT_DIR = Path(__file__).resolve().parent
    test_root = SCRIPT_DIR / 'ultralytics' / 'cfg' / 'pr_tests'
    fail_pr = 0
    test_script = SCRIPT_DIR/ 'tests_leap_custom_test.py'

    for test_case in os.listdir(test_root):
        test_path = os.path.join(test_root, test_case, 'tl_default.yaml')
        env = os.environ.copy()
        env['DIR_PATH'] = test_path

        print(f"\n🔍 Running test for: {test_case}")

        try:
            subprocess.run(
                [sys.executable, test_script],
                env=env,
                capture_output=True,
                text=True,
                check=True
            )
            print(f"✅ Test passed for: {test_case}")

        except subprocess.CalledProcessError as err:
            print(f"❌ Test FAILED for: {test_case}")
            print("------ STDOUT ------")
            print(err.stdout)
            print("------ STDERR ------")
            print(err.stderr)

            # This creates a visible GitHub Actions error annotation
            print(f"::error file={test_script},line=1::Test failed for: {test_case}")

            fail_pr += 1

    if fail_pr > 0:
        print(f"\n❌ {fail_pr} test(s) failed. Blocking PR.")
        sys.exit(1)
    else:
        print("\n✅ All tests passed. PR is ready to be merged.")
        sys.exit(0)

if __name__ == '__main__':
    main()
