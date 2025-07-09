import os, sys, subprocess


if __name__ == '__main__':
    fail_pr = 0
    tests_dir = os.listdir('ultralytics/cfg/pr_tests')
    dodo_path=r'tests_leap_custom_test.py'
    for test in tests_dir:
        DIR_PATH = os.path.join('ultralytics/cfg/pr_tests', test, "default.yaml")
        env = os.environ.copy()  # inherit parent vars
        env["DIR_PATH"] = str(DIR_PATH)
        try:

            completed = subprocess.run(
                [sys.executable, str(dodo_path)],
                env=env,  # pass the custom environment

                capture_output=True, text=True, check=True
            )
            print(f"✅  Test on {test} passed")
        except subprocess.CalledProcessError as err:
            print(f"\n❌  Test failed on {test}")
            print("----- child stdout -----")
            print(err.stdout)
            print("----- child stderr -----")
            print(err.stderr)
            fail_pr += 1
    if fail_pr:
        print("Test failed, PR denied")
    else:
        print("All tests passed! starting PR")