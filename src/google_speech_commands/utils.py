import sys

from tensorboard.main import run_main


def launch_tb():
    if "--logdir" not in " ".join(sys.argv):
        sys.argv.extend(["--logdir", "runs"])
    sys.exit(run_main())
