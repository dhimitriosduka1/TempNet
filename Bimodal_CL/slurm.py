import subprocess
import datetime
import os


def get_remaining_slurm_time():
    """
    Returns remaining time (in seconds) for the Slurm job.
    Returns `None` if not running under Slurm.
    """
    try:
        job_id = os.environ.get("SLURM_JOB_ID")
        if not job_id:
            return None

        # Get job end time using `scontrol`
        cmd = f"scontrol show job {job_id} -o"
        output = subprocess.check_output(cmd, shell=True).decode()

        # Parse the `EndTime` field
        end_time_str = [
            x.split("=")[1] for x in output.split() if x.startswith("EndTime=")
        ][0]
        end_time = datetime.datetime.strptime(end_time_str, "%Y-%m-%dT%H:%M:%S")

        # Calculate remaining time
        now = datetime.datetime.now()
        remaining = (end_time - now).total_seconds()
        return max(0, remaining)  # Avoid negative values

    except Exception as e:
        print(f"Error checking Slurm time: {e}")
        return None
