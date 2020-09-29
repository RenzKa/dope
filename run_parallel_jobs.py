import pickle
import os
from pathlib import Path
import subprocess
import time



def make_jobsub_file(video_path, job_number):
    os.makedirs("logs", exist_ok=True)
    os.makedirs("job_files", exist_ok=True)
    job_file = f"job_files/{job_number}.sh"
    qsub_template = f"""#!/bin/bash
#SBATCH --job-name=episode{job_number}-dope
#SBATCH -o logs/qsub_out{job_number}.log
#SBATCH -e logs/qsub_err{job_number}.log
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=katrin@robots.ox.ac.uk
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48gb
#SBATCH --time=6:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
# -------------------------------
source activate bsltrain
cd /users/katrin/coding/libs/dope/
python dope_BSLCP.py \\
--video_path {video_path}
"""
    with open(job_file, "w") as f:
        f.write(qsub_template)
    return job_file
 

def get_num_jobs(job_name="dope", username="katrin"):
    num_running_jobs = int(
        subprocess.check_output(
            f"SQUEUE_FORMAT2='username:7,name:130' squeue --sort V | grep {username} | grep {job_name} | wc -l",
            shell=True,
        )
        .decode("utf-8")
        .replace("\n", "")
    )
    max_num_parallel_jobs = int(open("max_num_job.txt", "r").read())
    return num_running_jobs, max_num_parallel_jobs

def main(video_paths):
    sdata = {}

    # Submit each episode as a separate job to run them in parallel
    # For each episode
    for eno, video in enumerate(video_paths):
        job_file = make_jobsub_file(video_path=video, job_number=eno)
        # HACK: Wait until submitting new jobs that the #jobs are at below max
        num_running_jobs, max_num_parallel_jobs = get_num_jobs()
        print(f"{num_running_jobs}/{max_num_parallel_jobs} jobs are running...")
        while num_running_jobs >= max_num_parallel_jobs:
            num_running_jobs, max_num_parallel_jobs = get_num_jobs()
            time.sleep(5)
        print(f"Submitting job {eno}/{len(video_paths)}: {job_file}")
        os.system(f"sbatch {job_file}")


if __name__ == "__main__":
    info_file = '/users/katrin/data/BSLCP/info/BSLCP_consecutive/info_cut_hist_10_clean_Katrin_merge_resize.pkl'
    video_folder = '/users/katrin/coding/libs/segmentation/bsltrain'
    with open(info_file, 'rb') as f: 
        info_data = pickle.load(f)

    video_list = list(set(info_data['videos']['org_name']))
    video_paths = [Path(video_folder) / Path(video) for video in video_list]

    #Path(str(video_paths[1]).replace('videos','Dope')).parent.mkdir(exist_ok=True, parents=True)

    main(video_paths)