My first attempt at launching jobs on UAF condor included many steps and missteps. Here is the approach I settled on:

- Choose a container image: `/cvmfs/singularity.opensciencegrid.org/opensciencegrid/osgvo-el9:latest`
  - Python 3.9.16
  - This will be used in the worker jobs
- Create a python venv on /ceph with this image
  - `apptainer exec /cvmfs/singularity.opensciencegrid.org/opensciencegrid/osgvo-el9:latest bash`
  - `python -m venv env`
  - `source env/bin/activate`
  - `pip install -r ../../requirements.txt`
- Use this container and venv in the worker jobs
  - In submission file: `+SingularityImage = "/cvmfs/singularity.opensciencegrid.org/opensciencegrid/osgvo-el9:latest"`
  - `${MY_VENV}/env/bin/python main.py ...`


Things which caused missteps:

- Ideally: dont rerun `pip install ...` during every worker job
- Trying to run with my local venv doesn't work because my venv symlinks to a python executable which doesn't exist at the same path on the workers
- Trying to run with my local venv also doesn't work because many workers have a relatively old python installation (3.6)
- Trying to run with a python docker image doesn't work, with this error:
  - `+SingularityImage = "docker://ubuntu:latest"`
  - `ERROR   Unable to access the Singularity image: docker://ubuntu:latest`
- Trying to run apptainer directly in the condor executable doesn't work because a lot of workers don't have an apptainer exectuable
- Trying to run singularity directly in the condor exectuable doesn't work, with this error:
  - `INFO: Converting SIF file to temporary sandbox... `
  - `FATAL: while extracting /ceph/users/atuna/work/gavin_tc_dnn/condor/python310.singularity.sif: root filesystem extraction failed: extract command failed:`
  - `ERROR : No setuid installation found, for unprivileged installation use: ./mconfig --without-suid : exit status 1`
- Trying to run with a local Singularity image file (.sif) doesn't work, with this error:
  - `+SingularityImage = "/ceph/users/atuna/work/gavin_tc_dnn/condor/python310.sif"`
  - `/srv/.gwms-user-job-wrapper.sh: 44: [[: not found`
  - `/srv/.gwms-user-job-wrapper.sh: 45: [[: not found`
  - `/srv/.gwms-user-job-wrapper.sh: 51: [[: not found`
  - `/srv/.gwms-user-job-wrapper.sh: 101: [[: not found`
  - `/srv/.gwms-user-job-wrapper.sh: 103: [[: not found`
  - `/srv/.gwms-user-job-wrapper.sh: 109: [[: not found`
  - `/srv/.gwms-user-job-wrapper.sh: 111: [[: not found`
  - `ERROR: /srv/.gwms-user-job-wrapper.sh: Unable to source singularity_lib.sh! File not found. Quitting`
  - `/srv/.gwms-user-job-wrapper.sh: 60: [[: not found`
  - `/srv/.gwms-user-job-wrapper.sh: 66: [[: not found`

