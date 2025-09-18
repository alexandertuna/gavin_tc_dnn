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
