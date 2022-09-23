# TrIP

I usually run the following commands to build and run the docker container:

        docker build -t trip .
        docker run -it --gpus all --shm-size=128g --ulimit memlock=-1 --ulimit stack=6710886400 --rm -v ${PWD}/results:/results trip:latest