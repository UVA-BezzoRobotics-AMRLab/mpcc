# syntax=docker/dockerfile:1
FROM ros:noetic
ARG GRB_VERSION=11.0.2
ARG GRB_SHORT_VERSION=11.0

LABEL Maintainer="Nick Mohammad <nm9ur@virginia.edu>" \
      Description="Safe Model Predictive Contour Controller"

# dependency layer
RUN apt-get update && apt-get upgrade -y && \
    apt-get install -y --no-install-recommends git wget vim build-essential && \
    apt-get install -y gcc g++ gfortran patch pkg-config liblapack-dev && \
    apt-get install -y libmetis-dev cppad ca-certificates ginac-tools libginac-dev && \
    apt-get install -y python3-catkin-tools python3-pip

# install ACADOS
WORKDIR /home
RUN git clone https://github.com/acados/acados.git && \
    cd acados && \
    git checkout 01452a6c902298da39947ccb6f44fb550cf51d07 && \
    git submodule update --init --recursive && \
    mkdir -p build && \
    cd build && \
    cmake -DACADOS_WITH_QPOASES=ON .. && \
    make install

ENV ACADOS_SOURCE_DIR=/home/acados
ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$ACADOS_SOURCE_DIR/lib
    
# install GUROBI
WORKDIR /opt

RUN export GRB_PLATFORM="linux64" && \
    update-ca-certificates && \
    wget -v https://packages.gurobi.com/${GRB_SHORT_VERSION}/gurobi${GRB_VERSION}_$GRB_PLATFORM.tar.gz && \
    tar -xvf gurobi${GRB_VERSION}_$GRB_PLATFORM.tar.gz  && \
    rm -f gurobi${GRB_VERSION}_$GRB_PLATFORM.tar.gz && \
    mv -f gurobi* gurobi && \
    rm -rf gurobi/$GRB_PLATFORM/docs && \
    mv -f gurobi/$GRB_PLATFORM*  gurobi/linux

WORKDIR /opt/gurobi/linux/src/build

RUN make && \
    cp libgurobi_c++.a /opt/gurobi/linux/lib && \
    mkdir -p /home/catkin_ws/src

ENV GUROBI_HOME=/opt/gurobi/linux
ENV PATH=$PATH:$GUROBI_HOME/bin
ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$GUROBI_HOME/lib


# install ROS packages and dependencies
RUN apt-get install -y --no-install-recommends ros-noetic-costmap-2d ros-noetic-tf2-geometry-msgs ros-noetic-gmapping && \
    apt-get install -y --no-install-recommends ros-noetic-rviz ros-noetic-backward-ros ros-noetic-pcl-ros ros-noetic-cv-bridge ros-noetic-filters

WORKDIR /home/catkin_ws

# RUN git clone https://github.com/nocholasrift/robust_fast_navigation.git src/robust_fast_navigation 

RUN git clone https://github.com/ANYbotics/grid_map.git src/grid_map && \
    rm -rf src/grid_map/grid_map_octomap && \
    rm -rf src/grid_map/grid_map_demos

# copy the package to the workspace and build
COPY . ./src/mpcc

RUN apt-get update && \
    apt-get install -y --no-install-recommends python3.8-venv && \
    python3.8 -m venv venv && \
    . /home/catkin_ws/venv/bin/activate && \
    pip install --upgrade pip && \
    pip install "setuptools>=61" && \
    pip install --upgrade importlib_metadata && \
    pip install importlib_resources && \
    pip install -r src/mpcc/requirements.txt && \
    pip install -e $ACADOS_SOURCE_DIR/interfaces/acados_template --use-pep517
    #mv ./src/mpcc/amrl_logging ./src

WORKDIR /home/catkin_ws/src/mpcc/scripts/tube_gen

# generate code to build tubes
RUN . /home/catkin_ws/venv/bin/activate && python3 tube_lp_gen.py --yaml=/home/catkin_ws/src/mpcc/params/unicycle_model_mpcc.yaml

WORKDIR /home/catkin_ws

#RUN /bin/bash -c "source /opt/ros/noetic/setup.bash && source /home/catkin_ws/venv/bin/activate && catkin build -DCMAKE_BUILD_TYPE=Release -DCMAKE_POLICY_VERSION_MINIMUM=3.5 -DPYTHON_EXECUTABLE=/usr/bin/python3"

RUN echo "source /opt/ros/noetic/setup.bash" >> /root/.bashrc && \
    echo "source /home/catkin_ws/devel/setup.bash" >> /root/.bashrc && \
    echo "set -o vi" >> /root/.bashrc

ENTRYPOINT ["/bin/bash"]

