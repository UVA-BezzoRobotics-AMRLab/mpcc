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
    apt-get install -y python3-catkin-tools

# install ACADOS
WORKDIR /home
RUN git clone https://github.com/acados/acados.git && \
    cd acados && \
    git submodule update --init --recursive && \
    mkdir -p build && \
    cd build && \
    cmake -DACADOS_WITH_QPOASES=ON -DACADOS_INSTALL_DIR=/usr/local/ .. && \
    make install
    
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

ENV GUROBI_HOME /opt/gurobi/linux
ENV PATH $PATH:$GUROBI_HOME/bin
ENV LD_LIBRARY_PATH $GUROBI_HOME/lib


# install ROS packages and dependencies
RUN apt-get install -y --no-install-recommends ros-noetic-costmap-2d ros-noetic-tf2-geometry-msgs ros-noetic-gmapping && \
    apt-get install -y --no-install-recommends ros-noetic-rviz ros-noetic-backward-ros ros-noetic-pcl-ros sqlite3 python3-pip

WORKDIR /home/catkin_ws

RUN git clone https://github.com/ANYbotics/grid_map.git src/grid_map

# copy the package to the workspace and build
COPY . ./src/mpcc

RUN pip install -r src/mpcc/requirements.txt

WORKDIR /home/catkin_ws/src/mpcc/scripts/tube_gen

# generate code to build tubes
RUN python tube_lp_gen.py --yaml=/home/catkin_ws/src/mpcc/params/mpcc.yaml

WORKDIR /home/catkin_ws

RUN /bin/bash -c "source /opt/ros/noetic/setup.bash && catkin build"

RUN echo "source /opt/ros/noetic/setup.bash" >> /root/.bashrc && \
    echo "source /home/catkin_ws/devel/setup.bash" >> /root/.bashrc

ENTRYPOINT ["/bin/bash"]

