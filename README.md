# Installation
You can install the package either with or without docker. Since there are several required pacakges and solvers needed to support everything in this repository, it is suggested to install with Docker as everything will be done automatically. 

To build the image, run the following command in the top level directory of this repository:

```bash
docker build --tag=mpcc . # this will take a while :)
```

If you'd like to install it onto your machine, you can do so through `catkin build`

A trajectory generation repository is also cloned into the container for testing and demonstration purposes (found [here](https://github.com/nocholasrift/robust_fast_navigation.git)). In order to run the trajectory generation nodes, you will need to get a WLS Gurobi license file from the [Gurobi Web License Manager](https://license.gurobi.com/manager/licenses). Place the `gurobi.lic` file in the top level directory of the repository and then execute the following command to spin up a container:

```bash
docker run --volume=$PWD/gurobi.lic:/opt/gurobi/gurobi.lic --network=host -it mpcc
```

# Running Nodes
The nodes use topic names designed to work out of the box with a Clearpath Jackal in a Gazebo environment. When launching the MPCC, a particle filter is also run using gmapping for vehiclel localization. To run the mpcc, simply run the following launch file:

```bash
roslaunch mpcc jackal_mpc_track.launch
```

To run the trajectory generator, run the following:

```bash
roslaunch robust_fast_navigation planner_gurobi.launch
```

To view the planning process, you can open an rviz pane by using the rviz configuration in `rviz/`:

```bash
# run in top level directory of project
rviz -d rviz/planner.rviz
```

The planner will wait until an occupancy map is provided to the `/map` topic and an initial start value is published to `/gmapping/odometry`. To give a goal, publish one to `/move_base_simple/goal` in the command line or in rviz.

## Adjusting Parameters
There are several knobs than can be tuned for controller, and they can be found in [mpcc.yaml](./params/mpcc.yaml) and [robo_params.yaml](./params/robo_params.yaml). The former is used for tuning weights and other parameters of the controller, whereas the former is used to adjust the maximum body rates of the system.

## Unity-Vicon Bridge
To set up ROS_MASTER_URI, first echo the ROS_MASTER_URI
