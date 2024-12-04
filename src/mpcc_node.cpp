#include <ros/ros.h>
#include <uav_mpc/mpcc_ros.h>

int main(int argc, char **argv)
{
    ros::init(argc, argv, "mpcc_ros");
    ros::NodeHandle nh;

    MPCCROS mpcc_ros(nh);
    ros::AsyncSpinner spinner(1);
    spinner.start();
    ros::waitForShutdown();

    return 0;
}
