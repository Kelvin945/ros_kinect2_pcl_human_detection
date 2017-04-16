#include <ros/ros.h>
// PCL specific includes
#include <sensor_msgs/PointCloud2.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>

#include <pcl/point_types.h>
#include <pcl/conversions.h>
#include <pcl_ros/transforms.h>
#include <pcl/visualization/pcl_visualizer.h>    
#include <pcl/people/ground_based_people_detection_app.h>
#include <pcl/visualization/cloud_viewer.h>

ros::Publisher pub;


typedef pcl::PointXYZRGBA PointT;
typedef pcl::PointCloud<PointT> PointCloudT;

bool new_cloud_available_flag;

pcl::visualization::PCLVisualizer viewer("PCL Viewer");
pcl::PointCloud<PointT>::Ptr cloud (new PointCloudT);
pcl::people::GroundBasedPeopleDetectionApp<PointT> people_detector;    // people detection object


void ros_pointcloud_cb(const sensor_msgs::PointCloud2ConstPtr& cloud_msg, PointCloudT::Ptr& cloud_input, bool* new_cloud_available_flag)
{
  // Container for original & filtered data
  pcl::PCLPointCloud2* cloud2 = new pcl::PCLPointCloud2; 
  pcl::PCLPointCloud2ConstPtr cloudPtr(cloud2);
  
  // Convert to PCL data type
  pcl_conversions::toPCL(*cloud_msg, *cloud2);

  pcl::PointCloud<pcl::PointXYZRGBA> output;
  pcl::fromROSMsg(*cloud_msg, output);
  
  pcl::PointCloud<pcl::PointXYZRGBA>::Ptr m_ptrCloud(new pcl::PointCloud<pcl::PointXYZRGBA>(output));
  
  *new_cloud_available_flag = true;
  *cloud_input = *m_ptrCloud;
  
}


struct callback_args{
  // structure used to pass arguments to the callback function
  PointCloudT::Ptr clicked_points_3d;
  pcl::visualization::PCLVisualizer::Ptr viewerPtr;
};

void mouseclick_cb_(const pcl::visualization::PointPickingEvent& event, void* args)
{
  struct callback_args* data = (struct callback_args *)args;
  if (event.getPointIndex () == -1)
    return;
  PointT current_point;
  event.getPoint(current_point.x, current_point.y, current_point.z);
  data->clicked_points_3d->points.push_back(current_point);
  //Draw clicked points in red:
  pcl::visualization::PointCloudColorHandlerCustom<PointT> red (data->clicked_points_3d, 255, 0, 0);
  data->viewerPtr->removePointCloud("clicked_points");
  data->viewerPtr->addPointCloud(data->clicked_points_3d, red, "clicked_points");
  data->viewerPtr->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 10, "clicked_points");
  std::cout << current_point.x << " " << current_point.y << " " << current_point.z << std::endl;
}

int main (int argc, char** argv)
{
  // Initialize ROS
  ros::init (argc, argv, "people_counting");
  ros::NodeHandle nh;
  
  // Create a ROS subscriber for the input point cloud
  new_cloud_available_flag = false;

  //spend 2hrs to write this line Q_Q
  boost::function<void (const sensor_msgs::PointCloud2ConstPtr&)> f = boost::bind (&ros_pointcloud_cb, _1, cloud, &new_cloud_available_flag);
  ros::Subscriber sub = nh.subscribe<sensor_msgs::PointCloud2ConstPtr> ("/kinect2/sd/points", 1, f);
  
  // Algorithm parameters:
  std::string svm_filename = "/home/user/dev/catkin_ws/src/people_counting/trainedLinearSVMForPeopleDetectionWithHOG.yaml";
  float min_height = 1.3;
  float max_height = 2.3;
  float voxel_size = 0.06;
  float min_confidence = -1.5;
  
  Eigen::Matrix3f rgb_intrinsics_matrix;
  rgb_intrinsics_matrix << 525, 0.0, 319.5, 0.0, 525, 239.5, 0.0, 0.0, 1.0; // Kinect RGB camera intrinsics

  while(!new_cloud_available_flag) 
  {
    cout << "Waiting for initial pointcloud frame" << endl;
    ros::spinOnce ();
  }


  pcl::visualization::PointCloudColorHandlerRGBField<PointT> rgb(cloud);
  viewer.addPointCloud<PointT> (cloud, rgb, "input_cloud");
  viewer.setCameraPosition(0,0,-2,0,-1,0,0);

  // Add point picking callback to viewer:
  struct callback_args cb_args;
  PointCloudT::Ptr clicked_points_3d (new PointCloudT);
  cb_args.clicked_points_3d = clicked_points_3d;
  cb_args.viewerPtr = pcl::visualization::PCLVisualizer::Ptr(&viewer);
  viewer.registerPointPickingCallback (mouseclick_cb_, (void*)&cb_args);
  std::cout << "Shift+click on three floor points, then press 'Q'..." << std::endl;

  viewer.spin();
  std::cout << "done." << std::endl;

  // Ground plane estimation:
  Eigen::VectorXf ground_coeffs;
  ground_coeffs.resize(4);
  std::vector<int> clicked_points_indices;
  for (unsigned int i = 0; i < clicked_points_3d->points.size(); i++)
    clicked_points_indices.push_back(i);
  pcl::SampleConsensusModelPlane<PointT> model_plane(clicked_points_3d);
  model_plane.computeModelCoefficients(clicked_points_indices,ground_coeffs);
  std::cout << "Ground plane: " << ground_coeffs(0) << " " << ground_coeffs(1) << " " << ground_coeffs(2) << " " << ground_coeffs(3) << std::endl;

  // Initialize new viewer:
  pcl::visualization::PCLVisualizer viewer("PCL Viewer");          // viewer initialization
  viewer.setCameraPosition(0,0,-2,0,-1,0,0);

  // Create classifier for people detection:
  pcl::people::PersonClassifier<pcl::RGB> person_classifier;
  person_classifier.loadSVMFromFile(svm_filename);   // load trained SVM

  // People detection app initialization:
  pcl::people::GroundBasedPeopleDetectionApp<PointT> people_detector;    // people detection object
  people_detector.setVoxelSize(voxel_size);                        // set the voxel size
  people_detector.setIntrinsics(rgb_intrinsics_matrix);            // set RGB camera intrinsic parameters
  people_detector.setClassifier(person_classifier);                // set person classifier
  people_detector.setHeightLimits(min_height, max_height);         // set person classifier
  // people_detector.setSensorPortraitOrientation(true);             // set sensor orientation to vertical
  while (!viewer.wasStopped())
  {
      if (new_cloud_available_flag)    // if a new cloud is available
      {
        new_cloud_available_flag = false;

        // Perform people detection on the new cloud:
        std::vector<pcl::people::PersonCluster<PointT> > clusters;   // vector containing persons clusters
        people_detector.setInputCloud(cloud);
        people_detector.setGround(ground_coeffs);                    // set floor coefficients
        people_detector.compute(clusters);                           // perform people detection

        ground_coeffs = people_detector.getGround();                 // get updated floor coefficients

        // Draw cloud and people bounding boxes in the viewer:
        viewer.removeAllPointClouds();
        viewer.removeAllShapes();
        pcl::visualization::PointCloudColorHandlerRGBField<PointT> rgb(cloud);
        viewer.addPointCloud<PointT> (cloud, rgb, "input_cloud");
        unsigned int k = 0;
        for(std::vector<pcl::people::PersonCluster<PointT> >::iterator it = clusters.begin(); it != clusters.end(); ++it)
        {
          if(it->getPersonConfidence() > min_confidence)             // draw only people with confidence above a threshold
          {
            // draw theoretical person bounding box in the PCL viewer:
            it->drawTBoundingBox(viewer, k);
            k++;
          }
        }
        std::cout << k << " people found" << std::endl;
        ros::spinOnce ();
        viewer.spinOnce();
      }
    
  }
   
   
}