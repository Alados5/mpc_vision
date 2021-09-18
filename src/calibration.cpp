#include <stdio.h>
#include <iostream>
#include <fstream>
#include <string>
#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Sparse>
#include "general_functions.h"
#include <ros/ros.h>
#include <mpc_vision/ClothMesh.h>
#include "geometry_msgs/PoseArray.h"
#include "geometry_msgs/Pose.h"
#include "geometry_msgs/Quaternion.h"
#include <tf2/LinearMath/Quaternion.h>
#include <tf2/LinearMath/Matrix3x3.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>
//#include <std_srvs/Empty.h>


using namespace std;


// Camera to End Effector Transform
Eigen::MatrixXf TCE = Eigen::MatrixXf::Zero(4,4);
// Base to End Effector Transform
Eigen::MatrixXf TRE = Eigen::MatrixXf::Zero(4,4);
// Robot Base to Camera Transform
Eigen::MatrixXf TRC = Eigen::MatrixXf::Zero(4,4);
// Calibration Weight
int NW = 0;
// Offset from cloth top to TCP
double OffsetClothTCP = 0.09;


// ROS SUBSCRIBERS - CALLBACK FUNCTIONS
void meshReceived(const mpc_vision::ClothMesh& msg) {

  // Extract data
  int nSOM = msg.size;
  
  // Camera base: x horizontal, y down, z depth
  Eigen::VectorXf xCAM = Eigen::Map<Eigen::VectorXf, Eigen::Unaligned>
                                   (((vector<float>) msg.x).data(), nSOM*nSOM);
  Eigen::VectorXf yCAM = Eigen::Map<Eigen::VectorXf, Eigen::Unaligned>
                                   (((vector<float>) msg.y).data(), nSOM*nSOM);
  Eigen::VectorXf zCAM = Eigen::Map<Eigen::VectorXf, Eigen::Unaligned>
                                   (((vector<float>) msg.z).data(), nSOM*nSOM);
	
	float dxC = xCAM.maxCoeff() - xCAM.minCoeff();
	
	// Reshape in square matrices
	Eigen::Map<Eigen::MatrixXf> xCsq(xCAM.data(), nSOM,nSOM);
	Eigen::Map<Eigen::MatrixXf> yCsq(yCAM.data(), nSOM,nSOM);
	Eigen::Map<Eigen::MatrixXf> zCsq(zCAM.data(), nSOM,nSOM);
	
	// Desired order: x increasing, y decreasing
	if (xCsq.col(0).maxCoeff() - xCsq.col(0).minCoeff() < 0.4*dxC) {
	  // First nSOM on same x --> Transpose!
	  xCsq.transposeInPlace();
	  yCsq.transposeInPlace();
	  zCsq.transposeInPlace();
	}
	if (xCsq(nSOM-1,0) - xCsq(0,0) < 0.0) {
	  // Decreasing x --> Invert!
	  xCsq = xCsq.colwise().reverse().eval();
	  yCsq = yCsq.colwise().reverse().eval();
	  zCsq = zCsq.colwise().reverse().eval();
	}
	if (yCsq(0,nSOM-1) - yCsq(0,0) > 0.0) {
	  // Increasing y --> Invert!
    xCsq = xCsq.rowwise().reverse().eval();
	  yCsq = yCsq.rowwise().reverse().eval();
	  zCsq = zCsq.rowwise().reverse().eval();
	}
	
	// Reshape back to vectors
	Eigen::Map<Eigen::VectorXf> xCAMord(xCsq.data(), nSOM*nSOM);
	Eigen::Map<Eigen::VectorXf> yCAMord(yCsq.data(), nSOM*nSOM);
	Eigen::Map<Eigen::VectorXf> zCAMord(zCsq.data(), nSOM*nSOM);
	
	// Get cloth base vectors
	Eigen::Vector3f xE, yE, zE;
	
	yE << xCAMord(nSOM*nSOM-1) - xCAMord(nSOM*(nSOM-1)),
	      yCAMord(nSOM*nSOM-1) - yCAMord(nSOM*(nSOM-1)),
	      zCAMord(nSOM*nSOM-1) - zCAMord(nSOM*(nSOM-1));
	yE.normalize();

	zE << xCAMord(0) - xCAMord(nSOM*(nSOM-1)),
	      yCAMord(0) - yCAMord(nSOM*(nSOM-1)),
	      zCAMord(0) - zCAMord(nSOM*(nSOM-1));
	zE.normalize();
	
	xE = yE.cross(zE);
	
	// Upper center point (near TCP, cloth origin)
	float xM = xCAMord.tail(nSOM).sum()/nSOM;
	float yM = yCAMord.tail(nSOM).sum()/nSOM;
	float zM = zCAMord.tail(nSOM).sum()/nSOM;
	
	//Offset from cloth top to TCP (negative y = up)
	yM -= OffsetClothTCP;
	
	// New camera to End Effector Transform
	Eigen::MatrixXf TEi(4,4);
	
	TEi.block(0,0, 3,3) << xE,yE,zE;
	TEi.block(0,3, 3,1) << xM,yM,zM;
	TEi.block(3,0, 1,4) << 0,0,0,1;
	
	float Tdiff = (TCE-TEi).array().abs().sum()/12;
	float wNewData = 1;
	
	if (!TEi.hasNaN()) {
	
		if (NW==0) {
		  NW++;
			TCE = TEi;
		}
		else if (Tdiff < 0.025) {
		  NW++;
			TCE = (TCE*(NW-1) + TEi*wNewData)/(NW-1+wNewData);
		}
	
	}
	
}

void tcpReceived(const geometry_msgs::PoseStamped& msg) {

  // Set position
  TRE(0,3) = msg.pose.position.x;
  TRE(1,3) = msg.pose.position.y;
  TRE(2,3) = msg.pose.position.z;
  
  // Set orientation
  geometry_msgs::Quaternion Rquat = msg.pose.orientation;
  
  Eigen::Quaterniond ERQuat;
  Eigen::Vector4d TCPori;
  TCPori << Rquat.x, Rquat.y, Rquat.z, Rquat.w;
  ERQuat = TCPori;
  
  Eigen::Matrix3d TRErot = ERQuat.normalized().toRotationMatrix();
  TRE.block(0,0, 3,3) = TRErot.cast<float>();
  
  // Set scale
  TRE(3,3) = 1;
  
}



// ---------------------------
// -------- MAIN PROG --------
// ---------------------------

int main(int argc, char **argv) {

	// Initialize the ROS system and become a node.
  ros::init(argc, argv, "calibration");
  ros::NodeHandle rosnh;
  
	// Define client objects to all services
  //ros::ServiceClient clt_foo = rosnh.serviceClient<node::Service>("service_name");
  //ros::service::waitForService("service_name");
  
  // Define Publishers
  ros::Publisher pub_cambase = rosnh.advertise<geometry_msgs::Pose>
                               ("/mpc_controller/camera_pose", 1000);
  
  // Define Subscribers
  ros::Subscriber sub_mesh = rosnh.subscribe("/cloth_segmentation/cloth_mesh",
                                             1000, &meshReceived);
  ros::Subscriber sub_tcp = rosnh.subscribe("/iri_wam_controller/libbarrett_link_tcp",
                                             1000, &tcpReceived);
	
	// Initial EE pose to avoid problems
	TRE << 0, 1, 0, 0.0,
	       1, 0, 0,-0.4,
	       0, 0,-1, 0.2,
	       0, 0, 0, 1.0;
	
	// START LOOP	
	ros::Rate rate(20);
	ROS_WARN_STREAM("Started Calibration process...");
	while(rosnh.ok()) {
	
		// Check subscriptions
		ros::spinOnce();
		
		// Multiply (R-E)*inv(C-E) to get R-C, Cam frame from Robot base
		TRC = TRE*TCE.inverse();
		
		// Publish Camera Transform as a Pose	(if not NaN)	
		if (!TRC.hasNaN()) {
		  
		  Eigen::Matrix3d TRC_Rd = TRC.block(0,0, 3,3).cast<double>();
		
		  tf2::Matrix3x3 tfR3;
		  tfR3.setValue(TRC_Rd(0,0), TRC_Rd(0,1), TRC_Rd(0,2),
		                TRC_Rd(1,0), TRC_Rd(1,1), TRC_Rd(1,2),
		                TRC_Rd(2,0), TRC_Rd(2,1), TRC_Rd(2,2));
		
		  tf2::Quaternion tfQ;
		  tfR3.getRotation(tfQ);
		
		  geometry_msgs::Quaternion msgQ = tf2::toMsg(tfQ);
		
		  geometry_msgs::Pose CamBase_pub;
		
		  CamBase_pub.position.x = TRC(0,3);
		  CamBase_pub.position.y = TRC(1,3);
		  CamBase_pub.position.z = TRC(2,3);
		  CamBase_pub.orientation = msgQ;
		
		  pub_cambase.publish(CamBase_pub);
		  
		}
		
		// Debugging
		//ROS_WARN_STREAM("TCE:" << endl << TCE << endl);
		//ROS_WARN_STREAM("TRE:" << endl << TRE << endl);
		//ROS_WARN_STREAM("TRC:" << endl << TRC << endl);
		
		// Execute at a fixed rate
		rate.sleep();
	
	}
	// END LOOP
	
	
  ROS_INFO_STREAM("Reached the end!" << endl);
  
}





//iri_wam_bringup 

























