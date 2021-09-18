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
#include <mpc_vision/SOMstate.h>
//#include <std_srvs/Empty.h>


using namespace std;


// Robot-Camera Transform
Eigen::MatrixXf TRC = Eigen::MatrixXf::Zero(4,4);
// SOM State Vector: initialization of required vars
int seqi = 0;
int nSOM = 0;
double tprev = 0.0;
ros::Duration vDelay;
ros::Duration vfDelay;
Eigen::VectorXf SOMpos;
Eigen::VectorXf SOMposprev;
Eigen::VectorXf SOMvel;
Eigen::VectorXf SOMstate;
Eigen::VectorXf SOMstate_prev1;
Eigen::VectorXf SOMstate_prev2;
Eigen::VectorXf SOMstate_fltr;
double wFltr = 0.6;


// ROS SUBSCRIBERS - CALLBACK FUNCTIONS
void meshReceived(const mpc_vision::ClothMesh& msg) {
  
  // Extract data
  int nMesh = msg.size;
  double dt = msg.header.stamp.toSec() - tprev;
  seqi = msg.header.seq;
  //ROS_WARN_STREAM(msg.header.stamp.toSec() <<" - "<< tprev <<" = "<< dt);
  
  // Warn if size changed, update variable and resize
  if (nMesh != nSOM) {
  	ROS_WARN_STREAM("Cloth Mesh size changed from "<<nSOM<<" to "<<nMesh<<". Resizing...");
  	
  	nSOM = nMesh;
  	
  	SOMpos.resize(3*nSOM*nSOM);
		SOMpos.setZero(3*nSOM*nSOM);
		SOMposprev.resize(3*nSOM*nSOM);
		SOMposprev.setZero(3*nSOM*nSOM);
		SOMvel.resize(3*nSOM*nSOM);
		SOMvel.setZero(3*nSOM*nSOM);
		SOMstate.resize(6*nSOM*nSOM);
		SOMstate.setZero(6*nSOM*nSOM);
		SOMstate_prev1.resize(6*nSOM*nSOM);
		SOMstate_prev1.setZero(6*nSOM*nSOM);
		SOMstate_prev2.resize(6*nSOM*nSOM);
		SOMstate_prev2.setZero(6*nSOM*nSOM);
		SOMstate_fltr.resize(6*nSOM*nSOM);
		SOMstate_fltr.setZero(6*nSOM*nSOM);
  }
  
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
	
	// Apply Transform to all points
	Eigen::VectorXf xSOM = Eigen::VectorXf::Zero(nSOM*nSOM);
	Eigen::VectorXf ySOM = Eigen::VectorXf::Zero(nSOM*nSOM);
	Eigen::VectorXf zSOM = Eigen::VectorXf::Zero(nSOM*nSOM);
	
	for (int pti=0; pti<(nSOM*nSOM); pti++) {
	
		// Filter NaNs in depth (z)
		if (isnan(zCAMord(pti))) {
			double zSum = 0;
			int zN = 0;

			if (pti/nSOM > 0 && !isnan(zCAMord(pti-nSOM))) {
				zSum += zCAMord(pti-nSOM);
				zN++;
			}
			if (pti/nSOM < nSOM-1 && !isnan(zCAMord(pti+nSOM))) {
				zSum += zCAMord(pti+nSOM);
				zN++;
			}
			if (pti%nSOM > 0 && !isnan(zCAMord(pti-1))) {
				zSum += zCAMord(pti-1);
				zN++;
			}
			if (pti%nSOM < nSOM-1 && !isnan(zCAMord(pti+1))) {
				zSum += zCAMord(pti+1);
				zN++;
			}
			
			// Change NaN for mean of non-NaN neighbors
			zCAMord(pti) = zSum/zN;
			
		}
		
		// Create [x,y,z,1] vector
		Eigen::VectorXf Pt(4); 
		Pt << xCAMord(pti), yCAMord(pti), zCAMord(pti), 1.0;
		
		// Transform to robot base	
		Pt = (TRC*Pt).eval();
		
		xSOM(pti) = Pt(0);
		ySOM(pti) = Pt(1);
		zSOM(pti) = Pt(2);
		
	}
	
	// Debugging
	/*
	ROS_WARN_STREAM("Cloth center: ["<<
									xSOM.sum()/xSOM.size()<<", "<<
									ySOM.sum()/ySOM.size()<<", "<<
									zSOM.sum()/zSOM.size()<<"]");
	*/
	/*
	ROS_WARN_STREAM("Cloth mid-top: ["<<
									xSOM.tail(nSOM).sum()/nSOM<<", "<<
									ySOM.tail(nSOM).sum()/nSOM<<", "<<
									zSOM.tail(nSOM).sum()/nSOM<<"]");
  */							
	
	SOMpos << xSOM, ySOM, zSOM;
	SOMvel << (SOMpos - SOMposprev)/dt;
	SOMstate << SOMpos, SOMvel;
	
	SOMposprev = SOMpos;
	SOMstate_prev2 = SOMstate_prev1;
	SOMstate_prev1 = SOMstate;
	SOMstate_fltr = (SOMstate_prev2*wFltr + SOMstate_prev1 + SOMstate*wFltr)/(1+2*wFltr);
	vfDelay = ros::Time::now() - msg.header.stamp + ros::Duration(dt);
	tprev = msg.header.stamp.toSec();
	vDelay = ros::Time::now() - msg.header.stamp;

}


void trcReceived(const geometry_msgs::Pose& msg) {

  // Set position
  TRC(0,3) = msg.position.x;
  TRC(1,3) = msg.position.y;
  TRC(2,3) = msg.position.z;
  
  // Set orientation
  geometry_msgs::Quaternion Rquat = msg.orientation;
  
  Eigen::Quaterniond ERQuat;
  Eigen::Vector4d CAMori;
  CAMori << Rquat.x, Rquat.y, Rquat.z, Rquat.w;
  ERQuat = CAMori;
  
  Eigen::Matrix3d TRCrot = ERQuat.normalized().toRotationMatrix();
  TRC.block(0,0, 3,3) = TRCrot.cast<float>();
  
  // Set scale
  TRC(3,3) = 1;
  
}




// ---------------------------
// -------- MAIN PROG --------
// ---------------------------

int main(int argc, char **argv) {

	// Initialize the ROS system and become a node.
  ros::init(argc, argv, "update_som");
  ros::NodeHandle rosnh;
  
	// Define client objects to all services
  //ros::ServiceClient clt_foo = rosnh.serviceClient<node::Service>("service_name");
  //ros::service::waitForService("service_name");
  
  // Define Publishers
  ros::Publisher pub_somstate = rosnh.advertise<mpc_vision::SOMstate>
                                ("mpc_controller/state_SOM", 1000);
  
  // Define Subscribers
  // cloth_mesh or cloth_mesh_filtered
  ros::Subscriber sub_mesh = rosnh.subscribe("/cloth_segmentation/cloth_mesh_filtered",
                                             1000, &meshReceived);
  ros::Subscriber sub_trc = rosnh.subscribe("/mpc_controller/camera_pose",
                                             1000, &trcReceived);
	
  // Initial Transform value, used if no calibration launched
  TRC << 1.0,  0.0,  0.0,  0.1,
         0.0,  0.0,  1.0, -1.4,
         0.0, -1.0,  0.0,  0.2,
         0.0,  0.0,  0.0,  1.0;
	
	// START LOOP	
	ros::Rate rate(20);
	while(rosnh.ok()) {
	
		// Check subscriptions
		ros::spinOnce();
		
		// Publish SOM state vector
		mpc_vision::SOMstate SOMstate_pub;
		SOMstate_pub.header.seq = seqi;
		SOMstate_pub.header.stamp = ros::Time::now() - vDelay; //vfDelay;
		SOMstate_pub.header.frame_id = "iri_wam_link_base";
		SOMstate_pub.size = nSOM;
		SOMstate_pub.states = vector<float>(SOMstate.data(), SOMstate.size()+SOMstate.data());
		
		/*
		SOMstate_pub.states = vector<float>(SOMstate_fltr.data(),
		                                    SOMstate_fltr.size()+SOMstate_fltr.data());
		*/
		
		// Publish if it has no NaNs and size is not 0
		if (!SOMstate.hasNaN() && nSOM != 0) {
			pub_somstate.publish(SOMstate_pub);
		}	
		
		// Debugging
		/*
		if (nSOM != 0) {
  		ROS_WARN_STREAM(SOMstate_fltr(0)<<", "<<SOMstate_fltr(nSOM-1));
	  	ROS_WARN_STREAM("Delay: "<<vfDelay);
	  }
	  */
	  
		// Execute at a fixed rate
		rate.sleep();
	
	}
	// END LOOP
	
	
  ROS_INFO_STREAM("Reached the end!" << endl);
  
}































