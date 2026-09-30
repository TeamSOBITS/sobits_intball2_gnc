#include "gcopter/geo_utils.hpp"
#include "gcopter/gcopter.hpp"
#include "gcopter/trajectory.hpp"
#include <Eigen/Eigen>
#include <iostream>
#include <cmath>
int main(){using G=gcopter::GCOPTER_PolytopeSFC; G::PolyhedraH c; for(int i=0;i<3;i++){Eigen::Matrix<double,6,4> h=Eigen::Matrix<double,6,4>::Zero(); h(0,i)=1;h(0,3)=-10;h(1,i)=-1;h(1,3)=8;h(2,(i+1)%3)=1;h(2,3)=-1;h(3,(i+1)%3)=-1;h(3,3)=-1;h(4,(i+2)%3)=1;h(4,3)=-6;h(5,(i+2)%3)=-1;h(5,3)=4;c.push_back(h);} Eigen::Matrix3d a=Eigen::Matrix3d::Zero(),b=a;a.col(0)<<8.5,0,5;b.col(0)<<9.5,0,5;Eigen::VectorXd m(5),w(5),p(6);m<<1,1,1,0,100;w<<1,1,1,1,1;p<<1,9.81,0,0,0,.1;G g;if(!g.setup(1,a,b,c,INFINITY,1e-3,8,m,w,p))return 1;Trajectory<5> tr;double cost=g.optimize(tr,1e-4);std::cout<<"pieces="<<tr.getPieceNum()<<" cost="<<cost<<"\n";return std::isfinite(cost)?0:2;}
