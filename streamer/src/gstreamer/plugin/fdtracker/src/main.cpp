//Copyright (c) 2023 niuzhibo. All Rights Reserved.

#include "ocsort.h"
#include "trajectory.h"

/*
*OCSort Tracker主要使用步骤
*1. 创建类OcSortTracker *octracker = new OcSortTracker(classid);
*2. update当前帧检测框，octracker->update(dets, true, false);
*3. 获取当前跟踪的结果，cv::Mat trks = octracker->get_trackers();
*/
int main(int argc, char** argv) {
  // example1: tracker update associated to picture of object
  // 检测框定义, [class, conf, xmin, ymin, xmax, ymax]*n, 每个共6位
  float rects[]={0, 0.9, 100,100,200,200,
                0, 0.9, 300,300,400,400,
                0, 0.9, 300,300,400,400};
  int classid = 0; //类id，需要每一类单独一个OcSortTracker，否则不同类之间会匹配混乱
  OcSortTracker *octracker = new OcSortTracker(classid);

  for (int i=0;i<2;i++) {
    cv::Mat dets(4,6,CV_32FC1,rects);
    octracker->update(dets, true, false);
    printf("\n step %d trackers number :%d\n", i, octracker->trackers.size());
    //return [class, id, xmin, ymin, xmax, ymax]*n
    cv::Mat trks = octracker->get_trackers();
    printmat(trks);
  };

  float rects2[]={0, 0.9, 120,120,220,220,
                0, 0.9, 320,320,420,420,
                0, 0.2, 300,300,400,400,
                0, 0.7, 180,180,280,280};
  for (int i=0;i<1;i++) {
    cv::Mat dets(4,6,CV_32FC1,rects2);
    octracker->update(dets, true, false);
    printf("\n step %d trackers number :%d\n", i+2, octracker->trackers.size());
    cv::Mat trks = octracker->get_trackers();
    printmat(trks);
  };

  for( int i=0;i<octracker->trackers.size();i++) {
    auto history = octracker->trackers[i]->observations;
    printf("\ntracker id: %d, len: %d\n", octracker->trackers[i]->id, octracker->trackers[i]->get_length());
    for (auto pair: history) {
      printf("ageid: %d, box: %f, %f, %f, %f\n", pair.first, pair.second[0], pair.second[1], pair.second[2], pair.second[3]);
    }
  }


  //example2: kalman predict check
  // OcSortTracker *octracker = new OcSortTracker(0);
  // float rects[]={0, 0.9, 100,100,200,200,
  //               0, 0.9, 300,300,400,400,
  //               0, 0.9, 300,300,400,400};
  // cv::Mat dets(3,6,CV_32FC1,rects);
  // float vecs[]={0, 0, 10,10,10,10,
  //               0, 0, 10,10,10,10,
  //               0, 0, 10,10,10,10};
  // cv::Mat vec(3,6,CV_32FC1,vecs);
  // // octracker->update(dets, true, false);

  // for (int i=0;i<20;i++) {
  //   printf("\n\n step %d", i);
  //   cv::add(dets, vec, dets);
  //   octracker->update(dets, true, false);
  //   for (int j=0;j<octracker->trackers.size();j++) {
  //     cv::Mat pred = xyah2ltrb(octracker->trackers[j]->predict());
  //     printf("\npred:");
  //     printmat(pred);
  //   }
    
    
  //   for (int j=0;j<octracker->trackers.size();j++) {
  //     cv::Mat state = octracker->trackers[j]->get_state();
  //     printf("\n\ncorrect:");
  //     printmat(state);
  //   }
  // };

  //example3: trajectory analysis
  // OcSortTracker *octracker = new OcSortTracker(0);
  // float rects[]={0, 0.9, 100,100,200,200,
  //               0, 0.9, 300,300,400,400,
  //               0, 0.9, 300,300,400,400};
  // cv::Mat dets(3,6,CV_32FC1,rects);
  // float vecs[]={0, 0, 10,10,10,10,
  //               0, 0, 10,10,10,10,
  //               0, 0, 10,10,10,10};
  // cv::Mat vec(3,6,CV_32FC1,vecs);

  // std::vector<int> breakarea = {150, 250, 150, 300, 200, 300, 200, 250};
  // Trajectory* trajectory = new Trajectory();
  // trajectory->set_region(horizontal, std::vector<int>(2,420));
  // trajectory->set_area(breakarea);

  // for (int i=0;i<20;i++) {
  //   printf("\n step %d", i);
  //   cv::add(dets, vec, dets);
  //   octracker->update(dets, true, false);
  //   trajectory->entrance_count(octracker);
  //   auto breakinids = trajectory->breakin_count(octracker);
  //   auto count = trajectory->get_count();
  //   printf("\ncountin: %d, countout: %d", count[0], count[1]);
  //   printf("\nbreakinids: %d\n", breakinids.size());
  // }

  return 0;
}
