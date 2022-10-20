<template>
  <el-row :gutter="20">
    <!--  det  -->
    <el-col :span="8">
      <el-card
        :body-style="{ padding: '0px' }"
        @click="openWindow('cv/detection/ScrewDetection/index.html')"
      >
        <img src="../../img/det.png" class="image" />
        <div style="padding: 14px">
          <h2>物体检测</h2>
          <div>基于检测模型，实现检测螺丝和螺母。</div>
        </div>
      </el-card>
    </el-col>
    <!--  face-det  -->
    <el-col :span="8">
      <el-card
        :body-style="{ padding: '0px' }"
        @click="openWindow('cv/detection/FaceDetection/index.html')"
      >
        <img src="../../img/facedet.png" class="image" />
        <div style="padding: 14px">
          <h2>人脸检测</h2>
          <div>基于检测模型，实现检测人脸。</div>
        </div>
      </el-card>
    </el-col>
  </el-row>
</template>

<script setup lang="ts">
import { openWindow } from "@/utils/openWindow";
</script>

<style scoped lang="less">
img {
  width: 100%;
  height: 200px;
  background-position: center center;
}
</style>
