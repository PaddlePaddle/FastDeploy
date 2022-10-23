<template>
  <el-row :gutter="20">
    <!--  gesture  -->
    <el-col :span="8">
      <el-card
        :body-style="{ padding: '0px' }"
        @click="openWindow('cv/recognition/GestureRecognition/index.html')"
      >
        <img src="../../img/gesture.png" class="image" />
        <div style="padding: 14px">
          <h2>手势识别AI猜丁壳</h2>
          <div>
            基于手势检测&识别模型，支持在h5、小程序中识别剪刀、石头、布、"1"、"ok"等手势。
          </div>
        </div>
      </el-card>
    </el-col>
    <!--  itemrec  -->
    <el-col :span="8">
      <el-card
        :body-style="{ padding: '0px' }"
        @click="openWindow('cv/recognition/ItemIdentification/index.html')"
      >
        <img src="../../img/itemrec.jpeg" class="image" />
        <div style="padding: 14px">
          <h2>1000物品识别</h2>
          <div>基于MobileNet_V2模型，可识别常见的1000物品。</div>
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
