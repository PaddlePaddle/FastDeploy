<template>
  <el-row :gutter="20">
    <!--  human_seg  -->
    <el-col :span="8">
      <el-card
        :body-style="{ padding: '0px' }"
        @click="openWindow('cv/segmentation/HumanSeg/index.html')"
      >
        <img src="../../img/humanseg.png" class="image" />
        <div style="padding: 14px">
          <h2>人像分割</h2>
          <div>使用者可以用于背景替换、背景虚化等。</div>
        </div>
      </el-card>
    </el-col>
    <!--  human_seg_gpu  -->
    <el-col :span="8">
      <el-card
        :body-style="{ padding: '0px' }"
        @click="openWindow('cv/segmentation/HumanSeg_gpu/index.html')"
      >
        <img src="../../img/humanseg.png" class="image" />
        <div style="padding: 14px">
          <h2>人像分割（GPU）</h2>
          <div>使用者可以用于背景替换、背景虚化等。</div>
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
