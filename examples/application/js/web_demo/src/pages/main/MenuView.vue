<template>
  <el-row>
    <el-col>
      <el-menu default-active="ocr" @open="handleOpen" @close="handleClose">
        <el-sub-menu index="1">
          <template #title>
            <span>智能视觉（Paddle.js-CV）</span>
          </template>
          <el-menu-item index="det" @click="routerJump('/det')">
            <span>图像检测</span>
          </el-menu-item>
          <el-menu-item index="seg" @click="routerJump('/seg')">
            <span>图像分割</span>
          </el-menu-item>
          <el-menu-item index="rec" @click="routerJump('/rec')">
            <span>图像识别</span>
          </el-menu-item>
          <el-menu-item index="ocr" @click="routerJump('/ocr')">
            <span>OCR</span>
          </el-menu-item>
        </el-sub-menu>
      </el-menu>
    </el-col>
  </el-row>
</template>

<script setup lang="ts">
import { routerJump } from "@/pages/main/utils/routerJump";

const handleOpen = (key: string, keyPath: string[]) => {
  console.log(key, keyPath);
};
const handleClose = (key: string, keyPath: string[]) => {
  console.log(key, keyPath);
};
</script>

<style scoped lang="less"></style>
