<script setup lang="ts">
import GestureRecognition from "@/pages/cv/recognition/GestureRecognition/GestureRecognition.vue";
</script>

<template>
  <el-container>
    <el-header>Header</el-header>
    <el-main>
      <GestureRecognition></GestureRecognition>
    </el-main>
    <el-footer>Footer</el-footer>
  </el-container>
</template>

<style scoped lang="less">
.el-container {
  height: 100%;
  width: 100%;
}
</style>
