<script setup lang="ts">
import ScrewDetection from "@/pages/cv/detection/ScrewDetection/ScrewDetection.vue";
</script>

<template>
  <el-container>
    <el-header>Header</el-header>
    <el-main>
      <ScrewDetection></ScrewDetection>
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
