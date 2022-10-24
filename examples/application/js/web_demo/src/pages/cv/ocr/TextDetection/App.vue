<script setup lang="ts">
import TextDetection from "@/pages/cv/ocr/TextDetection/TextDetection.vue";
</script>

<template>
  <el-container>
    <el-header>Header</el-header>
    <el-main>
      <TextDetection></TextDetection>
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
