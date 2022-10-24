<script setup lang="ts">
import TextRecognition from "@/pages/cv/ocr/TextRecognition/TextRecognition.vue";
</script>

<template>
  <el-container>
    <el-header>Header</el-header>
    <el-main>
      <TextRecognition></TextRecognition>
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
