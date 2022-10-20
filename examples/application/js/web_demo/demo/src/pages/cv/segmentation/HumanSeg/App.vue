<script setup lang="ts">
import HumanSeg from "@/pages/cv/segmentation/HumanSeg/HumanSeg.vue";
</script>

<template>
  <el-container>
    <el-header>Header</el-header>
    <el-main>
      <HumanSeg></HumanSeg>
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
