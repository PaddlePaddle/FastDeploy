<template>
  <el-container>
    <el-aside>
      <MenuView></MenuView>
    </el-aside>
    <el-container>
      <el-header>Header</el-header>
      <el-main>
        <RouterView></RouterView>
      </el-main>
      <el-footer>Footer</el-footer>
    </el-container>
  </el-container>
</template>

<script setup lang="ts">
import MenuView from "@/pages/main/MenuView.vue";
</script>

<style scoped lang="less">
.el-container {
  height: 100%;
  width: 100%;
}
</style>
