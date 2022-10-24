<template>
  <el-dialog
    v-model="isLoadingModel"
    title="提示"
    width="30%"
    center
    :lock-scroll="true"
    :show-close="false"
    :close-on-click-modal="false"
    :close-on-press-escape="false"
  >
    <span>正在加载模型，请稍等。</span>
  </el-dialog>
  <el-row :gutter="20">
    <el-col :span="8">
      <el-row class="small-title">
        <h2>上传文本图片</h2>
      </el-row>
      <el-row>
        <el-input type="file" v-model="fileName" @change="uploadImg"></el-input>
      </el-row>
      <el-row>
        <!-- 用于展示图片 -->
        <img id="show-img" class="show-area" />
        <!-- 用于存放真实图片进行文字识别 -->
        <img id="raw-img" style="display: none" />
      </el-row>
    </el-col>
    <el-col :span="8">
      <el-row class="small-title">
        <h2>文字区域检测</h2>
      </el-row>
      <el-row>
        <el-button type="primary" @click="predict">开始识别</el-button>
      </el-row>
      <el-row>
        <canvas id="canvas" class="show-area"></canvas>
      </el-row>
    </el-col>
    <el-col :span="8">
      <el-row class="small-title">
        <h2>识别结果展示</h2>
      </el-row>
      <span v-html="result"></span>
    </el-col>
  </el-row>
</template>

<script setup lang="ts">
import * as ocr from "@paddle-js-models/ocr";
import { onMounted, onUnmounted, ref } from "vue";

const fileName = ref(null);

const canvas = ref(null as unknown as HTMLCanvasElement);

const result = ref("");

const isLoadingModel = ref(true);

onMounted(async () => {
  canvas.value = document.getElementById("canvas") as HTMLCanvasElement;

  await ocr.init();
  isLoadingModel.value = false;
});

onUnmounted(() => {
  console.log("delete rec");
});

const uploadImg = () => {
  /**
   * 这里由于操作是绑定在 el-input 上；因此需要在内部重新获取 input 再拿到 file
   */
  const reader = new FileReader();
  // 用于展示
  const showImg = document.getElementById("show-img") as HTMLImageElement;
  // 用于识别
  const rawImg = document.getElementById("raw-img") as HTMLImageElement;
  const inputElement = document
    .getElementsByClassName("el-input")[0]
    .getElementsByTagName("input")[0];

  try {
    const file = inputElement.files![0];
    reader.onload = () => {
      showImg.src = URL.createObjectURL(file);
      rawImg.src = URL.createObjectURL(file);
    };
    reader.readAsDataURL(file);
  } catch (err) {
    console.error(err);
  }
};

const predict = async () => {
  const img = document.getElementById("raw-img") as HTMLImageElement;
  const res = await ocr.recognize(img, { canvas: canvas.value });
  console.log(res);
  if (res.text?.length) {
    // 页面展示识别内容
    result.value = res.text.reduce((total, cur) => total + `<p>${cur}</p>`);
  }
};
</script>

<style scoped lang="less">
.small-title {
  justify-content: space-between;
  align-items: center;
}

.show-area {
  width: 100%;
}

.el-row {
  margin-bottom: 20px;
}
</style>
