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
    <el-col :span="12">
      <el-row class="small-title">
        <h2>上传人脸图片</h2>
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
    <el-col :span="12">
      <el-row class="small-title">
        <h2>人脸区域检测</h2>
      </el-row>
      <el-row>
        <el-button type="primary" @click="predict">开始检测</el-button>
      </el-row>
      <el-row>
        <canvas id="canvas" class="show-area"></canvas>
      </el-row>
    </el-col>
  </el-row>
</template>

<script setup lang="ts">
import { FaceDetector } from "@paddle-js-models/facedetect";
import { onMounted, ref } from "vue";

interface TransformData {
  left: number;
  width: number;
  top: number;
  height: number;
  confidence: number;
}

const faceDet = new FaceDetector();

const fileName = ref(null);

const canvas = ref(null as unknown as HTMLCanvasElement);

const isLoadingModel = ref(true);

onMounted(async () => {
  canvas.value = document.getElementById("canvas") as HTMLCanvasElement;
  await faceDet.init();
  isLoadingModel.value = false;
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
  const res = await faceDet.detect(img);
  console.log(res);
  drawBox(res);
};

const drawBox = (data: TransformData[]) => {
  const img = document.getElementById("raw-img") as HTMLImageElement;
  const imgHeight = img.height;
  const imgWidth = img.width;
  canvas.value.width = imgWidth;
  canvas.value.height = imgHeight;
  const ctx = canvas.value.getContext("2d") as CanvasRenderingContext2D;
  ctx.drawImage(img, 0, 0, canvas.value.width, canvas.value.height);
  data.forEach((item: TransformData) => {
    // 开始一个新的绘制路径
    ctx.beginPath();
    // 设置线条颜色为蓝色
    ctx.strokeStyle = "red";
    // 获取检测框选坐标
    const x = item.left * imgWidth;
    const y = item.top * imgHeight;
    const w = item.width * imgWidth;
    const h = item.height * imgHeight;
    ctx.beginPath();
    // 绘制检测框选矩形
    ctx.rect(x, y, w, h);
    // 绘制label
    ctx.fillText(item.confidence.toFixed(6), x, y);
    ctx.stroke();
  });
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
