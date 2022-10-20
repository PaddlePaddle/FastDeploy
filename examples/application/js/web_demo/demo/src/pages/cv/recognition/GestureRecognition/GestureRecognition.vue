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
        <h2>上传手势图片</h2>
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
        <h2>手势检测</h2>
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
        <h2>手势识别结果</h2>
      </el-row>
      <span v-html="result"></span>
    </el-col>
  </el-row>
</template>

<script setup lang="ts">
import * as gestureRec from "@paddle-js-models/gesture";
import { onMounted, onUnmounted, ref } from "vue";

const fileName = ref(null);

const canvas = ref(null as unknown as HTMLCanvasElement);

const result = ref("");

const isLoadingModel = ref(true);

onMounted(async () => {
  canvas.value = document.getElementById("canvas") as HTMLCanvasElement;

  await gestureRec.load();
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
  const res = await gestureRec.classify(img);
  console.log(res);
  if (res.box && res.box.length) {
    result.value = res.type;
    drawBox(res.box as number[][]);
  } else {
    result.value = "未识别到手";
  }
};

const drawBox = (box: number[][]) => {
  const img = document.getElementById("raw-img") as HTMLImageElement;
  canvas.value.width = img.naturalWidth;
  canvas.value.height = img.naturalHeight;
  const ctx = canvas.value.getContext("2d") as CanvasRenderingContext2D;
  ctx.drawImage(img, 0, 0, canvas.value.width, canvas.value.height);

  /**
   * 计算缩放比率，得到实际绘制框的坐标
   */
  let offsetX = 0;
  let offsetY = 0;

  if (img.width < img.height) {
    offsetX = img.height - img.width;
  }

  if (img.width > img.height) {
    offsetY = img.width - img.height;
  }

  const widthRatio = (img.width + offsetX) / 256;
  const heightRatio = (img.height + offsetY) / 256;
  const points: number[][] = [];

  box.forEach((item) => {
    const tmpPonit = [];
    tmpPonit[0] = item[0] * widthRatio - offsetX / 2;
    tmpPonit[1] = item[1] * heightRatio - offsetY / 2;
    points.push(tmpPonit);
  });

  // 开始一个新的绘制路径
  ctx.beginPath();
  // 设置线条颜色为蓝色
  ctx.strokeStyle = "blue";
  // 设置路径起点坐标
  ctx.moveTo(points[0][0], points[0][1]);
  ctx.lineTo(points[1][0], points[1][1]);
  ctx.lineTo(points[2][0], points[2][1]);
  ctx.lineTo(points[3][0], points[3][1]);
  ctx.closePath();
  ctx.stroke();
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
