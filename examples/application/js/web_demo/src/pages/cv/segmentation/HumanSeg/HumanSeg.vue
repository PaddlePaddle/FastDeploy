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
        <h2>上传图片</h2>
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
    <el-col :span="4">
      <el-row class="small-title">
        <h2>上传替换背景图片</h2>
      </el-row>
      <el-row>
        <el-input
          type="file"
          v-model="newBackgroundImgFileName"
          @change="uploadNewBackgroundImg"
        ></el-input>
      </el-row>
      <el-row>
        <!-- 用于展示图片 -->
        <img id="new-background-img" class="show-area" />
      </el-row>
      <el-row>
        <canvas id="background" class="show-area"></canvas>
      </el-row>
    </el-col>
    <el-col :span="4">
      <el-row class="small-title">
        <h2>背景替换</h2>
      </el-row>
      <el-row>
        <el-button type="primary" @click="backgroundReplace">
          替换背景
        </el-button>
      </el-row>
      <el-row>
        <canvas id="replace-background" class="show-area"></canvas>
      </el-row>
    </el-col>
    <el-col :span="4">
      <el-row class="small-title">
        <h2>背景虚化</h2>
      </el-row>
      <el-row>
        <el-button type="primary" @click="blurBackground">背景虚化</el-button>
      </el-row>
      <el-row>
        <canvas id="blur" class="show-area"></canvas>
      </el-row>
    </el-col>
    <el-col :span="4">
      <el-row class="small-title">
        <h2>人形遮罩</h2>
      </el-row>
      <el-row>
        <el-button type="primary" @click="drawHumanoidMask">
          绘制人形遮罩
        </el-button>
      </el-row>
      <el-row>
        <canvas id="mask" class="show-area"></canvas>
      </el-row>
    </el-col>
  </el-row>
</template>

<script setup lang="ts">
import * as humanSeg from "@paddle-js-models/humanseg";
import { onMounted, ref } from "vue";

const fileName = ref(null);
const newBackgroundImgFileName = ref(null);

const backgroundCanvas = ref(null as unknown as HTMLCanvasElement);
const replaceBackgroundCanvas = ref(null as unknown as HTMLCanvasElement);
const blurCanvas = ref(null as unknown as HTMLCanvasElement);
const maskCanvas = ref(null as unknown as HTMLCanvasElement);

const isLoadingModel = ref(true);
const modelPredictDone = ref(false);

/**
 * 存储模型分割后的像素 alpha 值
 */
let garyData: number[];

onMounted(async () => {
  backgroundCanvas.value = document.getElementById(
    "background"
  ) as HTMLCanvasElement;
  replaceBackgroundCanvas.value = document.getElementById(
    "replace-background"
  ) as HTMLCanvasElement;
  blurCanvas.value = document.getElementById("blur") as HTMLCanvasElement;
  maskCanvas.value = document.getElementById("mask") as HTMLCanvasElement;
  await humanSeg.load();
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

const uploadNewBackgroundImg = () => {
  /**
   * 这里由于操作是绑定在 el-input 上；因此需要在内部重新获取 input 再拿到 file
   */
  const reader = new FileReader();
  // 用于展示
  const showImg = document.getElementById(
    "new-background-img"
  ) as HTMLImageElement;
  // 获取背景图片的 input
  const inputElement = document
    .getElementsByClassName("el-input")[1]
    .getElementsByTagName("input")[0];

  try {
    const file = inputElement.files![0];
    reader.onload = () => {
      showImg.src = URL.createObjectURL(file);
    };
    reader.readAsDataURL(file);
  } catch (err) {
    console.error(err);
  }
};

const backgroundReplace = async () => {
  if (!modelPredictDone.value) {
    await seg();
    modelPredictDone.value = true;
  }

  const showImg = document.getElementById(
    "new-background-img"
  ) as HTMLImageElement;
  backgroundCanvas.value.width = showImg.naturalWidth;
  backgroundCanvas.value.height = showImg.naturalHeight;
  backgroundCanvas.value
    .getContext("2d")!
    .drawImage(showImg, 0, 0, showImg.naturalWidth, showImg.naturalHeight);
  humanSeg.drawHumanSeg(
    garyData,
    replaceBackgroundCanvas.value,
    backgroundCanvas.value
  );
};

const blurBackground = async () => {
  if (!modelPredictDone.value) {
    await seg();
    modelPredictDone.value = true;
  }
  humanSeg.blurBackground(garyData, blurCanvas.value);
};

const drawHumanoidMask = async () => {
  if (!modelPredictDone.value) {
    await seg();
    modelPredictDone.value = true;
  }
  humanSeg.drawMask(garyData, maskCanvas.value, backgroundCanvas.value);
};

const seg = async () => {
  const img = document.getElementById("raw-img") as HTMLImageElement;
  const res = await humanSeg.getGrayValue(img);
  console.log(res);
  garyData = res.data;
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
