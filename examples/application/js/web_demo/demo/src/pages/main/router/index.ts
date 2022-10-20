import { createRouter, createWebHashHistory } from "vue-router";
import DetectionIntroduction from "@/pages/main/views/detection/DetectionIntroduction.vue";
import OcrIntroduction from "@/pages/main/views/ocr/OcrIntroduction.vue";
import RecognitionIntroduction from "@/pages/main/views/recognition/RecognitionIntroduction.vue";
import SegmentationView from "@/pages/main/views/segmentation/SegmentationView.vue";

const router = createRouter({
  history: createWebHashHistory(import.meta.env.BASE_URL),
  routes: [
    {
      path: "/",
      name: "main",
      component: OcrIntroduction,
    },
    {
      path: "/det",
      name: "det",
      component: DetectionIntroduction,
    },
    {
      path: "/rec",
      name: "rec",
      component: RecognitionIntroduction,
    },
    {
      path: "/seg",
      name: "seg",
      component: SegmentationView,
    },
    {
      path: "/ocr",
      name: "ocr",
      component: OcrIntroduction,
    },
  ],
});

export default router;
