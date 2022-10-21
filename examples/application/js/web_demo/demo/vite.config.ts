import { fileURLToPath, URL } from "node:url";
import path from "path";
import { defineConfig } from "vite";
import vue from "@vitejs/plugin-vue";
import AutoImport from "unplugin-auto-import/vite";
import Components from "unplugin-vue-components/vite";
import { ElementPlusResolver } from "unplugin-vue-components/resolvers";

// https://vitejs.dev/config/
export default defineConfig({
  root: "src/pages/",
  plugins: [
    vue(),
    AutoImport({
      resolvers: [ElementPlusResolver()],
    }),
    Components({
      resolvers: [ElementPlusResolver()],
    }),
  ],
  resolve: {
    alias: {
      "@": fileURLToPath(new URL("./src", import.meta.url)),
    },
  },
  build: {
    rollupOptions: {
      input: {
        entry: path.resolve(__dirname, "src/pages/main/index.html"),
        vis: path.resolve(__dirname, "src/pages/vis/index.html"),
        ocrdet: path.resolve(__dirname, "src/pages/cv/ocr/TextDetection/index.html"),
        ocr: path.resolve(__dirname, "src/pages/cv/ocr/TextRecognition/index.html"),
        screwdet: path.resolve(
          __dirname,
          "src/pages/cv/detection/ScrewDetection/index.html"
        ),
        facedet: path.resolve(
          __dirname,
          "src/pages/cv/detection/FaceDetection/index.html"
        ),
        gesturerec: path.resolve(
          __dirname,
          "src/pages/cv/recognition/GestureRecognition/index.html"
        ),
        humanseg: path.resolve(
          __dirname,
          "src/pages/cv/segmentation/HumanSeg/index.html"
        ),
        humanseg_gpu: path.resolve(
          __dirname,
          "src/pages/cv/segmentation/HumanSeg_gpu/index.html"
        ),
        itemrec: path.resolve(
          __dirname,
          "src/pages/cv/recognition/ItemIdentification/index.html"
        ),
      },
      output: {
        chunkFileNames: "static/js/[name]-[hash].js",
        entryFileNames: "static/js/[name]-[hash].js",
        assetFileNames: "static/[ext]/name-[hash].[ext]",
      },
    },
  },
});
