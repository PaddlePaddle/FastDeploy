<template>
  <div id="container"></div>
</template>

<script setup lang="ts">
import { Runner } from "@paddlejs/paddlejs-core";
import "@paddlejs/paddlejs-backend-webgl";
import { Graph } from '@antv/x6';
import {onMounted, ref} from "vue";

const data: {nodes: any[], edges: {}[]} = { nodes: [], edges: [] };
const size = ref(0);

onMounted(async () => {
  const detectRunner = new Runner({
    modelPath: 'https://paddlejs.bj.bcebos.com/models/fuse/detect/detect_fuse_activation/model.json',
    fill: '#fff',
    mean: [0.5, 0.5, 0.5],
    std: [0.5, 0.5, 0.5],
    bgr: true,
    keepRatio: false,
    webglFeedProcess: true
  });
  await detectRunner.init();
  console.log(detectRunner.model);
  console.log(detectRunner.weightMap);
  genData(detectRunner.weightMap);

  console.log("data", data);

  const graph = new Graph({
    container: document.getElementById('container')!,
    width: size.value,
    height: size.value,
  });
  graph.fromJSON(data);
})

const genData = (weightMap: any[]) => {
  const op2var: { [key: string]: string[] } = {};
  const var2op: { [key: string]: string[] } = {};
  const op2idx: { [key: string]: number } = {};

  weightMap.forEach((op, index) => {
    size.value += 80;
    op2idx[op.id] = index;
    data.nodes.push({
      id: op.id,
      x: 40,
      y: 40 * index + 40,
      width: 100,
      height: 30,
      label: op.id
    });

    const outputs = op.outputs;
    for (const out in outputs) {
      outputs[out].forEach((var_: any) => {
        if (!op2var[op.id]) {
          op2var[op.id] = [];
        }
        if (!var_.name) {
          op2var[op.id].push(var_);
        } else {
          op2var[op.id].push(var_.name);
        }
      })
    }

    const inputs = op.inputs;
    for (const input in inputs) {
      inputs[input].forEach((var_: any) => {
        if (!var_.name) {
          if (!var2op[var_]) {
            var2op[var_] = []
          }
          var2op[var_].push(op.id);
        } else {
          if (!var2op[var_.name]) {
            var2op[var_.name] = []
          }
          var2op[var_.name].push(op.id);
        }
      })
    }
  })

  console.log("op2var", op2var);
  console.log("var2op", var2op);

  for (const op in op2var) {
    op2var[op].forEach((var_) => {
      if (var2op[var_]) {
        var2op[var_].forEach((targetOp, index) => {
          data.edges.push({
            source: op, // String，必须，起始节点 id
            target: targetOp, // String，必须，目标节点 id
          });
          // if (index) {
          //   data.nodes[op2idx[targetOp]].x = data.nodes[op2idx[var2op[var_][0]]].x + 40 * (index - 1);
          //   data.nodes[op2idx[targetOp]].y = data.nodes[op2idx[var2op[var_][0]]].y;
          // }
        })
      }
    });
  }
}


</script>

<style scoped lang="less">
</style>
