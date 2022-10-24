#!/usr/bin/env python
# -*- coding: UTF-8 -*-


# pruning op tensor and relatad op with no sense, like ShapeTensor and OutSize
def pruningNoSenseTensor(model):
   global ops
   ops = model["ops"]
   global vars
   vars = model["vars"]
   for op in ops[:]:
      shapeTensor = op["inputs"].get("ShapeTensor")
      outSizeTensor = op["inputs"].get("OutSize")

      noSenseTensor = shapeTensor or outSizeTensor
      if not noSenseTensor:
         continue
      
      print(noSenseTensor)
      if shapeTensor:
         del op["inputs"]["ShapeTensor"]
      if outSizeTensor:
         del op["inputs"]["OutSize"]

      for tensorId in noSenseTensor:
         delLeafOpWithoutChildren(tensorId)


# delete leaf op which has no child
def delLeafOpWithoutChildren(tensorId):
   # judge if there is an op which used the tensor
   for op in ops[:]:
      inputsTensor = op["inputs"]
      input = inputsTensor.get("Input") or inputsTensor.get("X")
      
      if input and (tensorId in input):
         return

   op = getOpByOutputTensor(tensorId)
   if not op:
      return

   # del op
   ops.remove(op)
   # del vars
   del vars[tensorId]

   # upward recursion
   delOpinputsTensor = op["inputs"]
   input = delOpinputsTensor.get("Input") or delOpinputsTensor.get("X")
   if not input:
      return
   for inputTensorId in input:
      delLeafOpWithoutChildren(inputTensorId)
   
   
         
# find op by output tensor id
def getOpByOutputTensor(tensorId):
   for op in ops[:]:
      outputTensor = op["outputs"]
      out = outputTensor.get("Out") or outputTensor.get("Output") or outputTensor.get("Y")
      if out[0] == tensorId:
         return op
