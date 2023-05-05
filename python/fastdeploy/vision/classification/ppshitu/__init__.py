from __future__ import absolute_import
import logging
from .... import FastDeployModel, ModelFormat
from .... import c_lib_wrap as C
from ...common import ProcessorManager
from ...detection.ppdet import PicoDet


class PPShiTuV2Detector(PicoDet):
    ...
