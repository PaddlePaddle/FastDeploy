# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: ic.proto
"""Generated protocol buffer code."""
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import symbol_database as _symbol_database
from google.protobuf.internal import builder as _builder
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n\x08ic.proto\x12\x12language_inference\"\xb5\x01\n\x17ModelFetchRequestParams\x12\x10\n\x08model_id\x18\x01 \x03(\t\x12\x17\n\x0fmax_request_num\x18\x02 \x01(\x05\x12\x33\n\x08strategy\x18\x03 \x01(\x0e\x32!.language_inference.FetchStrategy\x12:\n\x0f\x62y_token_params\x18\x04 \x01(\x0b\x32!.language_inference.ByTokenParams\"M\n\rByTokenParams\x12\x11\n\tblock_num\x18\x01 \x01(\x05\x12\x12\n\nblock_size\x18\x02 \x01(\x05\x12\x15\n\rdec_token_num\x18\x03 \x01(\x05\"R\n\x17ModelFetchRequestResult\x12\x37\n\x08requests\x18\x01 \x03(\x0b\x32%.language_inference.ModelInferRequest\"\x19\n\x17ModelSendResponseResult\"Z\n\x11ModelInferRequest\x12\x10\n\x08model_id\x18\x01 \x01(\t\x12\x12\n\nrequest_id\x18\x02 \x01(\t\x12\x10\n\x08trace_id\x18\x03 \x01(\t\x12\r\n\x05input\x18\x04 \x01(\t\"W\n\x16ModelInferResponseList\x12=\n\rresponse_list\x18\x01 \x03(\x0b\x32&.language_inference.ModelInferResponse\"M\n\x12ModelInferResponse\x12\x12\n\nrequest_id\x18\x01 \x01(\t\x12\x13\n\x0bsentence_id\x18\x02 \x01(\x05\x12\x0e\n\x06output\x18\x03 \x01(\t*+\n\rFetchStrategy\x12\r\n\tByRequest\x10\x00\x12\x0b\n\x07\x42yToken\x10\x01\x32\xd2\x03\n\x14GRPCInferenceService\x12\x65\n\x10ModelStreamInfer\x12%.language_inference.ModelInferRequest\x1a&.language_inference.ModelInferResponse\"\x00\x30\x01\x12o\n\x11ModelFetchRequest\x12+.language_inference.ModelFetchRequestParams\x1a+.language_inference.ModelFetchRequestResult\"\x00\x12l\n\x11ModelSendResponse\x12&.language_inference.ModelInferResponse\x1a+.language_inference.ModelSendResponseResult\"\x00(\x01\x12t\n\x15ModelSendResponseList\x12*.language_inference.ModelInferResponseList\x1a+.language_inference.ModelSendResponseResult\"\x00(\x01\x62\x06proto3')

_globals = globals()
_builder.BuildMessageAndEnumDescriptors(DESCRIPTOR, _globals)
_builder.BuildTopDescriptorsAndMessages(DESCRIPTOR, 'ic_pb2', _globals)
if _descriptor._USE_C_DESCRIPTORS == False:
  DESCRIPTOR._options = None
  _globals['_FETCHSTRATEGY']._serialized_start=666
  _globals['_FETCHSTRATEGY']._serialized_end=709
  _globals['_MODELFETCHREQUESTPARAMS']._serialized_start=33
  _globals['_MODELFETCHREQUESTPARAMS']._serialized_end=214
  _globals['_BYTOKENPARAMS']._serialized_start=216
  _globals['_BYTOKENPARAMS']._serialized_end=293
  _globals['_MODELFETCHREQUESTRESULT']._serialized_start=295
  _globals['_MODELFETCHREQUESTRESULT']._serialized_end=377
  _globals['_MODELSENDRESPONSERESULT']._serialized_start=379
  _globals['_MODELSENDRESPONSERESULT']._serialized_end=404
  _globals['_MODELINFERREQUEST']._serialized_start=406
  _globals['_MODELINFERREQUEST']._serialized_end=496
  _globals['_MODELINFERRESPONSELIST']._serialized_start=498
  _globals['_MODELINFERRESPONSELIST']._serialized_end=585
  _globals['_MODELINFERRESPONSE']._serialized_start=587
  _globals['_MODELINFERRESPONSE']._serialized_end=664
  _globals['_GRPCINFERENCESERVICE']._serialized_start=712
  _globals['_GRPCINFERENCESERVICE']._serialized_end=1178
# @@protoc_insertion_point(module_scope)