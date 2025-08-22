
#include "../machete_mm_launcher.cuh"

namespace machete {



extern paddle::Tensor impl_f16u4b8f16voidvoidvoidf16f32_sch_128x128_2x1x1_TmaMI__TmaCoop_streamK(MMArgs);
extern paddle::Tensor impl_f16u4b8f16voidvoidvoidf16f32_sch_128x256_2x1x1_TmaMI__TmaCoop_streamK(MMArgs);
extern paddle::Tensor impl_f16u4b8f16voidvoidvoidf16f32_sch_128x64_2x1x1_TmaMI__TmaCoop_streamK(MMArgs);
extern paddle::Tensor impl_f16u4b8f16voidvoidvoidf16f32_sch_128x32_2x1x1_TmaMI__TmaCoop_streamK(MMArgs);
extern paddle::Tensor impl_f16u4b8f16voidvoidvoidf16f32_sch_256x128_2x1x1_TmaMI__TmaCoop_streamK(MMArgs);
extern paddle::Tensor impl_f16u4b8f16voidvoidvoidf16f32_sch_128x16_1x1x1_TmaMI__TmaCoop_streamK(MMArgs);
extern paddle::Tensor impl_f16u4b8f16voidvoidvoidf16f32_sch_256x64_2x1x1_TmaMI__TmaCoop_streamK(MMArgs);
extern paddle::Tensor impl_f16u4b8f16voidvoidvoidf16f32_sch_256x32_2x1x1_TmaMI__TmaCoop_streamK(MMArgs);
extern paddle::Tensor impl_f16u4b8f16voidvoidvoidf16f32_sch_256x16_1x1x1_TmaMI__TmaCoop_streamK(MMArgs);

paddle::Tensor mm_dispatch_f16u4b8f16voidvoidvoidf16f32(MMArgs args) {
  [[maybe_unused]] auto M = args.A.shape()[0];
  [[maybe_unused]] auto N = args.B.shape()[1];
  [[maybe_unused]] auto K = args.A.shape()[1];

  if (!args.maybe_schedule) {
    if (M > 256 && K <= 16384 && N <= 4096)
        return impl_f16u4b8f16voidvoidvoidf16f32_sch_128x128_2x1x1_TmaMI__TmaCoop_streamK(args);
    if (M > 256)
        return impl_f16u4b8f16voidvoidvoidf16f32_sch_128x256_2x1x1_TmaMI__TmaCoop_streamK(args);
    if (M > 128 && K <= 4096 && N <= 4096)
        return impl_f16u4b8f16voidvoidvoidf16f32_sch_128x64_2x1x1_TmaMI__TmaCoop_streamK(args);
    if (M > 128 && K <= 8192 && N <= 8192)
        return impl_f16u4b8f16voidvoidvoidf16f32_sch_128x128_2x1x1_TmaMI__TmaCoop_streamK(args);
    if (M > 128)
        return impl_f16u4b8f16voidvoidvoidf16f32_sch_128x256_2x1x1_TmaMI__TmaCoop_streamK(args);
    if (M > 64 && K <= 4069 && N <= 4069)
        return impl_f16u4b8f16voidvoidvoidf16f32_sch_128x32_2x1x1_TmaMI__TmaCoop_streamK(args);
    if (M > 64 && K <= 4069 && N <= 8192)
        return impl_f16u4b8f16voidvoidvoidf16f32_sch_128x64_2x1x1_TmaMI__TmaCoop_streamK(args);
    if (M > 64 && K >= 8192 && N >= 12288)
        return impl_f16u4b8f16voidvoidvoidf16f32_sch_256x128_2x1x1_TmaMI__TmaCoop_streamK(args);
    if (M > 64)
        return impl_f16u4b8f16voidvoidvoidf16f32_sch_128x128_2x1x1_TmaMI__TmaCoop_streamK(args);
    if (M > 32 && K <= 6144 && N <= 6144)
        return impl_f16u4b8f16voidvoidvoidf16f32_sch_128x16_1x1x1_TmaMI__TmaCoop_streamK(args);
    if (M > 32 && K >= 16384 && N >= 12288)
        return impl_f16u4b8f16voidvoidvoidf16f32_sch_256x64_2x1x1_TmaMI__TmaCoop_streamK(args);
    if (M > 32)
        return impl_f16u4b8f16voidvoidvoidf16f32_sch_128x64_2x1x1_TmaMI__TmaCoop_streamK(args);
    if (M > 16 && K <= 12288 && N <= 8192)
        return impl_f16u4b8f16voidvoidvoidf16f32_sch_128x32_2x1x1_TmaMI__TmaCoop_streamK(args);
    if (M > 16)
        return impl_f16u4b8f16voidvoidvoidf16f32_sch_256x32_2x1x1_TmaMI__TmaCoop_streamK(args);
    if (N >= 26624)
        return impl_f16u4b8f16voidvoidvoidf16f32_sch_256x16_1x1x1_TmaMI__TmaCoop_streamK(args);
    else
        return impl_f16u4b8f16voidvoidvoidf16f32_sch_128x16_1x1x1_TmaMI__TmaCoop_streamK(args);
  }
  if (*args.maybe_schedule == "128x128_2x1x1_TmaMI__TmaCoop_streamK")
    return impl_f16u4b8f16voidvoidvoidf16f32_sch_128x128_2x1x1_TmaMI__TmaCoop_streamK(args);
  if (*args.maybe_schedule == "128x256_2x1x1_TmaMI__TmaCoop_streamK")
    return impl_f16u4b8f16voidvoidvoidf16f32_sch_128x256_2x1x1_TmaMI__TmaCoop_streamK(args);
  if (*args.maybe_schedule == "128x64_2x1x1_TmaMI__TmaCoop_streamK")
    return impl_f16u4b8f16voidvoidvoidf16f32_sch_128x64_2x1x1_TmaMI__TmaCoop_streamK(args);
  if (*args.maybe_schedule == "128x32_2x1x1_TmaMI__TmaCoop_streamK")
    return impl_f16u4b8f16voidvoidvoidf16f32_sch_128x32_2x1x1_TmaMI__TmaCoop_streamK(args);
  if (*args.maybe_schedule == "256x128_2x1x1_TmaMI__TmaCoop_streamK")
    return impl_f16u4b8f16voidvoidvoidf16f32_sch_256x128_2x1x1_TmaMI__TmaCoop_streamK(args);
  if (*args.maybe_schedule == "128x16_1x1x1_TmaMI__TmaCoop_streamK")
    return impl_f16u4b8f16voidvoidvoidf16f32_sch_128x16_1x1x1_TmaMI__TmaCoop_streamK(args);
  if (*args.maybe_schedule == "256x64_2x1x1_TmaMI__TmaCoop_streamK")
    return impl_f16u4b8f16voidvoidvoidf16f32_sch_256x64_2x1x1_TmaMI__TmaCoop_streamK(args);
  if (*args.maybe_schedule == "256x32_2x1x1_TmaMI__TmaCoop_streamK")
    return impl_f16u4b8f16voidvoidvoidf16f32_sch_256x32_2x1x1_TmaMI__TmaCoop_streamK(args);
  if (*args.maybe_schedule == "256x16_1x1x1_TmaMI__TmaCoop_streamK")
    return impl_f16u4b8f16voidvoidvoidf16f32_sch_256x16_1x1x1_TmaMI__TmaCoop_streamK(args);
  PADDLE_ENFORCE(false, "machete_gemm(..) is not implemented for "
                                     "schedule = %s", *args.maybe_schedule);
}

extern paddle::Tensor impl_bf16u4b8bf16voidvoidvoidbf16f32_sch_128x128_2x1x1_TmaMI__TmaCoop_streamK(MMArgs);
extern paddle::Tensor impl_bf16u4b8bf16voidvoidvoidbf16f32_sch_128x256_2x1x1_TmaMI__TmaCoop_streamK(MMArgs);
extern paddle::Tensor impl_bf16u4b8bf16voidvoidvoidbf16f32_sch_128x64_2x1x1_TmaMI__TmaCoop_streamK(MMArgs);
extern paddle::Tensor impl_bf16u4b8bf16voidvoidvoidbf16f32_sch_128x32_2x1x1_TmaMI__TmaCoop_streamK(MMArgs);
extern paddle::Tensor impl_bf16u4b8bf16voidvoidvoidbf16f32_sch_256x128_2x1x1_TmaMI__TmaCoop_streamK(MMArgs);
extern paddle::Tensor impl_bf16u4b8bf16voidvoidvoidbf16f32_sch_128x16_1x1x1_TmaMI__TmaCoop_streamK(MMArgs);
extern paddle::Tensor impl_bf16u4b8bf16voidvoidvoidbf16f32_sch_256x64_2x1x1_TmaMI__TmaCoop_streamK(MMArgs);
extern paddle::Tensor impl_bf16u4b8bf16voidvoidvoidbf16f32_sch_256x32_2x1x1_TmaMI__TmaCoop_streamK(MMArgs);
extern paddle::Tensor impl_bf16u4b8bf16voidvoidvoidbf16f32_sch_256x16_1x1x1_TmaMI__TmaCoop_streamK(MMArgs);

paddle::Tensor mm_dispatch_bf16u4b8bf16voidvoidvoidbf16f32(MMArgs args) {
  [[maybe_unused]] auto M = args.A.shape()[0];
  [[maybe_unused]] auto N = args.B.shape()[1];
  [[maybe_unused]] auto K = args.A.shape()[1];

  if (!args.maybe_schedule) {
    if (M > 256 && K <= 16384 && N <= 4096)
        return impl_bf16u4b8bf16voidvoidvoidbf16f32_sch_128x128_2x1x1_TmaMI__TmaCoop_streamK(args);
    if (M > 256)
        return impl_bf16u4b8bf16voidvoidvoidbf16f32_sch_128x256_2x1x1_TmaMI__TmaCoop_streamK(args);
    if (M > 128 && K <= 4096 && N <= 4096)
        return impl_bf16u4b8bf16voidvoidvoidbf16f32_sch_128x64_2x1x1_TmaMI__TmaCoop_streamK(args);
    if (M > 128 && K <= 8192 && N <= 8192)
        return impl_bf16u4b8bf16voidvoidvoidbf16f32_sch_128x128_2x1x1_TmaMI__TmaCoop_streamK(args);
    if (M > 128)
        return impl_bf16u4b8bf16voidvoidvoidbf16f32_sch_128x256_2x1x1_TmaMI__TmaCoop_streamK(args);
    if (M > 64 && K <= 4069 && N <= 4069)
        return impl_bf16u4b8bf16voidvoidvoidbf16f32_sch_128x32_2x1x1_TmaMI__TmaCoop_streamK(args);
    if (M > 64 && K <= 4069 && N <= 8192)
        return impl_bf16u4b8bf16voidvoidvoidbf16f32_sch_128x64_2x1x1_TmaMI__TmaCoop_streamK(args);
    if (M > 64 && K >= 8192 && N >= 12288)
        return impl_bf16u4b8bf16voidvoidvoidbf16f32_sch_256x128_2x1x1_TmaMI__TmaCoop_streamK(args);
    if (M > 64)
        return impl_bf16u4b8bf16voidvoidvoidbf16f32_sch_128x128_2x1x1_TmaMI__TmaCoop_streamK(args);
    if (M > 32 && K <= 6144 && N <= 6144)
        return impl_bf16u4b8bf16voidvoidvoidbf16f32_sch_128x16_1x1x1_TmaMI__TmaCoop_streamK(args);
    if (M > 32 && K >= 16384 && N >= 12288)
        return impl_bf16u4b8bf16voidvoidvoidbf16f32_sch_256x64_2x1x1_TmaMI__TmaCoop_streamK(args);
    if (M > 32)
        return impl_bf16u4b8bf16voidvoidvoidbf16f32_sch_128x64_2x1x1_TmaMI__TmaCoop_streamK(args);
    if (M > 16 && K <= 12288 && N <= 8192)
        return impl_bf16u4b8bf16voidvoidvoidbf16f32_sch_128x32_2x1x1_TmaMI__TmaCoop_streamK(args);
    if (M > 16)
        return impl_bf16u4b8bf16voidvoidvoidbf16f32_sch_256x32_2x1x1_TmaMI__TmaCoop_streamK(args);
    if (N >= 26624)
        return impl_bf16u4b8bf16voidvoidvoidbf16f32_sch_256x16_1x1x1_TmaMI__TmaCoop_streamK(args);
    else
        return impl_bf16u4b8bf16voidvoidvoidbf16f32_sch_128x16_1x1x1_TmaMI__TmaCoop_streamK(args);
  }
  if (*args.maybe_schedule == "128x128_2x1x1_TmaMI__TmaCoop_streamK")
    return impl_bf16u4b8bf16voidvoidvoidbf16f32_sch_128x128_2x1x1_TmaMI__TmaCoop_streamK(args);
  if (*args.maybe_schedule == "128x256_2x1x1_TmaMI__TmaCoop_streamK")
    return impl_bf16u4b8bf16voidvoidvoidbf16f32_sch_128x256_2x1x1_TmaMI__TmaCoop_streamK(args);
  if (*args.maybe_schedule == "128x64_2x1x1_TmaMI__TmaCoop_streamK")
    return impl_bf16u4b8bf16voidvoidvoidbf16f32_sch_128x64_2x1x1_TmaMI__TmaCoop_streamK(args);
  if (*args.maybe_schedule == "128x32_2x1x1_TmaMI__TmaCoop_streamK")
    return impl_bf16u4b8bf16voidvoidvoidbf16f32_sch_128x32_2x1x1_TmaMI__TmaCoop_streamK(args);
  if (*args.maybe_schedule == "256x128_2x1x1_TmaMI__TmaCoop_streamK")
    return impl_bf16u4b8bf16voidvoidvoidbf16f32_sch_256x128_2x1x1_TmaMI__TmaCoop_streamK(args);
  if (*args.maybe_schedule == "128x16_1x1x1_TmaMI__TmaCoop_streamK")
    return impl_bf16u4b8bf16voidvoidvoidbf16f32_sch_128x16_1x1x1_TmaMI__TmaCoop_streamK(args);
  if (*args.maybe_schedule == "256x64_2x1x1_TmaMI__TmaCoop_streamK")
    return impl_bf16u4b8bf16voidvoidvoidbf16f32_sch_256x64_2x1x1_TmaMI__TmaCoop_streamK(args);
  if (*args.maybe_schedule == "256x32_2x1x1_TmaMI__TmaCoop_streamK")
    return impl_bf16u4b8bf16voidvoidvoidbf16f32_sch_256x32_2x1x1_TmaMI__TmaCoop_streamK(args);
  if (*args.maybe_schedule == "256x16_1x1x1_TmaMI__TmaCoop_streamK")
    return impl_bf16u4b8bf16voidvoidvoidbf16f32_sch_256x16_1x1x1_TmaMI__TmaCoop_streamK(args);
  PADDLE_ENFORCE(false, "machete_gemm(..) is not implemented for "
                                     "schedule = %s", *args.maybe_schedule);
}

extern paddle::Tensor impl_f16u8b128f16voidvoidvoidf16f32_sch_128x128_2x1x1_TmaMI__TmaCoop_streamK(MMArgs);
extern paddle::Tensor impl_f16u8b128f16voidvoidvoidf16f32_sch_128x256_2x1x1_TmaMI__TmaCoop_streamK(MMArgs);
extern paddle::Tensor impl_f16u8b128f16voidvoidvoidf16f32_sch_128x64_2x1x1_TmaMI__TmaCoop_streamK(MMArgs);
extern paddle::Tensor impl_f16u8b128f16voidvoidvoidf16f32_sch_128x32_2x1x1_TmaMI__TmaCoop_streamK(MMArgs);
extern paddle::Tensor impl_f16u8b128f16voidvoidvoidf16f32_sch_256x128_2x1x1_TmaMI__TmaCoop_streamK(MMArgs);
extern paddle::Tensor impl_f16u8b128f16voidvoidvoidf16f32_sch_128x16_1x1x1_TmaMI__TmaCoop_streamK(MMArgs);
extern paddle::Tensor impl_f16u8b128f16voidvoidvoidf16f32_sch_256x64_2x1x1_TmaMI__TmaCoop_streamK(MMArgs);
extern paddle::Tensor impl_f16u8b128f16voidvoidvoidf16f32_sch_256x32_2x1x1_TmaMI__TmaCoop_streamK(MMArgs);
extern paddle::Tensor impl_f16u8b128f16voidvoidvoidf16f32_sch_256x16_1x1x1_TmaMI__TmaCoop_streamK(MMArgs);

paddle::Tensor mm_dispatch_f16u8b128f16voidvoidvoidf16f32(MMArgs args) {
  [[maybe_unused]] auto M = args.A.shape()[0];
  [[maybe_unused]] auto N = args.B.shape()[1];
  [[maybe_unused]] auto K = args.A.shape()[1];

  if (!args.maybe_schedule) {
    if (M > 256 && K <= 16384 && N <= 4096)
        return impl_f16u8b128f16voidvoidvoidf16f32_sch_128x128_2x1x1_TmaMI__TmaCoop_streamK(args);
    if (M > 256)
        return impl_f16u8b128f16voidvoidvoidf16f32_sch_128x256_2x1x1_TmaMI__TmaCoop_streamK(args);
    if (M > 128 && K <= 4096 && N <= 4096)
        return impl_f16u8b128f16voidvoidvoidf16f32_sch_128x64_2x1x1_TmaMI__TmaCoop_streamK(args);
    if (M > 128 && K <= 8192 && N <= 8192)
        return impl_f16u8b128f16voidvoidvoidf16f32_sch_128x128_2x1x1_TmaMI__TmaCoop_streamK(args);
    if (M > 128)
        return impl_f16u8b128f16voidvoidvoidf16f32_sch_128x256_2x1x1_TmaMI__TmaCoop_streamK(args);
    if (M > 64 && K <= 4069 && N <= 4069)
        return impl_f16u8b128f16voidvoidvoidf16f32_sch_128x32_2x1x1_TmaMI__TmaCoop_streamK(args);
    if (M > 64 && K <= 4069 && N <= 8192)
        return impl_f16u8b128f16voidvoidvoidf16f32_sch_128x64_2x1x1_TmaMI__TmaCoop_streamK(args);
    if (M > 64 && K >= 8192 && N >= 12288)
        return impl_f16u8b128f16voidvoidvoidf16f32_sch_256x128_2x1x1_TmaMI__TmaCoop_streamK(args);
    if (M > 64)
        return impl_f16u8b128f16voidvoidvoidf16f32_sch_128x128_2x1x1_TmaMI__TmaCoop_streamK(args);
    if (M > 32 && K <= 6144 && N <= 6144)
        return impl_f16u8b128f16voidvoidvoidf16f32_sch_128x16_1x1x1_TmaMI__TmaCoop_streamK(args);
    if (M > 32 && K >= 16384 && N >= 12288)
        return impl_f16u8b128f16voidvoidvoidf16f32_sch_256x64_2x1x1_TmaMI__TmaCoop_streamK(args);
    if (M > 32)
        return impl_f16u8b128f16voidvoidvoidf16f32_sch_128x64_2x1x1_TmaMI__TmaCoop_streamK(args);
    if (M > 16 && K <= 12288 && N <= 8192)
        return impl_f16u8b128f16voidvoidvoidf16f32_sch_128x32_2x1x1_TmaMI__TmaCoop_streamK(args);
    if (M > 16)
        return impl_f16u8b128f16voidvoidvoidf16f32_sch_256x32_2x1x1_TmaMI__TmaCoop_streamK(args);
    if (N >= 26624)
        return impl_f16u8b128f16voidvoidvoidf16f32_sch_256x16_1x1x1_TmaMI__TmaCoop_streamK(args);
    else
        return impl_f16u8b128f16voidvoidvoidf16f32_sch_128x16_1x1x1_TmaMI__TmaCoop_streamK(args);
  }
  if (*args.maybe_schedule == "128x128_2x1x1_TmaMI__TmaCoop_streamK")
    return impl_f16u8b128f16voidvoidvoidf16f32_sch_128x128_2x1x1_TmaMI__TmaCoop_streamK(args);
  if (*args.maybe_schedule == "128x256_2x1x1_TmaMI__TmaCoop_streamK")
    return impl_f16u8b128f16voidvoidvoidf16f32_sch_128x256_2x1x1_TmaMI__TmaCoop_streamK(args);
  if (*args.maybe_schedule == "128x64_2x1x1_TmaMI__TmaCoop_streamK")
    return impl_f16u8b128f16voidvoidvoidf16f32_sch_128x64_2x1x1_TmaMI__TmaCoop_streamK(args);
  if (*args.maybe_schedule == "128x32_2x1x1_TmaMI__TmaCoop_streamK")
    return impl_f16u8b128f16voidvoidvoidf16f32_sch_128x32_2x1x1_TmaMI__TmaCoop_streamK(args);
  if (*args.maybe_schedule == "256x128_2x1x1_TmaMI__TmaCoop_streamK")
    return impl_f16u8b128f16voidvoidvoidf16f32_sch_256x128_2x1x1_TmaMI__TmaCoop_streamK(args);
  if (*args.maybe_schedule == "128x16_1x1x1_TmaMI__TmaCoop_streamK")
    return impl_f16u8b128f16voidvoidvoidf16f32_sch_128x16_1x1x1_TmaMI__TmaCoop_streamK(args);
  if (*args.maybe_schedule == "256x64_2x1x1_TmaMI__TmaCoop_streamK")
    return impl_f16u8b128f16voidvoidvoidf16f32_sch_256x64_2x1x1_TmaMI__TmaCoop_streamK(args);
  if (*args.maybe_schedule == "256x32_2x1x1_TmaMI__TmaCoop_streamK")
    return impl_f16u8b128f16voidvoidvoidf16f32_sch_256x32_2x1x1_TmaMI__TmaCoop_streamK(args);
  if (*args.maybe_schedule == "256x16_1x1x1_TmaMI__TmaCoop_streamK")
    return impl_f16u8b128f16voidvoidvoidf16f32_sch_256x16_1x1x1_TmaMI__TmaCoop_streamK(args);
  PADDLE_ENFORCE(false, "machete_gemm(..) is not implemented for "
                                     "schedule = %s", *args.maybe_schedule);
}

extern paddle::Tensor impl_bf16u8b128bf16voidvoidvoidbf16f32_sch_128x128_2x1x1_TmaMI__TmaCoop_streamK(MMArgs);
extern paddle::Tensor impl_bf16u8b128bf16voidvoidvoidbf16f32_sch_128x256_2x1x1_TmaMI__TmaCoop_streamK(MMArgs);
extern paddle::Tensor impl_bf16u8b128bf16voidvoidvoidbf16f32_sch_128x64_2x1x1_TmaMI__TmaCoop_streamK(MMArgs);
extern paddle::Tensor impl_bf16u8b128bf16voidvoidvoidbf16f32_sch_128x32_2x1x1_TmaMI__TmaCoop_streamK(MMArgs);
extern paddle::Tensor impl_bf16u8b128bf16voidvoidvoidbf16f32_sch_256x128_2x1x1_TmaMI__TmaCoop_streamK(MMArgs);
extern paddle::Tensor impl_bf16u8b128bf16voidvoidvoidbf16f32_sch_128x16_1x1x1_TmaMI__TmaCoop_streamK(MMArgs);
extern paddle::Tensor impl_bf16u8b128bf16voidvoidvoidbf16f32_sch_256x64_2x1x1_TmaMI__TmaCoop_streamK(MMArgs);
extern paddle::Tensor impl_bf16u8b128bf16voidvoidvoidbf16f32_sch_256x32_2x1x1_TmaMI__TmaCoop_streamK(MMArgs);
extern paddle::Tensor impl_bf16u8b128bf16voidvoidvoidbf16f32_sch_256x16_1x1x1_TmaMI__TmaCoop_streamK(MMArgs);

paddle::Tensor mm_dispatch_bf16u8b128bf16voidvoidvoidbf16f32(MMArgs args) {
  [[maybe_unused]] auto M = args.A.shape()[0];
  [[maybe_unused]] auto N = args.B.shape()[1];
  [[maybe_unused]] auto K = args.A.shape()[1];

  if (!args.maybe_schedule) {
    if (M > 256 && K <= 16384 && N <= 4096)
        return impl_bf16u8b128bf16voidvoidvoidbf16f32_sch_128x128_2x1x1_TmaMI__TmaCoop_streamK(args);
    if (M > 256)
        return impl_bf16u8b128bf16voidvoidvoidbf16f32_sch_128x256_2x1x1_TmaMI__TmaCoop_streamK(args);
    if (M > 128 && K <= 4096 && N <= 4096)
        return impl_bf16u8b128bf16voidvoidvoidbf16f32_sch_128x64_2x1x1_TmaMI__TmaCoop_streamK(args);
    if (M > 128 && K <= 8192 && N <= 8192)
        return impl_bf16u8b128bf16voidvoidvoidbf16f32_sch_128x128_2x1x1_TmaMI__TmaCoop_streamK(args);
    if (M > 128)
        return impl_bf16u8b128bf16voidvoidvoidbf16f32_sch_128x256_2x1x1_TmaMI__TmaCoop_streamK(args);
    if (M > 64 && K <= 4069 && N <= 4069)
        return impl_bf16u8b128bf16voidvoidvoidbf16f32_sch_128x32_2x1x1_TmaMI__TmaCoop_streamK(args);
    if (M > 64 && K <= 4069 && N <= 8192)
        return impl_bf16u8b128bf16voidvoidvoidbf16f32_sch_128x64_2x1x1_TmaMI__TmaCoop_streamK(args);
    if (M > 64 && K >= 8192 && N >= 12288)
        return impl_bf16u8b128bf16voidvoidvoidbf16f32_sch_256x128_2x1x1_TmaMI__TmaCoop_streamK(args);
    if (M > 64)
        return impl_bf16u8b128bf16voidvoidvoidbf16f32_sch_128x128_2x1x1_TmaMI__TmaCoop_streamK(args);
    if (M > 32 && K <= 6144 && N <= 6144)
        return impl_bf16u8b128bf16voidvoidvoidbf16f32_sch_128x16_1x1x1_TmaMI__TmaCoop_streamK(args);
    if (M > 32 && K >= 16384 && N >= 12288)
        return impl_bf16u8b128bf16voidvoidvoidbf16f32_sch_256x64_2x1x1_TmaMI__TmaCoop_streamK(args);
    if (M > 32)
        return impl_bf16u8b128bf16voidvoidvoidbf16f32_sch_128x64_2x1x1_TmaMI__TmaCoop_streamK(args);
    if (M > 16 && K <= 12288 && N <= 8192)
        return impl_bf16u8b128bf16voidvoidvoidbf16f32_sch_128x32_2x1x1_TmaMI__TmaCoop_streamK(args);
    if (M > 16)
        return impl_bf16u8b128bf16voidvoidvoidbf16f32_sch_256x32_2x1x1_TmaMI__TmaCoop_streamK(args);
    if (N >= 26624)
        return impl_bf16u8b128bf16voidvoidvoidbf16f32_sch_256x16_1x1x1_TmaMI__TmaCoop_streamK(args);
    else
        return impl_bf16u8b128bf16voidvoidvoidbf16f32_sch_128x16_1x1x1_TmaMI__TmaCoop_streamK(args);
  }
  if (*args.maybe_schedule == "128x128_2x1x1_TmaMI__TmaCoop_streamK")
    return impl_bf16u8b128bf16voidvoidvoidbf16f32_sch_128x128_2x1x1_TmaMI__TmaCoop_streamK(args);
  if (*args.maybe_schedule == "128x256_2x1x1_TmaMI__TmaCoop_streamK")
    return impl_bf16u8b128bf16voidvoidvoidbf16f32_sch_128x256_2x1x1_TmaMI__TmaCoop_streamK(args);
  if (*args.maybe_schedule == "128x64_2x1x1_TmaMI__TmaCoop_streamK")
    return impl_bf16u8b128bf16voidvoidvoidbf16f32_sch_128x64_2x1x1_TmaMI__TmaCoop_streamK(args);
  if (*args.maybe_schedule == "128x32_2x1x1_TmaMI__TmaCoop_streamK")
    return impl_bf16u8b128bf16voidvoidvoidbf16f32_sch_128x32_2x1x1_TmaMI__TmaCoop_streamK(args);
  if (*args.maybe_schedule == "256x128_2x1x1_TmaMI__TmaCoop_streamK")
    return impl_bf16u8b128bf16voidvoidvoidbf16f32_sch_256x128_2x1x1_TmaMI__TmaCoop_streamK(args);
  if (*args.maybe_schedule == "128x16_1x1x1_TmaMI__TmaCoop_streamK")
    return impl_bf16u8b128bf16voidvoidvoidbf16f32_sch_128x16_1x1x1_TmaMI__TmaCoop_streamK(args);
  if (*args.maybe_schedule == "256x64_2x1x1_TmaMI__TmaCoop_streamK")
    return impl_bf16u8b128bf16voidvoidvoidbf16f32_sch_256x64_2x1x1_TmaMI__TmaCoop_streamK(args);
  if (*args.maybe_schedule == "256x32_2x1x1_TmaMI__TmaCoop_streamK")
    return impl_bf16u8b128bf16voidvoidvoidbf16f32_sch_256x32_2x1x1_TmaMI__TmaCoop_streamK(args);
  if (*args.maybe_schedule == "256x16_1x1x1_TmaMI__TmaCoop_streamK")
    return impl_bf16u8b128bf16voidvoidvoidbf16f32_sch_256x16_1x1x1_TmaMI__TmaCoop_streamK(args);
  PADDLE_ENFORCE(false, "machete_gemm(..) is not implemented for "
                                     "schedule = %s", *args.maybe_schedule);
}

extern paddle::Tensor impl_f16u4f16f16voidvoidf16f32_sch_128x128_2x1x1_TmaMI__TmaCoop_streamK(MMArgs);
extern paddle::Tensor impl_f16u4f16f16voidvoidf16f32_sch_128x256_2x1x1_TmaMI__TmaCoop_streamK(MMArgs);
extern paddle::Tensor impl_f16u4f16f16voidvoidf16f32_sch_128x64_2x1x1_TmaMI__TmaCoop_streamK(MMArgs);
extern paddle::Tensor impl_f16u4f16f16voidvoidf16f32_sch_128x32_2x1x1_TmaMI__TmaCoop_streamK(MMArgs);
extern paddle::Tensor impl_f16u4f16f16voidvoidf16f32_sch_256x128_2x1x1_TmaMI__TmaCoop_streamK(MMArgs);
extern paddle::Tensor impl_f16u4f16f16voidvoidf16f32_sch_128x16_1x1x1_TmaMI__TmaCoop_streamK(MMArgs);
extern paddle::Tensor impl_f16u4f16f16voidvoidf16f32_sch_256x64_2x1x1_TmaMI__TmaCoop_streamK(MMArgs);
extern paddle::Tensor impl_f16u4f16f16voidvoidf16f32_sch_256x32_2x1x1_TmaMI__TmaCoop_streamK(MMArgs);
extern paddle::Tensor impl_f16u4f16f16voidvoidf16f32_sch_256x16_1x1x1_TmaMI__TmaCoop_streamK(MMArgs);

paddle::Tensor mm_dispatch_f16u4f16f16voidvoidf16f32(MMArgs args) {
  [[maybe_unused]] auto M = args.A.shape()[0];
  [[maybe_unused]] auto N = args.B.shape()[1];
  [[maybe_unused]] auto K = args.A.shape()[1];

  if (!args.maybe_schedule) {
    if (M > 256 && K <= 16384 && N <= 4096)
        return impl_f16u4f16f16voidvoidf16f32_sch_128x128_2x1x1_TmaMI__TmaCoop_streamK(args);
    if (M > 256)
        return impl_f16u4f16f16voidvoidf16f32_sch_128x256_2x1x1_TmaMI__TmaCoop_streamK(args);
    if (M > 128 && K <= 4096 && N <= 4096)
        return impl_f16u4f16f16voidvoidf16f32_sch_128x64_2x1x1_TmaMI__TmaCoop_streamK(args);
    if (M > 128 && K <= 8192 && N <= 8192)
        return impl_f16u4f16f16voidvoidf16f32_sch_128x128_2x1x1_TmaMI__TmaCoop_streamK(args);
    if (M > 128)
        return impl_f16u4f16f16voidvoidf16f32_sch_128x256_2x1x1_TmaMI__TmaCoop_streamK(args);
    if (M > 64 && K <= 4069 && N <= 4069)
        return impl_f16u4f16f16voidvoidf16f32_sch_128x32_2x1x1_TmaMI__TmaCoop_streamK(args);
    if (M > 64 && K <= 4069 && N <= 8192)
        return impl_f16u4f16f16voidvoidf16f32_sch_128x64_2x1x1_TmaMI__TmaCoop_streamK(args);
    if (M > 64 && K >= 8192 && N >= 12288)
        return impl_f16u4f16f16voidvoidf16f32_sch_256x128_2x1x1_TmaMI__TmaCoop_streamK(args);
    if (M > 64)
        return impl_f16u4f16f16voidvoidf16f32_sch_128x128_2x1x1_TmaMI__TmaCoop_streamK(args);
    if (M > 32 && K <= 6144 && N <= 6144)
        return impl_f16u4f16f16voidvoidf16f32_sch_128x16_1x1x1_TmaMI__TmaCoop_streamK(args);
    if (M > 32 && K >= 16384 && N >= 12288)
        return impl_f16u4f16f16voidvoidf16f32_sch_256x64_2x1x1_TmaMI__TmaCoop_streamK(args);
    if (M > 32)
        return impl_f16u4f16f16voidvoidf16f32_sch_128x64_2x1x1_TmaMI__TmaCoop_streamK(args);
    if (M > 16 && K <= 12288 && N <= 8192)
        return impl_f16u4f16f16voidvoidf16f32_sch_128x32_2x1x1_TmaMI__TmaCoop_streamK(args);
    if (M > 16)
        return impl_f16u4f16f16voidvoidf16f32_sch_256x32_2x1x1_TmaMI__TmaCoop_streamK(args);
    if (N >= 26624)
        return impl_f16u4f16f16voidvoidf16f32_sch_256x16_1x1x1_TmaMI__TmaCoop_streamK(args);
    else
        return impl_f16u4f16f16voidvoidf16f32_sch_128x16_1x1x1_TmaMI__TmaCoop_streamK(args);
  }
  if (*args.maybe_schedule == "128x128_2x1x1_TmaMI__TmaCoop_streamK")
    return impl_f16u4f16f16voidvoidf16f32_sch_128x128_2x1x1_TmaMI__TmaCoop_streamK(args);
  if (*args.maybe_schedule == "128x256_2x1x1_TmaMI__TmaCoop_streamK")
    return impl_f16u4f16f16voidvoidf16f32_sch_128x256_2x1x1_TmaMI__TmaCoop_streamK(args);
  if (*args.maybe_schedule == "128x64_2x1x1_TmaMI__TmaCoop_streamK")
    return impl_f16u4f16f16voidvoidf16f32_sch_128x64_2x1x1_TmaMI__TmaCoop_streamK(args);
  if (*args.maybe_schedule == "128x32_2x1x1_TmaMI__TmaCoop_streamK")
    return impl_f16u4f16f16voidvoidf16f32_sch_128x32_2x1x1_TmaMI__TmaCoop_streamK(args);
  if (*args.maybe_schedule == "256x128_2x1x1_TmaMI__TmaCoop_streamK")
    return impl_f16u4f16f16voidvoidf16f32_sch_256x128_2x1x1_TmaMI__TmaCoop_streamK(args);
  if (*args.maybe_schedule == "128x16_1x1x1_TmaMI__TmaCoop_streamK")
    return impl_f16u4f16f16voidvoidf16f32_sch_128x16_1x1x1_TmaMI__TmaCoop_streamK(args);
  if (*args.maybe_schedule == "256x64_2x1x1_TmaMI__TmaCoop_streamK")
    return impl_f16u4f16f16voidvoidf16f32_sch_256x64_2x1x1_TmaMI__TmaCoop_streamK(args);
  if (*args.maybe_schedule == "256x32_2x1x1_TmaMI__TmaCoop_streamK")
    return impl_f16u4f16f16voidvoidf16f32_sch_256x32_2x1x1_TmaMI__TmaCoop_streamK(args);
  if (*args.maybe_schedule == "256x16_1x1x1_TmaMI__TmaCoop_streamK")
    return impl_f16u4f16f16voidvoidf16f32_sch_256x16_1x1x1_TmaMI__TmaCoop_streamK(args);
  PADDLE_ENFORCE(false, "machete_gemm(..) is not implemented for "
                                     "schedule = %s", *args.maybe_schedule);
}

extern paddle::Tensor impl_bf16u4bf16bf16voidvoidbf16f32_sch_128x128_2x1x1_TmaMI__TmaCoop_streamK(MMArgs);
extern paddle::Tensor impl_bf16u4bf16bf16voidvoidbf16f32_sch_128x256_2x1x1_TmaMI__TmaCoop_streamK(MMArgs);
extern paddle::Tensor impl_bf16u4bf16bf16voidvoidbf16f32_sch_128x64_2x1x1_TmaMI__TmaCoop_streamK(MMArgs);
extern paddle::Tensor impl_bf16u4bf16bf16voidvoidbf16f32_sch_128x32_2x1x1_TmaMI__TmaCoop_streamK(MMArgs);
extern paddle::Tensor impl_bf16u4bf16bf16voidvoidbf16f32_sch_256x128_2x1x1_TmaMI__TmaCoop_streamK(MMArgs);
extern paddle::Tensor impl_bf16u4bf16bf16voidvoidbf16f32_sch_128x16_1x1x1_TmaMI__TmaCoop_streamK(MMArgs);
extern paddle::Tensor impl_bf16u4bf16bf16voidvoidbf16f32_sch_256x64_2x1x1_TmaMI__TmaCoop_streamK(MMArgs);
extern paddle::Tensor impl_bf16u4bf16bf16voidvoidbf16f32_sch_256x32_2x1x1_TmaMI__TmaCoop_streamK(MMArgs);
extern paddle::Tensor impl_bf16u4bf16bf16voidvoidbf16f32_sch_256x16_1x1x1_TmaMI__TmaCoop_streamK(MMArgs);

paddle::Tensor mm_dispatch_bf16u4bf16bf16voidvoidbf16f32(MMArgs args) {
  [[maybe_unused]] auto M = args.A.shape()[0];
  [[maybe_unused]] auto N = args.B.shape()[1];
  [[maybe_unused]] auto K = args.A.shape()[1];

  if (!args.maybe_schedule) {
    if (M > 256 && K <= 16384 && N <= 4096)
        return impl_bf16u4bf16bf16voidvoidbf16f32_sch_128x128_2x1x1_TmaMI__TmaCoop_streamK(args);
    if (M > 256)
        return impl_bf16u4bf16bf16voidvoidbf16f32_sch_128x256_2x1x1_TmaMI__TmaCoop_streamK(args);
    if (M > 128 && K <= 4096 && N <= 4096)
        return impl_bf16u4bf16bf16voidvoidbf16f32_sch_128x64_2x1x1_TmaMI__TmaCoop_streamK(args);
    if (M > 128 && K <= 8192 && N <= 8192)
        return impl_bf16u4bf16bf16voidvoidbf16f32_sch_128x128_2x1x1_TmaMI__TmaCoop_streamK(args);
    if (M > 128)
        return impl_bf16u4bf16bf16voidvoidbf16f32_sch_128x256_2x1x1_TmaMI__TmaCoop_streamK(args);
    if (M > 64 && K <= 4069 && N <= 4069)
        return impl_bf16u4bf16bf16voidvoidbf16f32_sch_128x32_2x1x1_TmaMI__TmaCoop_streamK(args);
    if (M > 64 && K <= 4069 && N <= 8192)
        return impl_bf16u4bf16bf16voidvoidbf16f32_sch_128x64_2x1x1_TmaMI__TmaCoop_streamK(args);
    if (M > 64 && K >= 8192 && N >= 12288)
        return impl_bf16u4bf16bf16voidvoidbf16f32_sch_256x128_2x1x1_TmaMI__TmaCoop_streamK(args);
    if (M > 64)
        return impl_bf16u4bf16bf16voidvoidbf16f32_sch_128x128_2x1x1_TmaMI__TmaCoop_streamK(args);
    if (M > 32 && K <= 6144 && N <= 6144)
        return impl_bf16u4bf16bf16voidvoidbf16f32_sch_128x16_1x1x1_TmaMI__TmaCoop_streamK(args);
    if (M > 32 && K >= 16384 && N >= 12288)
        return impl_bf16u4bf16bf16voidvoidbf16f32_sch_256x64_2x1x1_TmaMI__TmaCoop_streamK(args);
    if (M > 32)
        return impl_bf16u4bf16bf16voidvoidbf16f32_sch_128x64_2x1x1_TmaMI__TmaCoop_streamK(args);
    if (M > 16 && K <= 12288 && N <= 8192)
        return impl_bf16u4bf16bf16voidvoidbf16f32_sch_128x32_2x1x1_TmaMI__TmaCoop_streamK(args);
    if (M > 16)
        return impl_bf16u4bf16bf16voidvoidbf16f32_sch_256x32_2x1x1_TmaMI__TmaCoop_streamK(args);
    if (N >= 26624)
        return impl_bf16u4bf16bf16voidvoidbf16f32_sch_256x16_1x1x1_TmaMI__TmaCoop_streamK(args);
    else
        return impl_bf16u4bf16bf16voidvoidbf16f32_sch_128x16_1x1x1_TmaMI__TmaCoop_streamK(args);
  }
  if (*args.maybe_schedule == "128x128_2x1x1_TmaMI__TmaCoop_streamK")
    return impl_bf16u4bf16bf16voidvoidbf16f32_sch_128x128_2x1x1_TmaMI__TmaCoop_streamK(args);
  if (*args.maybe_schedule == "128x256_2x1x1_TmaMI__TmaCoop_streamK")
    return impl_bf16u4bf16bf16voidvoidbf16f32_sch_128x256_2x1x1_TmaMI__TmaCoop_streamK(args);
  if (*args.maybe_schedule == "128x64_2x1x1_TmaMI__TmaCoop_streamK")
    return impl_bf16u4bf16bf16voidvoidbf16f32_sch_128x64_2x1x1_TmaMI__TmaCoop_streamK(args);
  if (*args.maybe_schedule == "128x32_2x1x1_TmaMI__TmaCoop_streamK")
    return impl_bf16u4bf16bf16voidvoidbf16f32_sch_128x32_2x1x1_TmaMI__TmaCoop_streamK(args);
  if (*args.maybe_schedule == "256x128_2x1x1_TmaMI__TmaCoop_streamK")
    return impl_bf16u4bf16bf16voidvoidbf16f32_sch_256x128_2x1x1_TmaMI__TmaCoop_streamK(args);
  if (*args.maybe_schedule == "128x16_1x1x1_TmaMI__TmaCoop_streamK")
    return impl_bf16u4bf16bf16voidvoidbf16f32_sch_128x16_1x1x1_TmaMI__TmaCoop_streamK(args);
  if (*args.maybe_schedule == "256x64_2x1x1_TmaMI__TmaCoop_streamK")
    return impl_bf16u4bf16bf16voidvoidbf16f32_sch_256x64_2x1x1_TmaMI__TmaCoop_streamK(args);
  if (*args.maybe_schedule == "256x32_2x1x1_TmaMI__TmaCoop_streamK")
    return impl_bf16u4bf16bf16voidvoidbf16f32_sch_256x32_2x1x1_TmaMI__TmaCoop_streamK(args);
  if (*args.maybe_schedule == "256x16_1x1x1_TmaMI__TmaCoop_streamK")
    return impl_bf16u4bf16bf16voidvoidbf16f32_sch_256x16_1x1x1_TmaMI__TmaCoop_streamK(args);
  PADDLE_ENFORCE(false, "machete_gemm(..) is not implemented for "
                                     "schedule = %s", *args.maybe_schedule);
}

extern paddle::Tensor impl_f16u8f16f16voidvoidf16f32_sch_128x128_2x1x1_TmaMI__TmaCoop_streamK(MMArgs);
extern paddle::Tensor impl_f16u8f16f16voidvoidf16f32_sch_128x256_2x1x1_TmaMI__TmaCoop_streamK(MMArgs);
extern paddle::Tensor impl_f16u8f16f16voidvoidf16f32_sch_128x64_2x1x1_TmaMI__TmaCoop_streamK(MMArgs);
extern paddle::Tensor impl_f16u8f16f16voidvoidf16f32_sch_128x32_2x1x1_TmaMI__TmaCoop_streamK(MMArgs);
extern paddle::Tensor impl_f16u8f16f16voidvoidf16f32_sch_256x128_2x1x1_TmaMI__TmaCoop_streamK(MMArgs);
extern paddle::Tensor impl_f16u8f16f16voidvoidf16f32_sch_128x16_1x1x1_TmaMI__TmaCoop_streamK(MMArgs);
extern paddle::Tensor impl_f16u8f16f16voidvoidf16f32_sch_256x64_2x1x1_TmaMI__TmaCoop_streamK(MMArgs);
extern paddle::Tensor impl_f16u8f16f16voidvoidf16f32_sch_256x32_2x1x1_TmaMI__TmaCoop_streamK(MMArgs);
extern paddle::Tensor impl_f16u8f16f16voidvoidf16f32_sch_256x16_1x1x1_TmaMI__TmaCoop_streamK(MMArgs);

paddle::Tensor mm_dispatch_f16u8f16f16voidvoidf16f32(MMArgs args) {
  [[maybe_unused]] auto M = args.A.shape()[0];
  [[maybe_unused]] auto N = args.B.shape()[1];
  [[maybe_unused]] auto K = args.A.shape()[1];

  if (!args.maybe_schedule) {
    if (M > 256 && K <= 16384 && N <= 4096)
        return impl_f16u8f16f16voidvoidf16f32_sch_128x128_2x1x1_TmaMI__TmaCoop_streamK(args);
    if (M > 256)
        return impl_f16u8f16f16voidvoidf16f32_sch_128x256_2x1x1_TmaMI__TmaCoop_streamK(args);
    if (M > 128 && K <= 4096 && N <= 4096)
        return impl_f16u8f16f16voidvoidf16f32_sch_128x64_2x1x1_TmaMI__TmaCoop_streamK(args);
    if (M > 128 && K <= 8192 && N <= 8192)
        return impl_f16u8f16f16voidvoidf16f32_sch_128x128_2x1x1_TmaMI__TmaCoop_streamK(args);
    if (M > 128)
        return impl_f16u8f16f16voidvoidf16f32_sch_128x256_2x1x1_TmaMI__TmaCoop_streamK(args);
    if (M > 64 && K <= 4069 && N <= 4069)
        return impl_f16u8f16f16voidvoidf16f32_sch_128x32_2x1x1_TmaMI__TmaCoop_streamK(args);
    if (M > 64 && K <= 4069 && N <= 8192)
        return impl_f16u8f16f16voidvoidf16f32_sch_128x64_2x1x1_TmaMI__TmaCoop_streamK(args);
    if (M > 64 && K >= 8192 && N >= 12288)
        return impl_f16u8f16f16voidvoidf16f32_sch_256x128_2x1x1_TmaMI__TmaCoop_streamK(args);
    if (M > 64)
        return impl_f16u8f16f16voidvoidf16f32_sch_128x128_2x1x1_TmaMI__TmaCoop_streamK(args);
    if (M > 32 && K <= 6144 && N <= 6144)
        return impl_f16u8f16f16voidvoidf16f32_sch_128x16_1x1x1_TmaMI__TmaCoop_streamK(args);
    if (M > 32 && K >= 16384 && N >= 12288)
        return impl_f16u8f16f16voidvoidf16f32_sch_256x64_2x1x1_TmaMI__TmaCoop_streamK(args);
    if (M > 32)
        return impl_f16u8f16f16voidvoidf16f32_sch_128x64_2x1x1_TmaMI__TmaCoop_streamK(args);
    if (M > 16 && K <= 12288 && N <= 8192)
        return impl_f16u8f16f16voidvoidf16f32_sch_128x32_2x1x1_TmaMI__TmaCoop_streamK(args);
    if (M > 16)
        return impl_f16u8f16f16voidvoidf16f32_sch_256x32_2x1x1_TmaMI__TmaCoop_streamK(args);
    if (N >= 26624)
        return impl_f16u8f16f16voidvoidf16f32_sch_256x16_1x1x1_TmaMI__TmaCoop_streamK(args);
    else
        return impl_f16u8f16f16voidvoidf16f32_sch_128x16_1x1x1_TmaMI__TmaCoop_streamK(args);
  }
  if (*args.maybe_schedule == "128x128_2x1x1_TmaMI__TmaCoop_streamK")
    return impl_f16u8f16f16voidvoidf16f32_sch_128x128_2x1x1_TmaMI__TmaCoop_streamK(args);
  if (*args.maybe_schedule == "128x256_2x1x1_TmaMI__TmaCoop_streamK")
    return impl_f16u8f16f16voidvoidf16f32_sch_128x256_2x1x1_TmaMI__TmaCoop_streamK(args);
  if (*args.maybe_schedule == "128x64_2x1x1_TmaMI__TmaCoop_streamK")
    return impl_f16u8f16f16voidvoidf16f32_sch_128x64_2x1x1_TmaMI__TmaCoop_streamK(args);
  if (*args.maybe_schedule == "128x32_2x1x1_TmaMI__TmaCoop_streamK")
    return impl_f16u8f16f16voidvoidf16f32_sch_128x32_2x1x1_TmaMI__TmaCoop_streamK(args);
  if (*args.maybe_schedule == "256x128_2x1x1_TmaMI__TmaCoop_streamK")
    return impl_f16u8f16f16voidvoidf16f32_sch_256x128_2x1x1_TmaMI__TmaCoop_streamK(args);
  if (*args.maybe_schedule == "128x16_1x1x1_TmaMI__TmaCoop_streamK")
    return impl_f16u8f16f16voidvoidf16f32_sch_128x16_1x1x1_TmaMI__TmaCoop_streamK(args);
  if (*args.maybe_schedule == "256x64_2x1x1_TmaMI__TmaCoop_streamK")
    return impl_f16u8f16f16voidvoidf16f32_sch_256x64_2x1x1_TmaMI__TmaCoop_streamK(args);
  if (*args.maybe_schedule == "256x32_2x1x1_TmaMI__TmaCoop_streamK")
    return impl_f16u8f16f16voidvoidf16f32_sch_256x32_2x1x1_TmaMI__TmaCoop_streamK(args);
  if (*args.maybe_schedule == "256x16_1x1x1_TmaMI__TmaCoop_streamK")
    return impl_f16u8f16f16voidvoidf16f32_sch_256x16_1x1x1_TmaMI__TmaCoop_streamK(args);
  PADDLE_ENFORCE(false, "machete_gemm(..) is not implemented for "
                                     "schedule = %s", *args.maybe_schedule);
}

extern paddle::Tensor impl_bf16u8bf16bf16voidvoidbf16f32_sch_128x128_2x1x1_TmaMI__TmaCoop_streamK(MMArgs);
extern paddle::Tensor impl_bf16u8bf16bf16voidvoidbf16f32_sch_128x256_2x1x1_TmaMI__TmaCoop_streamK(MMArgs);
extern paddle::Tensor impl_bf16u8bf16bf16voidvoidbf16f32_sch_128x64_2x1x1_TmaMI__TmaCoop_streamK(MMArgs);
extern paddle::Tensor impl_bf16u8bf16bf16voidvoidbf16f32_sch_128x32_2x1x1_TmaMI__TmaCoop_streamK(MMArgs);
extern paddle::Tensor impl_bf16u8bf16bf16voidvoidbf16f32_sch_256x128_2x1x1_TmaMI__TmaCoop_streamK(MMArgs);
extern paddle::Tensor impl_bf16u8bf16bf16voidvoidbf16f32_sch_128x16_1x1x1_TmaMI__TmaCoop_streamK(MMArgs);
extern paddle::Tensor impl_bf16u8bf16bf16voidvoidbf16f32_sch_256x64_2x1x1_TmaMI__TmaCoop_streamK(MMArgs);
extern paddle::Tensor impl_bf16u8bf16bf16voidvoidbf16f32_sch_256x32_2x1x1_TmaMI__TmaCoop_streamK(MMArgs);
extern paddle::Tensor impl_bf16u8bf16bf16voidvoidbf16f32_sch_256x16_1x1x1_TmaMI__TmaCoop_streamK(MMArgs);

paddle::Tensor mm_dispatch_bf16u8bf16bf16voidvoidbf16f32(MMArgs args) {
  [[maybe_unused]] auto M = args.A.shape()[0];
  [[maybe_unused]] auto N = args.B.shape()[1];
  [[maybe_unused]] auto K = args.A.shape()[1];

  if (!args.maybe_schedule) {
    if (M > 256 && K <= 16384 && N <= 4096)
        return impl_bf16u8bf16bf16voidvoidbf16f32_sch_128x128_2x1x1_TmaMI__TmaCoop_streamK(args);
    if (M > 256)
        return impl_bf16u8bf16bf16voidvoidbf16f32_sch_128x256_2x1x1_TmaMI__TmaCoop_streamK(args);
    if (M > 128 && K <= 4096 && N <= 4096)
        return impl_bf16u8bf16bf16voidvoidbf16f32_sch_128x64_2x1x1_TmaMI__TmaCoop_streamK(args);
    if (M > 128 && K <= 8192 && N <= 8192)
        return impl_bf16u8bf16bf16voidvoidbf16f32_sch_128x128_2x1x1_TmaMI__TmaCoop_streamK(args);
    if (M > 128)
        return impl_bf16u8bf16bf16voidvoidbf16f32_sch_128x256_2x1x1_TmaMI__TmaCoop_streamK(args);
    if (M > 64 && K <= 4069 && N <= 4069)
        return impl_bf16u8bf16bf16voidvoidbf16f32_sch_128x32_2x1x1_TmaMI__TmaCoop_streamK(args);
    if (M > 64 && K <= 4069 && N <= 8192)
        return impl_bf16u8bf16bf16voidvoidbf16f32_sch_128x64_2x1x1_TmaMI__TmaCoop_streamK(args);
    if (M > 64 && K >= 8192 && N >= 12288)
        return impl_bf16u8bf16bf16voidvoidbf16f32_sch_256x128_2x1x1_TmaMI__TmaCoop_streamK(args);
    if (M > 64)
        return impl_bf16u8bf16bf16voidvoidbf16f32_sch_128x128_2x1x1_TmaMI__TmaCoop_streamK(args);
    if (M > 32 && K <= 6144 && N <= 6144)
        return impl_bf16u8bf16bf16voidvoidbf16f32_sch_128x16_1x1x1_TmaMI__TmaCoop_streamK(args);
    if (M > 32 && K >= 16384 && N >= 12288)
        return impl_bf16u8bf16bf16voidvoidbf16f32_sch_256x64_2x1x1_TmaMI__TmaCoop_streamK(args);
    if (M > 32)
        return impl_bf16u8bf16bf16voidvoidbf16f32_sch_128x64_2x1x1_TmaMI__TmaCoop_streamK(args);
    if (M > 16 && K <= 12288 && N <= 8192)
        return impl_bf16u8bf16bf16voidvoidbf16f32_sch_128x32_2x1x1_TmaMI__TmaCoop_streamK(args);
    if (M > 16)
        return impl_bf16u8bf16bf16voidvoidbf16f32_sch_256x32_2x1x1_TmaMI__TmaCoop_streamK(args);
    if (N >= 26624)
        return impl_bf16u8bf16bf16voidvoidbf16f32_sch_256x16_1x1x1_TmaMI__TmaCoop_streamK(args);
    else
        return impl_bf16u8bf16bf16voidvoidbf16f32_sch_128x16_1x1x1_TmaMI__TmaCoop_streamK(args);
  }
  if (*args.maybe_schedule == "128x128_2x1x1_TmaMI__TmaCoop_streamK")
    return impl_bf16u8bf16bf16voidvoidbf16f32_sch_128x128_2x1x1_TmaMI__TmaCoop_streamK(args);
  if (*args.maybe_schedule == "128x256_2x1x1_TmaMI__TmaCoop_streamK")
    return impl_bf16u8bf16bf16voidvoidbf16f32_sch_128x256_2x1x1_TmaMI__TmaCoop_streamK(args);
  if (*args.maybe_schedule == "128x64_2x1x1_TmaMI__TmaCoop_streamK")
    return impl_bf16u8bf16bf16voidvoidbf16f32_sch_128x64_2x1x1_TmaMI__TmaCoop_streamK(args);
  if (*args.maybe_schedule == "128x32_2x1x1_TmaMI__TmaCoop_streamK")
    return impl_bf16u8bf16bf16voidvoidbf16f32_sch_128x32_2x1x1_TmaMI__TmaCoop_streamK(args);
  if (*args.maybe_schedule == "256x128_2x1x1_TmaMI__TmaCoop_streamK")
    return impl_bf16u8bf16bf16voidvoidbf16f32_sch_256x128_2x1x1_TmaMI__TmaCoop_streamK(args);
  if (*args.maybe_schedule == "128x16_1x1x1_TmaMI__TmaCoop_streamK")
    return impl_bf16u8bf16bf16voidvoidbf16f32_sch_128x16_1x1x1_TmaMI__TmaCoop_streamK(args);
  if (*args.maybe_schedule == "256x64_2x1x1_TmaMI__TmaCoop_streamK")
    return impl_bf16u8bf16bf16voidvoidbf16f32_sch_256x64_2x1x1_TmaMI__TmaCoop_streamK(args);
  if (*args.maybe_schedule == "256x32_2x1x1_TmaMI__TmaCoop_streamK")
    return impl_bf16u8bf16bf16voidvoidbf16f32_sch_256x32_2x1x1_TmaMI__TmaCoop_streamK(args);
  if (*args.maybe_schedule == "256x16_1x1x1_TmaMI__TmaCoop_streamK")
    return impl_bf16u8bf16bf16voidvoidbf16f32_sch_256x16_1x1x1_TmaMI__TmaCoop_streamK(args);
  PADDLE_ENFORCE(false, "machete_gemm(..) is not implemented for "
                                     "schedule = %s", *args.maybe_schedule);
}

extern paddle::Tensor impl_s8u4b8f16voidf32f32f16s32_sch_128x128_2x1x1_TmaMI__TmaCoop_streamK(MMArgs);
extern paddle::Tensor impl_s8u4b8f16voidf32f32f16s32_sch_128x64_2x1x1_TmaMI__TmaCoop_streamK(MMArgs);
extern paddle::Tensor impl_s8u4b8f16voidf32f32f16s32_sch_128x32_2x1x1_TmaMI__TmaCoop_streamK(MMArgs);
extern paddle::Tensor impl_s8u4b8f16voidf32f32f16s32_sch_256x128_2x1x1_TmaMI__TmaCoop_streamK(MMArgs);
extern paddle::Tensor impl_s8u4b8f16voidf32f32f16s32_sch_128x16_1x1x1_TmaMI__TmaCoop_streamK(MMArgs);
extern paddle::Tensor impl_s8u4b8f16voidf32f32f16s32_sch_256x32_2x1x1_TmaMI__TmaCoop_streamK(MMArgs);
extern paddle::Tensor impl_s8u4b8f16voidf32f32f16s32_sch_256x16_1x1x1_TmaMI__TmaCoop_streamK(MMArgs);

paddle::Tensor mm_dispatch_s8u4b8f16voidf32f32f16s32(MMArgs args) {
  [[maybe_unused]] auto M = args.A.shape()[0];
  [[maybe_unused]] auto N = args.B.shape()[1];
  [[maybe_unused]] auto K = args.A.shape()[1];

  if (!args.maybe_schedule) {
    if (M > 256)
        return impl_s8u4b8f16voidf32f32f16s32_sch_128x128_2x1x1_TmaMI__TmaCoop_streamK(args);
    if (M > 128 && K <= 4096 && N <= 4096)
        return impl_s8u4b8f16voidf32f32f16s32_sch_128x64_2x1x1_TmaMI__TmaCoop_streamK(args);
    if (M > 128 && K <= 8192 && N <= 8192)
        return impl_s8u4b8f16voidf32f32f16s32_sch_128x128_2x1x1_TmaMI__TmaCoop_streamK(args);
    if (M > 128)
        return impl_s8u4b8f16voidf32f32f16s32_sch_128x128_2x1x1_TmaMI__TmaCoop_streamK(args);
    if (M > 64 && K <= 4069 && N <= 4069)
        return impl_s8u4b8f16voidf32f32f16s32_sch_128x32_2x1x1_TmaMI__TmaCoop_streamK(args);
    if (M > 64 && K <= 4069 && N <= 8192)
        return impl_s8u4b8f16voidf32f32f16s32_sch_128x64_2x1x1_TmaMI__TmaCoop_streamK(args);
    if (M > 64 && K >= 8192 && N >= 12288)
        return impl_s8u4b8f16voidf32f32f16s32_sch_256x128_2x1x1_TmaMI__TmaCoop_streamK(args);
    if (M > 64)
        return impl_s8u4b8f16voidf32f32f16s32_sch_128x128_2x1x1_TmaMI__TmaCoop_streamK(args);
    if (M > 32 && K <= 6144 && N <= 6144)
        return impl_s8u4b8f16voidf32f32f16s32_sch_128x16_1x1x1_TmaMI__TmaCoop_streamK(args);
    if (M > 32)
        return impl_s8u4b8f16voidf32f32f16s32_sch_128x64_2x1x1_TmaMI__TmaCoop_streamK(args);
    if (M > 16 && K <= 12288 && N <= 8192)
        return impl_s8u4b8f16voidf32f32f16s32_sch_128x32_2x1x1_TmaMI__TmaCoop_streamK(args);
    if (M > 16)
        return impl_s8u4b8f16voidf32f32f16s32_sch_256x32_2x1x1_TmaMI__TmaCoop_streamK(args);
    if (N >= 26624)
        return impl_s8u4b8f16voidf32f32f16s32_sch_256x16_1x1x1_TmaMI__TmaCoop_streamK(args);
    else
        return impl_s8u4b8f16voidf32f32f16s32_sch_128x16_1x1x1_TmaMI__TmaCoop_streamK(args);
  }
  if (*args.maybe_schedule == "128x128_2x1x1_TmaMI__TmaCoop_streamK")
    return impl_s8u4b8f16voidf32f32f16s32_sch_128x128_2x1x1_TmaMI__TmaCoop_streamK(args);
  if (*args.maybe_schedule == "128x64_2x1x1_TmaMI__TmaCoop_streamK")
    return impl_s8u4b8f16voidf32f32f16s32_sch_128x64_2x1x1_TmaMI__TmaCoop_streamK(args);
  if (*args.maybe_schedule == "128x32_2x1x1_TmaMI__TmaCoop_streamK")
    return impl_s8u4b8f16voidf32f32f16s32_sch_128x32_2x1x1_TmaMI__TmaCoop_streamK(args);
  if (*args.maybe_schedule == "256x128_2x1x1_TmaMI__TmaCoop_streamK")
    return impl_s8u4b8f16voidf32f32f16s32_sch_256x128_2x1x1_TmaMI__TmaCoop_streamK(args);
  if (*args.maybe_schedule == "128x16_1x1x1_TmaMI__TmaCoop_streamK")
    return impl_s8u4b8f16voidf32f32f16s32_sch_128x16_1x1x1_TmaMI__TmaCoop_streamK(args);
  if (*args.maybe_schedule == "256x32_2x1x1_TmaMI__TmaCoop_streamK")
    return impl_s8u4b8f16voidf32f32f16s32_sch_256x32_2x1x1_TmaMI__TmaCoop_streamK(args);
  if (*args.maybe_schedule == "256x16_1x1x1_TmaMI__TmaCoop_streamK")
    return impl_s8u4b8f16voidf32f32f16s32_sch_256x16_1x1x1_TmaMI__TmaCoop_streamK(args);
  PADDLE_ENFORCE(false, "machete_gemm(..) is not implemented for "
                                     "schedule = %s", *args.maybe_schedule);
}

extern paddle::Tensor impl_s8u4b8voidvoidf32f32f16s32_sch_128x128_2x1x1_TmaMI__TmaCoop_streamK(MMArgs);
extern paddle::Tensor impl_s8u4b8voidvoidf32f32f16s32_sch_128x64_2x1x1_TmaMI__TmaCoop_streamK(MMArgs);
extern paddle::Tensor impl_s8u4b8voidvoidf32f32f16s32_sch_128x32_2x1x1_TmaMI__TmaCoop_streamK(MMArgs);
extern paddle::Tensor impl_s8u4b8voidvoidf32f32f16s32_sch_256x128_2x1x1_TmaMI__TmaCoop_streamK(MMArgs);
extern paddle::Tensor impl_s8u4b8voidvoidf32f32f16s32_sch_128x16_1x1x1_TmaMI__TmaCoop_streamK(MMArgs);
extern paddle::Tensor impl_s8u4b8voidvoidf32f32f16s32_sch_256x32_2x1x1_TmaMI__TmaCoop_streamK(MMArgs);
extern paddle::Tensor impl_s8u4b8voidvoidf32f32f16s32_sch_256x16_1x1x1_TmaMI__TmaCoop_streamK(MMArgs);

paddle::Tensor mm_dispatch_s8u4b8voidvoidf32f32f16s32(MMArgs args) {
  [[maybe_unused]] auto M = args.A.shape()[0];
  [[maybe_unused]] auto N = args.B.shape()[1];
  [[maybe_unused]] auto K = args.A.shape()[1];

  if (!args.maybe_schedule) {
    if (M > 256)
        return impl_s8u4b8voidvoidf32f32f16s32_sch_128x128_2x1x1_TmaMI__TmaCoop_streamK(args);
    if (M > 128 && K <= 4096 && N <= 4096)
        return impl_s8u4b8voidvoidf32f32f16s32_sch_128x64_2x1x1_TmaMI__TmaCoop_streamK(args);
    if (M > 128 && K <= 8192 && N <= 8192)
        return impl_s8u4b8voidvoidf32f32f16s32_sch_128x128_2x1x1_TmaMI__TmaCoop_streamK(args);
    if (M > 128)
        return impl_s8u4b8voidvoidf32f32f16s32_sch_128x128_2x1x1_TmaMI__TmaCoop_streamK(args);
    if (M > 64 && K <= 4069 && N <= 4069)
        return impl_s8u4b8voidvoidf32f32f16s32_sch_128x32_2x1x1_TmaMI__TmaCoop_streamK(args);
    if (M > 64 && K <= 4069 && N <= 8192)
        return impl_s8u4b8voidvoidf32f32f16s32_sch_128x64_2x1x1_TmaMI__TmaCoop_streamK(args);
    if (M > 64 && K >= 8192 && N >= 12288)
        return impl_s8u4b8voidvoidf32f32f16s32_sch_256x128_2x1x1_TmaMI__TmaCoop_streamK(args);
    if (M > 64)
        return impl_s8u4b8voidvoidf32f32f16s32_sch_128x128_2x1x1_TmaMI__TmaCoop_streamK(args);
    if (M > 32 && K <= 6144 && N <= 6144)
        return impl_s8u4b8voidvoidf32f32f16s32_sch_128x16_1x1x1_TmaMI__TmaCoop_streamK(args);
    if (M > 32)
        return impl_s8u4b8voidvoidf32f32f16s32_sch_128x64_2x1x1_TmaMI__TmaCoop_streamK(args);
    if (M > 16 && K <= 12288 && N <= 8192)
        return impl_s8u4b8voidvoidf32f32f16s32_sch_128x32_2x1x1_TmaMI__TmaCoop_streamK(args);
    if (M > 16)
        return impl_s8u4b8voidvoidf32f32f16s32_sch_256x32_2x1x1_TmaMI__TmaCoop_streamK(args);
    if (N >= 26624)
        return impl_s8u4b8voidvoidf32f32f16s32_sch_256x16_1x1x1_TmaMI__TmaCoop_streamK(args);
    else
        return impl_s8u4b8voidvoidf32f32f16s32_sch_128x16_1x1x1_TmaMI__TmaCoop_streamK(args);
  }
  if (*args.maybe_schedule == "128x128_2x1x1_TmaMI__TmaCoop_streamK")
    return impl_s8u4b8voidvoidf32f32f16s32_sch_128x128_2x1x1_TmaMI__TmaCoop_streamK(args);
  if (*args.maybe_schedule == "128x64_2x1x1_TmaMI__TmaCoop_streamK")
    return impl_s8u4b8voidvoidf32f32f16s32_sch_128x64_2x1x1_TmaMI__TmaCoop_streamK(args);
  if (*args.maybe_schedule == "128x32_2x1x1_TmaMI__TmaCoop_streamK")
    return impl_s8u4b8voidvoidf32f32f16s32_sch_128x32_2x1x1_TmaMI__TmaCoop_streamK(args);
  if (*args.maybe_schedule == "256x128_2x1x1_TmaMI__TmaCoop_streamK")
    return impl_s8u4b8voidvoidf32f32f16s32_sch_256x128_2x1x1_TmaMI__TmaCoop_streamK(args);
  if (*args.maybe_schedule == "128x16_1x1x1_TmaMI__TmaCoop_streamK")
    return impl_s8u4b8voidvoidf32f32f16s32_sch_128x16_1x1x1_TmaMI__TmaCoop_streamK(args);
  if (*args.maybe_schedule == "256x32_2x1x1_TmaMI__TmaCoop_streamK")
    return impl_s8u4b8voidvoidf32f32f16s32_sch_256x32_2x1x1_TmaMI__TmaCoop_streamK(args);
  if (*args.maybe_schedule == "256x16_1x1x1_TmaMI__TmaCoop_streamK")
    return impl_s8u4b8voidvoidf32f32f16s32_sch_256x16_1x1x1_TmaMI__TmaCoop_streamK(args);
  PADDLE_ENFORCE(false, "machete_gemm(..) is not implemented for "
                                     "schedule = %s", *args.maybe_schedule);
}

extern paddle::Tensor impl_e4m3u4b8f16voidf32f32f16f32_sch_128x128_2x1x1_TmaMI__TmaCoop_streamK(MMArgs);
extern paddle::Tensor impl_e4m3u4b8f16voidf32f32f16f32_sch_128x64_2x1x1_TmaMI__TmaCoop_streamK(MMArgs);
extern paddle::Tensor impl_e4m3u4b8f16voidf32f32f16f32_sch_128x32_2x1x1_TmaMI__TmaCoop_streamK(MMArgs);
extern paddle::Tensor impl_e4m3u4b8f16voidf32f32f16f32_sch_256x128_2x1x1_TmaMI__TmaCoop_streamK(MMArgs);
extern paddle::Tensor impl_e4m3u4b8f16voidf32f32f16f32_sch_128x16_1x1x1_TmaMI__TmaCoop_streamK(MMArgs);
extern paddle::Tensor impl_e4m3u4b8f16voidf32f32f16f32_sch_256x32_2x1x1_TmaMI__TmaCoop_streamK(MMArgs);
extern paddle::Tensor impl_e4m3u4b8f16voidf32f32f16f32_sch_256x16_1x1x1_TmaMI__TmaCoop_streamK(MMArgs);

paddle::Tensor mm_dispatch_e4m3u4b8f16voidf32f32f16f32(MMArgs args) {
  [[maybe_unused]] auto M = args.A.shape()[0];
  [[maybe_unused]] auto N = args.B.shape()[1];
  [[maybe_unused]] auto K = args.A.shape()[1];

  if (!args.maybe_schedule) {
    if (M > 256)
        return impl_e4m3u4b8f16voidf32f32f16f32_sch_128x128_2x1x1_TmaMI__TmaCoop_streamK(args);
    if (M > 128 && K <= 4096 && N <= 4096)
        return impl_e4m3u4b8f16voidf32f32f16f32_sch_128x64_2x1x1_TmaMI__TmaCoop_streamK(args);
    if (M > 128 && K <= 8192 && N <= 8192)
        return impl_e4m3u4b8f16voidf32f32f16f32_sch_128x128_2x1x1_TmaMI__TmaCoop_streamK(args);
    if (M > 128)
        return impl_e4m3u4b8f16voidf32f32f16f32_sch_128x128_2x1x1_TmaMI__TmaCoop_streamK(args);
    if (M > 64 && K <= 4069 && N <= 4069)
        return impl_e4m3u4b8f16voidf32f32f16f32_sch_128x32_2x1x1_TmaMI__TmaCoop_streamK(args);
    if (M > 64 && K <= 4069 && N <= 8192)
        return impl_e4m3u4b8f16voidf32f32f16f32_sch_128x64_2x1x1_TmaMI__TmaCoop_streamK(args);
    if (M > 64 && K >= 8192 && N >= 12288)
        return impl_e4m3u4b8f16voidf32f32f16f32_sch_256x128_2x1x1_TmaMI__TmaCoop_streamK(args);
    if (M > 64)
        return impl_e4m3u4b8f16voidf32f32f16f32_sch_128x128_2x1x1_TmaMI__TmaCoop_streamK(args);
    if (M > 32 && K <= 6144 && N <= 6144)
        return impl_e4m3u4b8f16voidf32f32f16f32_sch_128x16_1x1x1_TmaMI__TmaCoop_streamK(args);
    if (M > 32)
        return impl_e4m3u4b8f16voidf32f32f16f32_sch_128x64_2x1x1_TmaMI__TmaCoop_streamK(args);
    if (M > 16 && K <= 12288 && N <= 8192)
        return impl_e4m3u4b8f16voidf32f32f16f32_sch_128x32_2x1x1_TmaMI__TmaCoop_streamK(args);
    if (M > 16)
        return impl_e4m3u4b8f16voidf32f32f16f32_sch_256x32_2x1x1_TmaMI__TmaCoop_streamK(args);
    if (N >= 26624)
        return impl_e4m3u4b8f16voidf32f32f16f32_sch_256x16_1x1x1_TmaMI__TmaCoop_streamK(args);
    else
        return impl_e4m3u4b8f16voidf32f32f16f32_sch_128x16_1x1x1_TmaMI__TmaCoop_streamK(args);
  }
  if (*args.maybe_schedule == "128x128_2x1x1_TmaMI__TmaCoop_streamK")
    return impl_e4m3u4b8f16voidf32f32f16f32_sch_128x128_2x1x1_TmaMI__TmaCoop_streamK(args);
  if (*args.maybe_schedule == "128x64_2x1x1_TmaMI__TmaCoop_streamK")
    return impl_e4m3u4b8f16voidf32f32f16f32_sch_128x64_2x1x1_TmaMI__TmaCoop_streamK(args);
  if (*args.maybe_schedule == "128x32_2x1x1_TmaMI__TmaCoop_streamK")
    return impl_e4m3u4b8f16voidf32f32f16f32_sch_128x32_2x1x1_TmaMI__TmaCoop_streamK(args);
  if (*args.maybe_schedule == "256x128_2x1x1_TmaMI__TmaCoop_streamK")
    return impl_e4m3u4b8f16voidf32f32f16f32_sch_256x128_2x1x1_TmaMI__TmaCoop_streamK(args);
  if (*args.maybe_schedule == "128x16_1x1x1_TmaMI__TmaCoop_streamK")
    return impl_e4m3u4b8f16voidf32f32f16f32_sch_128x16_1x1x1_TmaMI__TmaCoop_streamK(args);
  if (*args.maybe_schedule == "256x32_2x1x1_TmaMI__TmaCoop_streamK")
    return impl_e4m3u4b8f16voidf32f32f16f32_sch_256x32_2x1x1_TmaMI__TmaCoop_streamK(args);
  if (*args.maybe_schedule == "256x16_1x1x1_TmaMI__TmaCoop_streamK")
    return impl_e4m3u4b8f16voidf32f32f16f32_sch_256x16_1x1x1_TmaMI__TmaCoop_streamK(args);
  PADDLE_ENFORCE(false, "machete_gemm(..) is not implemented for "
                                     "schedule = %s", *args.maybe_schedule);
}

extern paddle::Tensor impl_e4m3u4b8voidvoidf32f32f16f32_sch_128x128_2x1x1_TmaMI__TmaCoop_streamK(MMArgs);
extern paddle::Tensor impl_e4m3u4b8voidvoidf32f32f16f32_sch_128x64_2x1x1_TmaMI__TmaCoop_streamK(MMArgs);
extern paddle::Tensor impl_e4m3u4b8voidvoidf32f32f16f32_sch_128x32_2x1x1_TmaMI__TmaCoop_streamK(MMArgs);
extern paddle::Tensor impl_e4m3u4b8voidvoidf32f32f16f32_sch_256x128_2x1x1_TmaMI__TmaCoop_streamK(MMArgs);
extern paddle::Tensor impl_e4m3u4b8voidvoidf32f32f16f32_sch_128x16_1x1x1_TmaMI__TmaCoop_streamK(MMArgs);
extern paddle::Tensor impl_e4m3u4b8voidvoidf32f32f16f32_sch_256x32_2x1x1_TmaMI__TmaCoop_streamK(MMArgs);
extern paddle::Tensor impl_e4m3u4b8voidvoidf32f32f16f32_sch_256x16_1x1x1_TmaMI__TmaCoop_streamK(MMArgs);

paddle::Tensor mm_dispatch_e4m3u4b8voidvoidf32f32f16f32(MMArgs args) {
  [[maybe_unused]] auto M = args.A.shape()[0];
  [[maybe_unused]] auto N = args.B.shape()[1];
  [[maybe_unused]] auto K = args.A.shape()[1];

  if (!args.maybe_schedule) {
    if (M > 256)
        return impl_e4m3u4b8voidvoidf32f32f16f32_sch_128x128_2x1x1_TmaMI__TmaCoop_streamK(args);
    if (M > 128 && K <= 4096 && N <= 4096)
        return impl_e4m3u4b8voidvoidf32f32f16f32_sch_128x64_2x1x1_TmaMI__TmaCoop_streamK(args);
    if (M > 128 && K <= 8192 && N <= 8192)
        return impl_e4m3u4b8voidvoidf32f32f16f32_sch_128x128_2x1x1_TmaMI__TmaCoop_streamK(args);
    if (M > 128)
        return impl_e4m3u4b8voidvoidf32f32f16f32_sch_128x128_2x1x1_TmaMI__TmaCoop_streamK(args);
    if (M > 64 && K <= 4069 && N <= 4069)
        return impl_e4m3u4b8voidvoidf32f32f16f32_sch_128x32_2x1x1_TmaMI__TmaCoop_streamK(args);
    if (M > 64 && K <= 4069 && N <= 8192)
        return impl_e4m3u4b8voidvoidf32f32f16f32_sch_128x64_2x1x1_TmaMI__TmaCoop_streamK(args);
    if (M > 64 && K >= 8192 && N >= 12288)
        return impl_e4m3u4b8voidvoidf32f32f16f32_sch_256x128_2x1x1_TmaMI__TmaCoop_streamK(args);
    if (M > 64)
        return impl_e4m3u4b8voidvoidf32f32f16f32_sch_128x128_2x1x1_TmaMI__TmaCoop_streamK(args);
    if (M > 32 && K <= 6144 && N <= 6144)
        return impl_e4m3u4b8voidvoidf32f32f16f32_sch_128x16_1x1x1_TmaMI__TmaCoop_streamK(args);
    if (M > 32)
        return impl_e4m3u4b8voidvoidf32f32f16f32_sch_128x64_2x1x1_TmaMI__TmaCoop_streamK(args);
    if (M > 16 && K <= 12288 && N <= 8192)
        return impl_e4m3u4b8voidvoidf32f32f16f32_sch_128x32_2x1x1_TmaMI__TmaCoop_streamK(args);
    if (M > 16)
        return impl_e4m3u4b8voidvoidf32f32f16f32_sch_256x32_2x1x1_TmaMI__TmaCoop_streamK(args);
    if (N >= 26624)
        return impl_e4m3u4b8voidvoidf32f32f16f32_sch_256x16_1x1x1_TmaMI__TmaCoop_streamK(args);
    else
        return impl_e4m3u4b8voidvoidf32f32f16f32_sch_128x16_1x1x1_TmaMI__TmaCoop_streamK(args);
  }
  if (*args.maybe_schedule == "128x128_2x1x1_TmaMI__TmaCoop_streamK")
    return impl_e4m3u4b8voidvoidf32f32f16f32_sch_128x128_2x1x1_TmaMI__TmaCoop_streamK(args);
  if (*args.maybe_schedule == "128x64_2x1x1_TmaMI__TmaCoop_streamK")
    return impl_e4m3u4b8voidvoidf32f32f16f32_sch_128x64_2x1x1_TmaMI__TmaCoop_streamK(args);
  if (*args.maybe_schedule == "128x32_2x1x1_TmaMI__TmaCoop_streamK")
    return impl_e4m3u4b8voidvoidf32f32f16f32_sch_128x32_2x1x1_TmaMI__TmaCoop_streamK(args);
  if (*args.maybe_schedule == "256x128_2x1x1_TmaMI__TmaCoop_streamK")
    return impl_e4m3u4b8voidvoidf32f32f16f32_sch_256x128_2x1x1_TmaMI__TmaCoop_streamK(args);
  if (*args.maybe_schedule == "128x16_1x1x1_TmaMI__TmaCoop_streamK")
    return impl_e4m3u4b8voidvoidf32f32f16f32_sch_128x16_1x1x1_TmaMI__TmaCoop_streamK(args);
  if (*args.maybe_schedule == "256x32_2x1x1_TmaMI__TmaCoop_streamK")
    return impl_e4m3u4b8voidvoidf32f32f16f32_sch_256x32_2x1x1_TmaMI__TmaCoop_streamK(args);
  if (*args.maybe_schedule == "256x16_1x1x1_TmaMI__TmaCoop_streamK")
    return impl_e4m3u4b8voidvoidf32f32f16f32_sch_256x16_1x1x1_TmaMI__TmaCoop_streamK(args);
  PADDLE_ENFORCE(false, "machete_gemm(..) is not implemented for "
                                     "schedule = %s", *args.maybe_schedule);
}


static inline std::optional<paddle::DataType> maybe_scalartype(
    std::optional<paddle::Tensor> const& t) {
    if (!t) {
      return std::nullopt;
    } else {
      return t->dtype();
    };
}

paddle::Tensor mm_dispatch(MMArgs args) {
  auto out_type = args.maybe_out_type.value_or(args.A.dtype());
  auto a_type = args.A.dtype();
  auto maybe_g_scales_type = maybe_scalartype(args.maybe_group_scales);
  auto maybe_g_zeros_type = maybe_scalartype(args.maybe_group_zeros);
  auto maybe_ch_scales_type = maybe_scalartype(args.maybe_channel_scales);
  auto maybe_tok_scales_type = maybe_scalartype(args.maybe_token_scales);


  if (args.b_type == machete::kU4B8
      && a_type == paddle::DataType::FLOAT16
      && out_type == paddle::DataType::FLOAT16
      && maybe_g_scales_type == paddle::DataType::FLOAT16
      && !maybe_g_zeros_type
      && !maybe_ch_scales_type
      && !maybe_tok_scales_type
  ) {
      return mm_dispatch_f16u4b8f16voidvoidvoidf16f32(args);
  }
  if (args.b_type == machete::kU4B8
      && a_type == paddle::DataType::BFLOAT16
      && out_type == paddle::DataType::BFLOAT16
      && maybe_g_scales_type == paddle::DataType::BFLOAT16
      && !maybe_g_zeros_type
      && !maybe_ch_scales_type
      && !maybe_tok_scales_type
  ) {
      return mm_dispatch_bf16u4b8bf16voidvoidvoidbf16f32(args);
  }
  if (args.b_type == machete::kU8B128
      && a_type == paddle::DataType::FLOAT16
      && out_type == paddle::DataType::FLOAT16
      && maybe_g_scales_type == paddle::DataType::FLOAT16
      && !maybe_g_zeros_type
      && !maybe_ch_scales_type
      && !maybe_tok_scales_type
  ) {
      return mm_dispatch_f16u8b128f16voidvoidvoidf16f32(args);
  }
  if (args.b_type == machete::kU8B128
      && a_type == paddle::DataType::BFLOAT16
      && out_type == paddle::DataType::BFLOAT16
      && maybe_g_scales_type == paddle::DataType::BFLOAT16
      && !maybe_g_zeros_type
      && !maybe_ch_scales_type
      && !maybe_tok_scales_type
  ) {
      return mm_dispatch_bf16u8b128bf16voidvoidvoidbf16f32(args);
  }
  if (args.b_type == machete::kU4
      && a_type == paddle::DataType::FLOAT16
      && out_type == paddle::DataType::FLOAT16
      && maybe_g_scales_type == paddle::DataType::FLOAT16
      && maybe_g_zeros_type == paddle::DataType::FLOAT16
      && !maybe_ch_scales_type
      && !maybe_tok_scales_type
  ) {
      return mm_dispatch_f16u4f16f16voidvoidf16f32(args);
  }
  if (args.b_type == machete::kU4
      && a_type == paddle::DataType::BFLOAT16
      && out_type == paddle::DataType::BFLOAT16
      && maybe_g_scales_type == paddle::DataType::BFLOAT16
      && maybe_g_zeros_type == paddle::DataType::BFLOAT16
      && !maybe_ch_scales_type
      && !maybe_tok_scales_type
  ) {
      return mm_dispatch_bf16u4bf16bf16voidvoidbf16f32(args);
  }
  if (args.b_type == machete::kU8
      && a_type == paddle::DataType::FLOAT16
      && out_type == paddle::DataType::FLOAT16
      && maybe_g_scales_type == paddle::DataType::FLOAT16
      && maybe_g_zeros_type == paddle::DataType::FLOAT16
      && !maybe_ch_scales_type
      && !maybe_tok_scales_type
  ) {
      return mm_dispatch_f16u8f16f16voidvoidf16f32(args);
  }
  if (args.b_type == machete::kU8
      && a_type == paddle::DataType::BFLOAT16
      && out_type == paddle::DataType::BFLOAT16
      && maybe_g_scales_type == paddle::DataType::BFLOAT16
      && maybe_g_zeros_type == paddle::DataType::BFLOAT16
      && !maybe_ch_scales_type
      && !maybe_tok_scales_type
  ) {
      return mm_dispatch_bf16u8bf16bf16voidvoidbf16f32(args);
  }
  if (args.b_type == machete::kU4B8
      && a_type == paddle::DataType::INT8
      && out_type == paddle::DataType::FLOAT16
      && maybe_g_scales_type == paddle::DataType::FLOAT16
      && !maybe_g_zeros_type
      && maybe_ch_scales_type == paddle::DataType::FLOAT32
      && maybe_tok_scales_type == paddle::DataType::FLOAT32
  ) {
      return mm_dispatch_s8u4b8f16voidf32f32f16s32(args);
  }
  if (args.b_type == machete::kU4B8
      && a_type == paddle::DataType::INT8
      && out_type == paddle::DataType::FLOAT16
      && !maybe_g_scales_type
      && !maybe_g_zeros_type
      && maybe_ch_scales_type == paddle::DataType::FLOAT32
      && maybe_tok_scales_type == paddle::DataType::FLOAT32
  ) {
      return mm_dispatch_s8u4b8voidvoidf32f32f16s32(args);
  }
  if (args.b_type == machete::kU4B8
      && a_type == paddle::DataType::FLOAT8_E4M3FN
      && out_type == paddle::DataType::FLOAT16
      && maybe_g_scales_type == paddle::DataType::FLOAT16
      && !maybe_g_zeros_type
      && maybe_ch_scales_type == paddle::DataType::FLOAT32
      && maybe_tok_scales_type == paddle::DataType::FLOAT32
  ) {
      return mm_dispatch_e4m3u4b8f16voidf32f32f16f32(args);
  }
  if (args.b_type == machete::kU4B8
      && a_type == paddle::DataType::FLOAT8_E4M3FN
      && out_type == paddle::DataType::FLOAT16
      && !maybe_g_scales_type
      && !maybe_g_zeros_type
      && maybe_ch_scales_type == paddle::DataType::FLOAT32
      && maybe_tok_scales_type == paddle::DataType::FLOAT32
  ) {
      return mm_dispatch_e4m3u4b8voidvoidf32f32f16f32(args);
  }

  PADDLE_ENFORCE(
    false, "machete_mm(..) is not implemented for "
    "a_type=", args.A.dtype(),
    ", b_type=", args.b_type.str(),
    ", out_type=", out_type,
    // ", with_group_scale_type=", maybe_g_scales_type
    //     ? toString(*maybe_g_scales_type) : "None",
    // ", with_group_zeropoint_type=", maybe_g_zeros_type
    //     ? toString(*maybe_g_zeros_type) : "None",
    // ", with_channel_scale_type=", maybe_ch_scales_type
    //     ? toString(*maybe_ch_scales_type) : "None",
    // ", with_token_scale_type=", maybe_tok_scales_type
    //     ? toString(*maybe_tok_scales_type) : "None",
    "; implemented types are: \n",
    "\ta_type=f16, b_type=u4b8, with_group_scale_type=f16, with_group_zeropoint_type=void, with_channel_scale_type=void, a_token_scale_type=void, out_type=f16, accumulator_type=f32\n",
    "\ta_type=bf16, b_type=u4b8, with_group_scale_type=bf16, with_group_zeropoint_type=void, with_channel_scale_type=void, a_token_scale_type=void, out_type=bf16, accumulator_type=f32\n",
    "\ta_type=f16, b_type=u8b128, with_group_scale_type=f16, with_group_zeropoint_type=void, with_channel_scale_type=void, a_token_scale_type=void, out_type=f16, accumulator_type=f32\n",
    "\ta_type=bf16, b_type=u8b128, with_group_scale_type=bf16, with_group_zeropoint_type=void, with_channel_scale_type=void, a_token_scale_type=void, out_type=bf16, accumulator_type=f32\n",
    "\ta_type=f16, b_type=u4, with_group_scale_type=f16, with_group_zeropoint_type=f16, with_channel_scale_type=void, a_token_scale_type=void, out_type=f16, accumulator_type=f32\n",
    "\ta_type=bf16, b_type=u4, with_group_scale_type=bf16, with_group_zeropoint_type=bf16, with_channel_scale_type=void, a_token_scale_type=void, out_type=bf16, accumulator_type=f32\n",
    "\ta_type=f16, b_type=u8, with_group_scale_type=f16, with_group_zeropoint_type=f16, with_channel_scale_type=void, a_token_scale_type=void, out_type=f16, accumulator_type=f32\n",
    "\ta_type=bf16, b_type=u8, with_group_scale_type=bf16, with_group_zeropoint_type=bf16, with_channel_scale_type=void, a_token_scale_type=void, out_type=bf16, accumulator_type=f32\n",
    "\ta_type=s8, b_type=u4b8, with_group_scale_type=f16, with_group_zeropoint_type=void, with_channel_scale_type=f32, a_token_scale_type=f32, out_type=f16, accumulator_type=s32\n",
    "\ta_type=s8, b_type=u4b8, with_group_scale_type=void, with_group_zeropoint_type=void, with_channel_scale_type=f32, a_token_scale_type=f32, out_type=f16, accumulator_type=s32\n",
    "\ta_type=e4m3, b_type=u4b8, with_group_scale_type=f16, with_group_zeropoint_type=void, with_channel_scale_type=f32, a_token_scale_type=f32, out_type=f16, accumulator_type=f32\n",
    "\ta_type=e4m3, b_type=u4b8, with_group_scale_type=void, with_group_zeropoint_type=void, with_channel_scale_type=f32, a_token_scale_type=f32, out_type=f16, accumulator_type=f32\n",
    "");
}

std::vector<std::string> supported_schedules_dispatch(
    SupportedSchedulesArgs args) {
    auto out_type = args.maybe_out_type.value_or(args.a_type);


    if (args.b_type == machete::kU4B8
        && args.a_type == paddle::DataType::FLOAT16
        && out_type == paddle::DataType::FLOAT16
        && args.maybe_group_scales_type == paddle::DataType::FLOAT16
        && !args.maybe_group_zeros_type
    ) {
        return {
            "128x128_2x1x1_TmaMI__TmaCoop_streamK",
            "128x256_2x1x1_TmaMI__TmaCoop_streamK",
            "128x64_2x1x1_TmaMI__TmaCoop_streamK",
            "128x32_2x1x1_TmaMI__TmaCoop_streamK",
            "256x128_2x1x1_TmaMI__TmaCoop_streamK",
            "128x16_1x1x1_TmaMI__TmaCoop_streamK",
            "256x64_2x1x1_TmaMI__TmaCoop_streamK",
            "256x32_2x1x1_TmaMI__TmaCoop_streamK",
            "256x16_1x1x1_TmaMI__TmaCoop_streamK"
        };
    }
    if (args.b_type == machete::kU4B8
        && args.a_type == paddle::DataType::BFLOAT16
        && out_type == paddle::DataType::BFLOAT16
        && args.maybe_group_scales_type == paddle::DataType::BFLOAT16
        && !args.maybe_group_zeros_type
    ) {
        return {
            "128x128_2x1x1_TmaMI__TmaCoop_streamK",
            "128x256_2x1x1_TmaMI__TmaCoop_streamK",
            "128x64_2x1x1_TmaMI__TmaCoop_streamK",
            "128x32_2x1x1_TmaMI__TmaCoop_streamK",
            "256x128_2x1x1_TmaMI__TmaCoop_streamK",
            "128x16_1x1x1_TmaMI__TmaCoop_streamK",
            "256x64_2x1x1_TmaMI__TmaCoop_streamK",
            "256x32_2x1x1_TmaMI__TmaCoop_streamK",
            "256x16_1x1x1_TmaMI__TmaCoop_streamK"
        };
    }
    if (args.b_type == machete::kU8B128
        && args.a_type == paddle::DataType::FLOAT16
        && out_type == paddle::DataType::FLOAT16
        && args.maybe_group_scales_type == paddle::DataType::FLOAT16
        && !args.maybe_group_zeros_type
    ) {
        return {
            "128x128_2x1x1_TmaMI__TmaCoop_streamK",
            "128x256_2x1x1_TmaMI__TmaCoop_streamK",
            "128x64_2x1x1_TmaMI__TmaCoop_streamK",
            "128x32_2x1x1_TmaMI__TmaCoop_streamK",
            "256x128_2x1x1_TmaMI__TmaCoop_streamK",
            "128x16_1x1x1_TmaMI__TmaCoop_streamK",
            "256x64_2x1x1_TmaMI__TmaCoop_streamK",
            "256x32_2x1x1_TmaMI__TmaCoop_streamK",
            "256x16_1x1x1_TmaMI__TmaCoop_streamK"
        };
    }
    if (args.b_type == machete::kU8B128
        && args.a_type == paddle::DataType::BFLOAT16
        && out_type == paddle::DataType::BFLOAT16
        && args.maybe_group_scales_type == paddle::DataType::BFLOAT16
        && !args.maybe_group_zeros_type
    ) {
        return {
            "128x128_2x1x1_TmaMI__TmaCoop_streamK",
            "128x256_2x1x1_TmaMI__TmaCoop_streamK",
            "128x64_2x1x1_TmaMI__TmaCoop_streamK",
            "128x32_2x1x1_TmaMI__TmaCoop_streamK",
            "256x128_2x1x1_TmaMI__TmaCoop_streamK",
            "128x16_1x1x1_TmaMI__TmaCoop_streamK",
            "256x64_2x1x1_TmaMI__TmaCoop_streamK",
            "256x32_2x1x1_TmaMI__TmaCoop_streamK",
            "256x16_1x1x1_TmaMI__TmaCoop_streamK"
        };
    }
    if (args.b_type == machete::kU4
        && args.a_type == paddle::DataType::FLOAT16
        && out_type == paddle::DataType::FLOAT16
        && args.maybe_group_scales_type == paddle::DataType::FLOAT16
        && args.maybe_group_zeros_type == paddle::DataType::FLOAT16
    ) {
        return {
            "128x128_2x1x1_TmaMI__TmaCoop_streamK",
            "128x256_2x1x1_TmaMI__TmaCoop_streamK",
            "128x64_2x1x1_TmaMI__TmaCoop_streamK",
            "128x32_2x1x1_TmaMI__TmaCoop_streamK",
            "256x128_2x1x1_TmaMI__TmaCoop_streamK",
            "128x16_1x1x1_TmaMI__TmaCoop_streamK",
            "256x64_2x1x1_TmaMI__TmaCoop_streamK",
            "256x32_2x1x1_TmaMI__TmaCoop_streamK",
            "256x16_1x1x1_TmaMI__TmaCoop_streamK"
        };
    }
    if (args.b_type == machete::kU4
        && args.a_type == paddle::DataType::BFLOAT16
        && out_type == paddle::DataType::BFLOAT16
        && args.maybe_group_scales_type == paddle::DataType::BFLOAT16
        && args.maybe_group_zeros_type == paddle::DataType::BFLOAT16
    ) {
        return {
            "128x128_2x1x1_TmaMI__TmaCoop_streamK",
            "128x256_2x1x1_TmaMI__TmaCoop_streamK",
            "128x64_2x1x1_TmaMI__TmaCoop_streamK",
            "128x32_2x1x1_TmaMI__TmaCoop_streamK",
            "256x128_2x1x1_TmaMI__TmaCoop_streamK",
            "128x16_1x1x1_TmaMI__TmaCoop_streamK",
            "256x64_2x1x1_TmaMI__TmaCoop_streamK",
            "256x32_2x1x1_TmaMI__TmaCoop_streamK",
            "256x16_1x1x1_TmaMI__TmaCoop_streamK"
        };
    }
    if (args.b_type == machete::kU8
        && args.a_type == paddle::DataType::FLOAT16
        && out_type == paddle::DataType::FLOAT16
        && args.maybe_group_scales_type == paddle::DataType::FLOAT16
        && args.maybe_group_zeros_type == paddle::DataType::FLOAT16
    ) {
        return {
            "128x128_2x1x1_TmaMI__TmaCoop_streamK",
            "128x256_2x1x1_TmaMI__TmaCoop_streamK",
            "128x64_2x1x1_TmaMI__TmaCoop_streamK",
            "128x32_2x1x1_TmaMI__TmaCoop_streamK",
            "256x128_2x1x1_TmaMI__TmaCoop_streamK",
            "128x16_1x1x1_TmaMI__TmaCoop_streamK",
            "256x64_2x1x1_TmaMI__TmaCoop_streamK",
            "256x32_2x1x1_TmaMI__TmaCoop_streamK",
            "256x16_1x1x1_TmaMI__TmaCoop_streamK"
        };
    }
    if (args.b_type == machete::kU8
        && args.a_type == paddle::DataType::BFLOAT16
        && out_type == paddle::DataType::BFLOAT16
        && args.maybe_group_scales_type == paddle::DataType::BFLOAT16
        && args.maybe_group_zeros_type == paddle::DataType::BFLOAT16
    ) {
        return {
            "128x128_2x1x1_TmaMI__TmaCoop_streamK",
            "128x256_2x1x1_TmaMI__TmaCoop_streamK",
            "128x64_2x1x1_TmaMI__TmaCoop_streamK",
            "128x32_2x1x1_TmaMI__TmaCoop_streamK",
            "256x128_2x1x1_TmaMI__TmaCoop_streamK",
            "128x16_1x1x1_TmaMI__TmaCoop_streamK",
            "256x64_2x1x1_TmaMI__TmaCoop_streamK",
            "256x32_2x1x1_TmaMI__TmaCoop_streamK",
            "256x16_1x1x1_TmaMI__TmaCoop_streamK"
        };
    }
    if (args.b_type == machete::kU4B8
        && args.a_type == paddle::DataType::INT8
        && out_type == paddle::DataType::FLOAT16
        && args.maybe_group_scales_type == paddle::DataType::FLOAT16
        && !args.maybe_group_zeros_type
    ) {
        return {
            "128x128_2x1x1_TmaMI__TmaCoop_streamK",
            "128x64_2x1x1_TmaMI__TmaCoop_streamK",
            "128x32_2x1x1_TmaMI__TmaCoop_streamK",
            "256x128_2x1x1_TmaMI__TmaCoop_streamK",
            "128x16_1x1x1_TmaMI__TmaCoop_streamK",
            "256x32_2x1x1_TmaMI__TmaCoop_streamK",
            "256x16_1x1x1_TmaMI__TmaCoop_streamK"
        };
    }
    if (args.b_type == machete::kU4B8
        && args.a_type == paddle::DataType::INT8
        && out_type == paddle::DataType::FLOAT16
        && !args.maybe_group_scales_type
        && !args.maybe_group_zeros_type
    ) {
        return {
            "128x128_2x1x1_TmaMI__TmaCoop_streamK",
            "128x64_2x1x1_TmaMI__TmaCoop_streamK",
            "128x32_2x1x1_TmaMI__TmaCoop_streamK",
            "256x128_2x1x1_TmaMI__TmaCoop_streamK",
            "128x16_1x1x1_TmaMI__TmaCoop_streamK",
            "256x32_2x1x1_TmaMI__TmaCoop_streamK",
            "256x16_1x1x1_TmaMI__TmaCoop_streamK"
        };
    }
    if (args.b_type == machete::kU4B8
        && args.a_type == paddle::DataType::FLOAT8_E4M3FN
        && out_type == paddle::DataType::FLOAT16
        && args.maybe_group_scales_type == paddle::DataType::FLOAT16
        && !args.maybe_group_zeros_type
    ) {
        return {
            "128x128_2x1x1_TmaMI__TmaCoop_streamK",
            "128x64_2x1x1_TmaMI__TmaCoop_streamK",
            "128x32_2x1x1_TmaMI__TmaCoop_streamK",
            "256x128_2x1x1_TmaMI__TmaCoop_streamK",
            "128x16_1x1x1_TmaMI__TmaCoop_streamK",
            "256x32_2x1x1_TmaMI__TmaCoop_streamK",
            "256x16_1x1x1_TmaMI__TmaCoop_streamK"
        };
    }
    if (args.b_type == machete::kU4B8
        && args.a_type == paddle::DataType::FLOAT8_E4M3FN
        && out_type == paddle::DataType::FLOAT16
        && !args.maybe_group_scales_type
        && !args.maybe_group_zeros_type
    ) {
        return {
            "128x128_2x1x1_TmaMI__TmaCoop_streamK",
            "128x64_2x1x1_TmaMI__TmaCoop_streamK",
            "128x32_2x1x1_TmaMI__TmaCoop_streamK",
            "256x128_2x1x1_TmaMI__TmaCoop_streamK",
            "128x16_1x1x1_TmaMI__TmaCoop_streamK",
            "256x32_2x1x1_TmaMI__TmaCoop_streamK",
            "256x16_1x1x1_TmaMI__TmaCoop_streamK"
        };
    }

    return {};
};

}; // namespace machete
