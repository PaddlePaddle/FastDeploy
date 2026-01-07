#pragma once

#include <stdint.h>
#include <stdio.h>

#define VERSION_BIT_WIDTH 8U
#define MAJOR_SHIFT (3U * VERSION_BIT_WIDTH)
#define MINOR_SHIFT (2U * VERSION_BIT_WIDTH)
#define PATCH_SHIFT (1U * VERSION_BIT_WIDTH)
#define BUILD_SHIFT (0U * VERSION_BIT_WIDTH)

#define MACA_VERSION_TRANSFORM(major, minor, patch, build) \
  (((major) << MAJOR_SHIFT) | ((minor) << MINOR_SHIFT) |   \
   ((patch) << PATCH_SHIFT) | ((build) << BUILD_SHIFT))

#ifndef MACA_VERSION
#define MACA_VERSION MACA_VERSION_TRANSFORM(3, 3, 0, 11)
#endif

#define MACA_VERSION_LT(major, minor, patch, build) \
  (MACA_VERSION < MACA_VERSION_TRANSFORM(major, minor, patch, build)) && 1U

#define MACA_VERSION_LE(major, minor, patch, build) \
  (MACA_VERSION <= MACA_VERSION_TRANSFORM(major, minor, patch, build)) && 1U

#define MACA_VERSION_GE(major, minor, patch, build) \
  (MACA_VERSION >= MACA_VERSION_TRANSFORM(major, minor, patch, build)) && 1U

#define MACA_VERSION_GT(major, minor, patch, build) \
  (MACA_VERSION > MACA_VERSION_TRANSFORM(major, minor, patch, build)) && 1U

#define MACA_VERSION_EQ(major, minor, patch, build) \
  (MACA_VERSION == MACA_VERSION_TRANSFORM(major, minor, patch, build)) && 1U
