# vkcontrast POC

How fast can RAWmakase's Basic tone sliders update a RAW photo on this laptop (Intel
UHD Graphics 620, i5-8365U, Linux/Wayland)? A proof of concept in C++, Vulkan 1.3,
LibRaw, GLFW and Dear ImGui. It ports RAWmakase's engine 4 Exposure, Contrast,
Highlights, Shadows, Whites and Blacks to the GPU and runs them on the photo where
RAWmakase showed no frame while a slider was dragged.

## Test photo

The photo used for the measurements below (Nikon Z f DNG, 6064 × 4040, 26.6 MB) is
attached to the release:
[DSC_004600041.dng](https://github.com/lszl84/rawmakase-tone-gpu-poc/releases/download/v0.1/DSC_004600041.dng).
Its camera and lens serial numbers were removed; the image data is unchanged.

To reproduce the RAWmakase side, import it into RAWmakase, set Contrast −76 and
Highlights −3 (the edit used here), and pass that catalog to the POC with
`--catalog`, which gives it the photo's profile tone curve and saved sliders.

## What is ported

From `crates/rawmakase-engine/src/develop/` of the RAWmakase checkout:

- **Tone stage** (`pipeline/pixel.rs`): Exposure and the DNG black ramp in Rec. 2020,
  then the camera profile's tone curve as DNG RGBTone in ProPhoto
  (`camera_profiles` `finish`).
- **Shadows and Highlights** (`local_tone.rs`): the guided filter of log2 luminance
  on the photo reduced to 512 pixels (radius 3.2% of the long edge, ε 1.5), the
  image keys (luminance percentiles), and the measured gain tables.
- **Contrast, Whites, Blacks** (`basic_tone.rs`): the composed 1025-point curve from
  the measured tables, applied as DNG RGBTone in encoded ProPhoto, with the photo's
  Contrast pivot (48-column blocks, fitted formula) and positive Whites following
  the photo's 98th percentile of encoded luminance.

The measured tables are read from the RAWmakase checkout at build time
(`tools/gen_tables.py`). The photo's profile tone curve, camera exposure and current
slider values are read from the RAWmakase catalog at startup, read-only; Adobe's
profile data is not stored here.

Not ported: the camera profile's colour tables (HueSatMap, LookTable) and RAWmakase's
own camera matrix; LibRaw's camera-to-sRGB conversion stands in for them, so colours
differ slightly from RAWmakase. Nothing after the Basic curve is ported (this edit
uses none of it).

## How it runs

Everything runs on the GPU, with no readbacks:

| Pass | What |
| --- | --- |
| tone the small copy | the 512-pixel copy through the tone stage: luminance, log2, encoded |
| select | exact percentiles by radix select (4 rounds of 8-bit histograms) |
| guided filter | box means in 16 × 16 shared-memory tiles with the halo, rows then columns, twice |
| pivot | blocks, then a bitonic sort in shared memory for the 1st and 99th percentile |
| Basic curve | the 1025-point curve, made monotone |
| photo pass | per screen pixel: tone stage, Shadows/Highlights gain, Basic curve, sRGB |

Shadows' and Highlights' 48-bin curves are built on the CPU per change (96 values),
as RAWmakase builds them. Zoom and resize resample the photo to the screen size once.

**Cache what a slider doesn't change** (on by default) keeps each result until what
it depends on changes:

- The map, keys and Whites highlights depend on Exposure only.
- The Contrast pivot depends on none of the six sliders.
- The Basic curve depends on Contrast, Whites, Blacks (and Exposure through
  Whites' highlights).
- The screen image after the tone stage depends on Exposure.

So moving Contrast, Highlights, Shadows, Whites or Blacks runs at most the curve and
the photo pass from the toned screen image. The result is the same as recomputing
everything. Off, every change recomputes what RAWmakase would: the measures it needs,
the curve, and the full per-pixel pass.

## Build and run

Needs `vulkan-headers`, `glfw` (3.4), `libraw`, `sqlite`, `shaderc` (`glslc`),
Python 3, CMake, Ninja, and a RAWmakase checkout (`-DRAWMAKASE_DIR`, default
`~/Developer/rawmakase`). Dear ImGui is a submodule.

```sh
git submodule update --init
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build
./build/vkcontrast PHOTO.dng --catalog ~/Pictures/Photos.rawmakase
```

`--catalog` points at the RAWmakase catalog holding the photo: the POC then starts
from the photo's saved edit and renders with its camera profile's tone curve.
Without it the sliders start at 0 and there is no profile tone curve, so the photo
looks flat.

Other options: `--full` (full AHD demosaic instead of half size), `--height N`
(photo height on screen in physical pixels; default 1402, the Fit size the photo
had in RAWmakase on the test laptop), `--no-cache` (start with the cache off),
`--check`.

Environment: `VKCONTRAST_BENCH=1` (every slider swept 120 steps in both modes),
`VKCONTRAST_PROFILE=1|cache|cache-exposure` (GPU time per pass),
`VKCONTRAST_VALIDATE=1`, `VKCONTRAST_DUMP=file.ppm`, `VKCONTRAST_PRESENT=fifo|immediate`.

`--check` renders with every slider set and compares the GPU's results with a
line-by-line C++ translation of the Rust code (`src/tone_ref.h`).

## Results — 2026-10-09

Nikon Z f DNG (6064 × 4040, half-size decode), Adobe Standard, the saved edit
(Contrast −76, Highlights −3), shown at 934 × 1402.

`--check` (Shadows +35, Whites +25, Blacks −20 added), both modes, validation layer
clean:

| | GPU | C++ reference |
| --- | ---: | ---: |
| Contrast pivot | 0.482422 | 0.482423 |
| Whites highlights | 0.848802 | 0.848801 |
| Shadows / Highlights keys | −0.479819 / −3.745609 | the same |
| Map a, b | max difference 4e-6, 2e-5 | |
| Basic curve | max difference 8e-7 | |
| Final colour, 26,934 pixels | at most 1/255 | |

`VKCONTRAST_BENCH=1`, three runs, with Firefox busy on the same CPU, GPU and power
budget, so the heavy paths vary between runs:

| Per slider change | Everything recomputed | Cache on |
| --- | ---: | ---: |
| Contrast, Highlights, Shadows, Whites, Blacks (GPU) | 6–36 ms | 2.6–3.3 ms |
| the same, change to frame presented | 12–45 ms | 3.3–4.8 ms (one run 11–15 ms) |
| Exposure (GPU) | 6–21 ms | 9–20 ms |

GPU time per pass with everything recomputed (`VKCONTRAST_PROFILE=1`, one run): tone
the small copy 0.26 ms, select 0.7 ms, guided filter 3.1 ms, pivot 0.5 ms, Basic
curve 0.2 ms, photo pass 2.6–3.2 ms.

For comparison, RAWmakase on the same photo and machine takes 235–314 ms per Fit
render, and showed no frame during a drag.

---

Written with [Claude Code](https://claude.com/claude-code). MIT licensed; Dear ImGui
(submodule) is MIT. The measured tone tables are RAWmakase's (MIT) and are read
from a checkout at build time, not copied here.
