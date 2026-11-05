# bethutil

This fork adds `btu::kf`, a custom Animation Compressor for Fallout New Vegas `.kf` files. The compressor core (`include/btu/kf/detail/{kfcompress_core,dedupe_types,bspline,byteio,niftypes}.hpp` and `src/kf/kfcompress_core.cpp`) is the author's own work and is licensed **GPL-2.0-only** ([COPYING-GPL-2.0.txt](COPYING-GPL-2.0.txt)). Everything else in this repository keeps its original [Mozilla Public License 2.0](LICENSE) headers; MPL-2.0 allows its files to be combined into a GPL-2.0 work, so binaries that include the compressor are distributed under GPL-2.0.
