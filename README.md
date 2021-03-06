## Gif-h

## This is a fork of a fork

### The original

https://github.com/ginsweater/gif-h - by Charlie Tangora

### Fork #1 - https://github.com/rversteegen/gif-h - by Ralph Versteegen

Substantial algorithmical and feature improvements -> (much) better GIF quality
 
+ better quantization when building a palette
+ support for 8-bit frames
+ support for global palette
+ support for adjusting delay of the last frame
+ support for varying frame sizes
+ support for customizing dithering 'strength'
+ kd-tree statistics collection
+ multiple fixes

### Fork #2 - https://github.com/apankrat/gif-h - by yours truly

Mostly structural and cosmetic improvements to the code

+ converted all file and heap operations into callbacks, this removed dependency on <stdio.h>
+ reworked code to preallocate and reuse buffers
+ converted all 3 API entries to void functions because they can only fail in a callback and the calling code already knows about these failures
+ converted input argument checks to asserts - no reason to gracefully handle junk being passed in
+ rearranged code a bit - grouped together macros, types, structs and functions
+ switched to using from int/uint32_t to size_t where appropriate
+ ported over GIF_FLIP_VERT patch
+ simplifed and tightened function argument lists a bit, passing GifWriter* instead where appropriate
+ trivial cosmetic changes

Also:

+ removed support for 8-bit frames
+ removed support for varying frame sizes
+ removed support for global palette

Removed parts weren't just something that was needed in the project that this fork was for, and removing them led to a simpler API and code.
