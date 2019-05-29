## Gif-h

## This is a fork of a fork

The original by Charlie Tangora (ctangora @ gmail)
https://github.com/ginsweater/gif-h

Fork #1 - https://github.com/rversteegen/gif-h
Substantial improvements to the quality of generated GIFs
 
+ better quantization when building a palette
+ support for 8-bit frames
+ support for global palette
+ support for adjusting delay of the last frame
+ support for varying frame sizes
+ support for customizing dithering 'strength'
+ kd-tree statistics collection
+ multiple fixes

Fork #2 - https://github.com/apankrat/gif-h
Mostly structural and cosmetic improvements to the code
(you are here)

+ converted all file and heap operations into callbacks, this removed dependency on <stdio.h>
+ converted all 3 API entries to void functions because they can only fail in a callback and the calling code already knows about these failures
+ converted input argument checks to asserts - no reason to gracefully handle junk being passed in
+ rearranged code a bit - grouped macros, structs and functions
+ switched to using from int/uint32_t to size_t where appropriate
+ reworked code to preallocate and reuse buffers
+ ported over GIF_FLIP_VERT patch
+ simplifed and tightened function argument lists a bit, passing GifWriter* instead where appropriate
+ trivial cosmetic changes

Also:

+ removed support for 8-bit frames
+ removed support for varying frame sizes
+ removed support for global palette

Removed parts weren't just something that was needed in the project that this fork was for, and removing them led to a simpler API and code.
