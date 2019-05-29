/*
 *	https://github.com/ginsweater/gif-h
 *	The original by Charlie Tangora (ctangora @ gmail)
 *
 *	Fork #1 - https://github.com/rversteegen/gif-h
 *	Substantial improvements to the quality of generated GIFs
 *
 *	+ better quantization when building a palette
 *	+ support for 8-bit frames
 *	+ support for global palette
 *	+ support for adjusting delay of the last frame
 *	+ support for varying frame sizes
 *	+ support for customizing dithering 'strength'
 *	+ kd-tree statistics collection
 *	+ multiple fixes
 *
 *	Fork #2 - https://github.com/apankrat/gif-h
 *	Mostly structural and cosmetic improvements to the code
 *
 *	+ converted all file and heap operations into callbacks, this removed dependency on <stdio.h>
 *	+ converted all 3 API entries to void functions because they can only fail in a callback and the calling code already knows about these failures
 *	+ converted input argument checks to asserts - no reason to gracefully handle junk being passed in
 *	+ rearranged code a bit - grouped macros, structs and functions
 *	+ switched to using from int/uint32_t to size_t where appropriate
 *	+ reworked code to preallocate and reuse buffers
 *	+ ported over GIF_FLIP_VERT patch
 *	+ simplifed and tightened function argument lists a bit, passing GifWriter* instead where appropriate
 *	+ trivial cosmetic changes
 *	- removed support for 8-bit frames
 *	- removed support for varying frame sizes
 *	- removed support for global palette
 *
 *	Removed parts weren't just something that was needed in the project that 
 *	this fork was for, and removing them led to a simpler API and code.
 */

/*
 *	What follows is the comment from the original gif-h.h,
 *	which was expanded in rversteegen's fork.
 */

//
// gif.h
// by Charlie Tangora
// Public domain.
// Email me : ctangora -at- gmail -dot- com
//
// This file offers a simple, very limited way to create animated GIFs directly in code.
//
// Those looking for particular cleverness are likely to be disappointed; it's pretty
// much a straight-ahead implementation of the GIF format with optional Floyd-Steinberg
// dithering. (It does at least use delta encoding - only the changed portions of each
// frame are saved.)
//
// So resulting files are often quite large. The hope is that it will be handy nonetheless
// as a quick and easily-integrated way for programs to spit out animations.
//
// Only RGBA8 is currently supported as an input format. (The alpha is ignored.)
//
// If capturing a buffer with a bottom-left origin (such as OpenGL), define GIF_FLIP_VERT
// to automatically flip the buffer data when writing the image (the buffer itself is
// unchanged.
//
// USAGE:
// Create a GifWriter struct. Pass it to GifBegin() to initialize and write the header.
// Pass subsequent frames to GifWriteFrame().
// Finally, call GifEnd() to close the file handle and free memory.
//

#ifndef gif_h
#define gif_h

#include <string.h>  // for memcpy and bzero
#include <stdint.h>  // for integer typedefs
#include <stddef.h>  // for offsetof

/*
 *	macros
 */
#if 1
#  define GIF_ASSERT(x) (void)0
#else
#  include <assert.h>
#  define GIF_ASSERT assert
#endif

/*
 *	structs
 */

// Layout of a pixel for GifWriteFrame. You can reorder this, but must be the same as GifRGBA32.
struct GifRGBA
{
    uint8_t r, g, b, a;

    uint8_t & comps(int comp) { return ((uint8_t*)this)[comp]; }
};

struct GifRGBA32
{
    int32_t r, g, b, a;
};

struct GifPalette
{
    int bitDepth;  // log2 of the possible number of colors
//  int numColors; // The number of colors actually used (including transpColIndex)
    GifRGBA colors[256]; // alpha component is ignored
};

struct GifHeapQueue
{
    int len;
    struct qitem
    {
        int cost, nodeIndex;
    } items[256];
};

struct GifKDNode
{
    uint8_t splitComp : 7;     // Color component index (dimension) to split on (actually member offset)
    uint8_t isLeaf : 1;
    uint8_t palIndex;          // Leaf nodes only

    // The following are not used (and are uninitialised) in leaf nodes
    uint8_t splitVal;          // If the component color is >= this, it's in the right subtree.
    uint16_t left, right;      // Indices of children in GifKDTree.nodes[]

    // The following are used only when building the tree
    int firstPixel, lastPixel; // Range of pixels in the image contained in this box
    int cost;
};

// k-d tree over RGB space
struct GifKDTree
{
    int numNodes;
    GifKDNode nodes[512];
    GifPalette pal;
    GifHeapQueue queue;
};

// Simple structure to write out the LZW-compressed portion of the image
// one bit at a time
struct GifBitStatus
{
    uint8_t bitIndex;  // how many bits in the partial byte written so far
    uint8_t byte;      // current partial byte
    size_t chunkIndex;
    uint8_t chunk[256];   // bytes are written in here until we have 256 of them, then written to the file
};

// The LZW dictionary is a 256-ary tree constructed as the file is encoded,
// this is one node
struct GifLzwNode
{
    uint16_t m_next[256];
};

struct GifLzwTree
{
    GifLzwNode nodes[4096];
};

//
struct GifWriterCb
{
    virtual void write(const void * data, size_t size) = 0;

    virtual void * alloc(size_t bytes) = 0;
    virtual void free(void * ptr) = 0;

    void putc(unsigned char byte) { write(&byte, 1); }
    void puts(const char * str)   { write(str, strlen(str)); }
};

struct GifWriter
{
    GifWriterCb * cb;

    GifRGBA * frame;         // preallocated
    GifRGBA * spareFrame;    // preallocated

    size_t width;
    size_t height;
    size_t framePixels;      // width * height
    size_t frameBytes;       // width * height * sizeof(GifRGBA);
    bool   empty;            // if haven't added any frames yet

    GifLzwTree * codeTree;   // preallocated
    GifRGBA32 * errorPixels; // allocated on first use

    //
    int transpColIndex;      // 0, index of transparent color

    // Maximum amount of accumulated diffused error in each color component, for F-S dithering.
    // Can be set above 256, but don't go much higher.
    // Set to a low value like 64 to minimise color bleeding, or a very
    // low value like 16 to reduce the amount of dithering and noise.
    int maxAccumError; // 256

    //
    GifWriter()  { cb = NULL;       }
    ~GifWriter() { GIF_ASSERT(!cb); }
};

/*
 *	code
 */
int GifIMax(int l, int r) { return l>r?l:r; }
int GifIMin(int l, int r) { return l<r?l:r; }
int GifIAbs(int i) { return i<0?-i:i; }

uint8_t GifUI8Max(uint8_t l, uint8_t r) { return l>r?l:r; }
uint8_t GifUI8Min(uint8_t l, uint8_t r) { return l<r?l:r; }

//
void GifHeapPop( GifHeapQueue * q )
{
    GIF_ASSERT(q->len);

    // We pop off the last element and look where to put it (don't resize yet)
    int hole = 1;
    q->len--;
    GifHeapQueue::qitem * to_insert = &q->items[1 + q->len];

    // Bubble up the hole from root until finding a spot to put to_insert
    while ( hole*2 < 1 + q->len )
    {
        // Find the largest child
        int child = hole*2;  // left child
        if ( child + 1 < 1 + q->len && q->items[child + 1].cost > q->items[child].cost )
            child++;
        if ( to_insert->cost >= q->items[child].cost )
            break;
        q->items[hole] = q->items[child];
        hole = child;
    }
    q->items[hole] = *to_insert;
}

void GifHeapPush( GifHeapQueue * q, int cost, int key )
{
    // Start from end and bubble down the value until its parent isn't larger
    int hole = 1 + q->len;  // where to place

    q->len++;
    GIF_ASSERT(q->len < 511);

    while ( hole > 1 && q->items[hole/2].cost < cost )
    {
        q->items[hole] = q->items[hole/2];
        hole /= 2;
    }
    q->items[hole].cost = cost;
    q->items[hole].nodeIndex = key;
}

bool GifRGBEqual( GifRGBA pixA, GifRGBA pixB )
{
    return pixA.r == pixB.r && pixA.g == pixB.g && pixA.b == pixB.b;
}

// Check if two palettes have the same colours (k-d tree stuff ignored)
bool GifPalettesEqual( const GifPalette * pPal1, const GifPalette * pPal2 )
{
    return pPal1->bitDepth == pPal2->bitDepth &&
           !memcmp(pPal1->colors, pPal2->colors, sizeof(GifRGBA) * (1 << pPal1->bitDepth));
}

// Update bestDiff and return true if color 'ind' is closer to r,g,b than bestDiff.
bool GifBetterColorMatch(const GifPalette * pPal, int ind, GifRGBA color, int & bestDiff)
{
    int r_err = color.r - (int)pPal->colors[ind].r;
    int g_err = color.g - (int)pPal->colors[ind].g;
    int b_err = color.b - (int)pPal->colors[ind].b;
    int diff = 2*r_err*r_err + 4*g_err*g_err + 3*b_err*b_err;
    if (diff >= bestDiff)
        return false;
    bestDiff = diff;
    return true;
}

// walks the k-d tree to pick the palette entry for a desired color.
// Takes as in/out parameters the current best color and its error -
// only changes them if it finds a better color in its subtree.
// this is the major hotspot in the code at the moment.
void GifGetClosestPaletteColor(GifKDTree * tree, GifRGBA color, int & bestInd, int & bestDiff, int nodeIndex = 0)
{
    GifKDNode &node = tree->nodes[nodeIndex];

    // base case, reached the bottom of the tree
    if (node.isLeaf)
    {
        // check whether this color is better than the current winner
        if ( GifBetterColorMatch(&tree->pal, node.palIndex, color, bestDiff) )
            bestInd = node.palIndex;
        return;
    }

    // r g b -> 2 4 3
    int comp_mult = (0x030402 >> (node.splitComp << 8)) & 0xff;

    // Compare to the appropriate color component (r, g, or b) for this node of the k-d tree
    int comp = color.comps(node.splitComp);
    if (node.splitVal > comp)
    {
        // check the left subtree
        GifGetClosestPaletteColor(tree, color, bestInd, bestDiff, node.left);
        int cmpdiff = node.splitVal - comp;
        if ( bestDiff > comp_mult * cmpdiff*cmpdiff )
        {
            // cannot prove there's not a better value in the right subtree, check that too
            GifGetClosestPaletteColor(tree, color, bestInd, bestDiff, node.right);
        }
    }
    else
    {
        GifGetClosestPaletteColor(tree, color, bestInd, bestDiff, node.right);
        // The left subtree has component values <= (node.splitVal - 1)
        int cmpdiff = comp - (node.splitVal - 1);
        if ( bestDiff > comp_mult * cmpdiff*cmpdiff )
        {
            GifGetClosestPaletteColor(tree, color, bestInd, bestDiff, node.left);
        }
    }
}

void GifSwapPixels(GifRGBA * image, int pixA, int pixB)
{
    GifRGBA temp = image[pixA];
    image[pixA] = image[pixB];
    image[pixB] = temp;
}

// just the partition operation from quicksort 3-way
// Center element used as pivot. Afterwards, the pixels in [left, right) have
// 'com' component equal to the pivot; those before/after are lesser/greater.
uint8_t GifPartition(GifRGBA * image, int com, int &left, int &right)
{
    GifSwapPixels(image, left, left + (right - left) / 2);
    uint8_t comPivot = image[left].comps(com);
    for (int i1=left+1; i1<right; i1++)
    {
        uint8_t comArray = image[i1].comps(com);
        if ( comArray < comPivot )
        {
            GifSwapPixels(image, i1, left);
            left++;
        }
        else if ( comArray > comPivot )
        {
            right--;
            GifSwapPixels(image, i1, right);
            i1--;
        }
    }
    return comPivot;
}

// Perform an incomplete sort, finding all elements above and below the desired median
int GifPartitionByMedian(GifRGBA * image, int com, uint8_t & pivotVal, int left, int right)
{
    int neededCenter = left + (right - left) / 2;
    int initLeft = left, initRight = right;
    GIF_ASSERT(left < right-1);
    while (left < right-1)
    {
        int centerLeft = left, centerRight = right;
        pivotVal = GifPartition(image, com, centerLeft, centerRight);
        // Pixels with com equal to pivotVal are now in the interval [centerLeft, centerRight)

        if ( neededCenter < centerLeft )
            right = centerLeft;
        else if ( neededCenter >= centerRight )
            left = centerRight;
        else if ( (centerLeft != initLeft && neededCenter - centerLeft <= centerRight - neededCenter) ||
                 centerRight == initRight )
            // Found the median, but have to decide whether to put it in left or right partition.
            // Never return initLeft or initRight: must carve off at least one pixel
            // (We can assume that there's at least one pixel not equal to pivotVal)
            return centerLeft;
        else
        {
            GIF_ASSERT(pivotVal < 255);
            pivotVal++;
            return centerRight;
        }
    }
    // This happens when neededCenter == left == right - 1. Those two pixels may or may not be equal
    GIF_ASSERT(left > initLeft);
    GIF_ASSERT(neededCenter == left);
    pivotVal = image[left].comps(com);  // [left,initRight) is the right subtree
    return neededCenter;
}

// Create the palette, by taking the average of all colors in each subcube (k-d tree leaf)
void GifAverageColors(GifRGBA * image, int transpColIndex, GifKDTree * tree)
{
    tree->pal.colors[transpColIndex].r = 0;
    tree->pal.colors[transpColIndex].g = 0;
    tree->pal.colors[transpColIndex].b = 0;
    tree->pal.colors[transpColIndex].a = 0;

    // Fill the rest of the palette with the leaf nodes. Nodes still on the queue are leaves
    int palIndex = 0;
    for ( int qIndex = 1; qIndex <= tree->queue.len; qIndex++, palIndex++ )
    {
        if ( palIndex == transpColIndex ) palIndex++;
        GifHeapQueue::qitem& qitem = tree->queue.items[qIndex];
        GifKDNode & node = tree->nodes[qitem.nodeIndex];

        uint64_t r=0, g=0, b=0;
        // If there are no changed pixels, then there is a single leaf node, with no pixels
        if ( node.firstPixel != node.lastPixel )
        {
                // Possible optimisation: if node.cost == 0, just use image[node.firstPixel]
                for ( int ii = node.firstPixel; ii < node.lastPixel; ii++ )
                {
                    r += image[ii].r;
                    g += image[ii].g;
                    b += image[ii].b;
                }

                uint32_t numPixels = node.lastPixel - node.firstPixel;
                r += (uint64_t)numPixels / 2;  // round to nearest
                g += (uint64_t)numPixels / 2;
                b += (uint64_t)numPixels / 2;

                r /= (uint32_t)numPixels;
                g /= (uint32_t)numPixels;
                b /= (uint32_t)numPixels;
        }
        GifRGBA & col = tree->pal.colors[palIndex];
        col.r = (uint8_t)r;
        col.g = (uint8_t)g;
        col.b = (uint8_t)b;
        node.palIndex = palIndex;
    }
    //tree->pal.numColors = GifIMax(palIndex, transpColIndex + 1);
}

// Calculate cost of a node, and which way we should split it
void GifEvalNode( GifRGBA * image, GifKDNode & node )
{
    // (Note: node.firstPixel == node.lastPixel is possible)
    // Find the axis with the largest range
    int minR = 255, maxR = 0;
    int minG = 255, maxG = 0;
    int minB = 255, maxB = 0;
    for (int ii = node.firstPixel; ii < node.lastPixel; ii++)
    {
        int r = image[ii].r;
        int g = image[ii].g;
        int b = image[ii].b;

        if (r > maxR) maxR = r;
        if (r < minR) minR = r;

        if (g > maxG) maxG = g;
        if (g < minG) minG = g;

        if (b > maxB) maxB = b;
        if (b < minB) minB = b;
    }

    int rRange = maxR - minR;
    int gRange = maxG - minG;
    int bRange = maxB - minB;
    int maxRange = GifIMax(GifIMax(rRange, gRange), bRange);
    node.cost = maxRange * (node.lastPixel - node.firstPixel);

    node.splitComp = offsetof(struct GifRGBA, g);
    if (bRange > gRange) node.splitComp = offsetof(struct GifRGBA, b);
    if (rRange > bRange && rRange > gRange) node.splitComp = offsetof(struct GifRGBA, r);
}

int GifAddNode( GifRGBA * image, int firstPixel, int lastPixel, GifKDTree * tree )
{
    int nodeIndex = tree->numNodes++;
    GIF_ASSERT(nodeIndex < 512);

    GifKDNode & node = tree->nodes[nodeIndex];
    node.firstPixel = firstPixel;
    node.lastPixel = lastPixel;
    node.isLeaf = true;
    GifEvalNode(image, node);
    GifHeapPush(&tree->queue, node.cost, nodeIndex);
    return nodeIndex;
}

// Split a leaf node in two. It must not be entirely one color (splitting by splitComp must reduce cost).
void GifSplitNode( GifRGBA * image, GifKDTree * tree, GifKDNode & node )
{
    int medianPixel = GifPartitionByMedian(image, node.splitComp, node.splitVal, node.firstPixel, node.lastPixel);

    GIF_ASSERT(medianPixel > node.firstPixel);
    GIF_ASSERT(medianPixel < node.lastPixel);
    GIF_ASSERT(node.splitVal > 0);

    node.left  = GifAddNode(image, node.firstPixel, medianPixel, tree);
    node.right = GifAddNode(image, medianPixel, node.lastPixel, tree);
    node.isLeaf = false;
}

// Builds a palette by creating a k-d tree of all pixels in the image
void GifSplitPalette( GifRGBA * image, GifKDTree * tree )
{
    // -1 for transparent color
    int maxLeaves = (1 << tree->pal.bitDepth) - 1;
    while ( tree->queue.len < maxLeaves )
    {
        int nodeIndex = tree->queue.items[1].nodeIndex;  // Top of the heap
        GifKDNode & node = tree->nodes[nodeIndex];

        if ( node.cost <= 0 )
            break;
        GIF_ASSERT(node.lastPixel > node.firstPixel + 1);  // At least two pixels

        GifHeapPop(&tree->queue);
        GifSplitNode(image, tree, node);
    }
}

// Finds all pixels that have changed from the previous image and
// moves them to the front of the buffer.
// This allows us to build a palette optimized for the colors of the
// changed pixels only.
int GifPickChangedPixels( const GifRGBA * lastFrame, GifRGBA * frame, size_t numPixels )
{
    int numChanged = 0;
    GifRGBA * writeIter = frame;

    for (size_t ii=0; ii<numPixels; ii++)
    {
        if ( !GifRGBEqual(*lastFrame, *frame) )
        {
            *writeIter++ = *frame;
            numChanged++;
        }
        lastFrame++;
        frame++;
    }

    return numChanged;
}

// Creates a palette by placing all the image pixels in a k-d tree and then averaging the blocks at the bottom.
// This is known as the "modified median split" technique
void GifMakePalette( GifWriter * writer, const GifRGBA * lastFrame, const GifRGBA * nextFrame, int bitDepth, GifKDTree * tree )
{
    size_t width  = writer->width;
    size_t height = writer->height;
    size_t pixels = writer->framePixels;

    tree->pal.bitDepth = bitDepth;
    tree->numNodes = 0;
    tree->queue.len = 0;

    // SplitPalette is destructive (it sorts the pixels by color) so
    // we must create a copy of the image for it to destroy
    memcpy(writer->spareFrame, nextFrame, writer->frameBytes);

    if (lastFrame)
        pixels = GifPickChangedPixels(lastFrame, writer->spareFrame, pixels);

    // initial node
    GifAddNode(writer->spareFrame, 0, pixels, tree);
    GifSplitPalette(writer->spareFrame, tree);
    GifAverageColors(writer->spareFrame, writer->transpColIndex, tree);
}

// Implements Floyd-Steinberg dithering, writes palette index to alpha
void GifDitherImage( GifWriter * writer, const GifRGBA * lastFrame, const GifRGBA * nextFrame, GifKDTree * tree )
{
    GifRGBA * outFrame = writer->frame;
    size_t width = writer->width;
    size_t height = writer->height;
    size_t pixels = writer->framePixels;
    size_t errorBytes = writer->framePixels * sizeof(GifRGBA32);

    if (! writer->errorPixels)
	    writer->errorPixels = (GifRGBA32*)writer->cb->alloc(errorBytes);

    // errorPixels holds the accumulated error for each pixel; alpha channel ignored.
    // The extra 8 bits of precision allow for sub-single-color error values
    // to be propagated

    GifRGBA32 * errorPixels = writer->errorPixels;
    const int & transpColIndex = writer->transpColIndex;
    const int & maxAccumError = writer->maxAccumError;

    memset(errorPixels, 0, errorBytes);

    for ( size_t yy=0; yy<height; yy++ )
    {
        for ( size_t xx=0; xx<width; xx++ )
        {
            GifRGBA nextPix = nextFrame[yy*width+xx];  // input
            GifRGBA32 errorPix = errorPixels[yy*width+xx];  // input
            GifRGBA & outPix = outFrame[yy*width+xx];  // output
            const GifRGBA * lastPix = lastFrame? &lastFrame[yy*width+xx] : NULL;

            // Cap the diffused error to prevent excessive bleeding.
            errorPix.r = GifIMin( maxAccumError * 256, GifIMax( -maxAccumError * 256, errorPix.r) );
            errorPix.g = GifIMin( maxAccumError * 256, GifIMax( -maxAccumError * 256, errorPix.g) );
            errorPix.b = GifIMin( maxAccumError * 256, GifIMax( -maxAccumError * 256, errorPix.b) );
            errorPix.r += (int32_t)nextPix.r * 256;
            errorPix.g += (int32_t)nextPix.g * 256;
            errorPix.b += (int32_t)nextPix.b * 256;

            // Compute the colors we want (rounding to nearest)
            GifRGBA searchColor;
            searchColor.r = (uint8_t)GifIMin(255, GifIMax(0, (errorPix.r + 127) / 256));
            searchColor.g = (uint8_t)GifIMin(255, GifIMax(0, (errorPix.g + 127) / 256));
            searchColor.b = (uint8_t)GifIMin(255, GifIMax(0, (errorPix.b + 127) / 256));
            searchColor.a = 0;

            // if it happens that we want the color from last frame, then just write out
            // a transparent pixel
            if ( lastFrame && GifRGBEqual(searchColor, *lastPix) )
            {
                outPix = searchColor;
                outPix.a = transpColIndex;
                continue;
            }

            int32_t bestDiff = 1000000;
            int32_t bestInd = transpColIndex;

            // Search the palette
            GifGetClosestPaletteColor(tree, searchColor, bestInd, bestDiff);

            // Write the result to the temp buffer
            outPix = tree->pal.colors[bestInd];
            outPix.a = bestInd;

            int32_t r_err = errorPix.r - outPix.r * 256;
            int32_t g_err = errorPix.g - outPix.g * 256;
            int32_t b_err = errorPix.b - outPix.b * 256;

            // Propagate the error to the four adjacent locations
            // that we haven't touched yet
            size_t quantloc_7 = (yy*width+xx+1);
            size_t quantloc_3 = (yy*width+width+xx-1);
            size_t quantloc_5 = (yy*width+width+xx);
            size_t quantloc_1 = (yy*width+width+xx+1);

            if (quantloc_7 < pixels)
            {
                GifRGBA32 & pix7 = errorPixels[quantloc_7];
                pix7.r += r_err * 6 / 16;
                pix7.g += g_err * 6 / 16;
                pix7.b += b_err * 6 / 16;
            }

            if (quantloc_3 < pixels)
            {
                GifRGBA32 & pix3 = errorPixels[quantloc_3];
                pix3.r += r_err * 3 / 16;
                pix3.g += g_err * 3 / 16;
                pix3.b += b_err * 3 / 16;
            }

            if (quantloc_5 < pixels)
            {
                GifRGBA32 & pix5 = errorPixels[quantloc_5];
                pix5.r += r_err * 4 / 16;
                pix5.g += g_err * 4 / 16;
                pix5.b += b_err * 4 / 16;
            }

            if (quantloc_1 < pixels)
            {
                GifRGBA32 & pix1 = errorPixels[quantloc_1];
                pix1.r += r_err / 16;
                pix1.g += g_err / 16;
                pix1.b += b_err / 16;
            }
        }
    }
}

// Picks palette colors for the image using simple thresholding, no dithering. Writes palette index to alpha.
void GifThreshImage( GifWriter * writer, const GifRGBA * lastFrame, const GifRGBA * nextFrame, GifKDTree * tree )
{
    GifRGBA * outFrame = writer->frame;
    size_t width = writer->width;
    size_t height = writer->height;
    size_t pixels = writer->framePixels;

    const int & transpColIndex = writer->transpColIndex;

    for ( size_t ii=0; ii<pixels; ii++ )
    {
        // if a previous color is available, and it matches the current color,
        // set the pixel to transparent
        if (lastFrame && GifRGBEqual(*lastFrame, *nextFrame))
        {
            *outFrame = *lastFrame;
            outFrame->a = transpColIndex;
        }
        else
        {
            // palettize the pixel
            int32_t bestDiff = 1000000;
            int32_t bestInd = 1;
            GifGetClosestPaletteColor(tree, *nextFrame, bestInd, bestDiff);

            // Write the resulting color to the output buffer
            *outFrame = tree->pal.colors[bestInd];
            outFrame->a = (uint8_t)bestInd;
        }

        if (lastFrame) lastFrame++;
        outFrame++;
        nextFrame++;
    }
}

// insert a single bit
void GifWriteBit( GifBitStatus & stat, size_t bit )
{
    bit = bit & 1;
    bit = bit << stat.bitIndex;
    stat.byte |= bit;

    stat.bitIndex++;
    if ( stat.bitIndex > 7 )
    {
        // move the newly-finished byte to the chunk buffer
        stat.chunk[stat.chunkIndex++] = stat.byte;
        // and start a new byte
        stat.bitIndex = 0;
        stat.byte = 0;
    }
}

// write all bytes so far to the file
void GifWriteChunk( GifWriterCb * cb, GifBitStatus & stat )
{
    cb->putc((int)stat.chunkIndex);
    cb->write(stat.chunk, stat.chunkIndex);

    stat.bitIndex = 0;
    stat.byte = 0;
    stat.chunkIndex = 0;
}

void GifWriteCode( GifWriterCb * cb, GifBitStatus & stat, uint32_t code, uint32_t length )
{
    for ( size_t ii=0; ii<length; ii++ )
    {
        GifWriteBit(stat, code);
        code = code >> 1;

        if ( stat.chunkIndex == 255 )
        {
            GifWriteChunk(cb, stat);
        }
    }
}

// write an image palette to the file
void GifWritePalette( const GifPalette * pPal, GifWriterCb * cb )
{
    cb->putc(0);  // first color: transparency
    cb->putc(0);
    cb->putc(0);
    for (int ii=1; ii<(1 << pPal->bitDepth); ii++)
    {
        const GifRGBA & col = pPal->colors[ii];
        cb->putc((int)col.r);
        cb->putc((int)col.g);
        cb->putc((int)col.b);
    }
}

// write the image header, LZW-compress and write out the image
// localPalette is true to write out pPal as a local palette; otherwise it is the global palette.
void GifWriteLzwImage(GifWriter * writer, size_t delay, const GifPalette * pPal)
{
    GifWriterCb * cb = writer->cb;
    GifRGBA * image = writer->frame;
    size_t width = writer->width;
    size_t height = writer->height;

    // graphics control extension
    cb->putc(0x21);
    cb->putc(0xf9);
    cb->putc(0x04);

    // disposal method
    cb->putc(0x05); // leave this frame in place (next will draw on top)
//  cb->putc(0x09); // replace this frame with the background (so next can have transparent areas)

    cb->putc(delay & 0xff);
    cb->putc((delay >> 8) & 0xff);
    cb->putc(writer->transpColIndex); // transparent color index
    cb->putc(0);

    cb->putc(0x2c); // image descriptor block

    cb->putc(0); // (left & 0xff);
    cb->putc(0); // ((left >> 8) & 0xff);
    cb->putc(0); // (top & 0xff);
    cb->putc(0); // ((top >> 8) & 0xff);

    cb->putc(width & 0xff);
    cb->putc((width >> 8) & 0xff);
    cb->putc(height & 0xff);
    cb->putc((height >> 8) & 0xff);

    cb->putc(0x80 + pPal->bitDepth-1); // local color table present, 2 ^ bitDepth entries
    GifWritePalette(pPal, cb);

    //
    const uint32_t minCodeSize = pPal->bitDepth;
    const uint32_t clearCode = 1 << pPal->bitDepth;

    cb->putc(pPal->bitDepth); // min code size 8 bits

    GifLzwTree * codeTree = writer->codeTree;
    memset(codeTree, 0, sizeof(*codeTree));

    int32_t curCode = -1;
    uint32_t codeSize = minCodeSize + 1;
    uint32_t maxCode = clearCode + 1;

    GifBitStatus stat;
    stat.byte = 0;
    stat.bitIndex = 0;
    stat.chunkIndex = 0;

    GifWriteCode(cb, stat, clearCode, codeSize);  // start with a fresh LZW dictionary

    for (size_t yy=0; yy<height; yy++)
    {
        for (size_t xx=0; xx<width; xx++)
        {
//          uint8_t nextValue = image[yy*width+xx].a;

#ifdef GIF_FLIP_VERT
            // bottom-left origin image (such as an OpenGL capture)
            uint8_t nextValue = image[((height-1-yy)*width+xx)].a;
#else
            // top-left origin
            uint8_t nextValue = image[yy*width+xx].a;
#endif

            // "loser mode" - no compression, every single code is followed immediately by a clear
            //WriteCode( f, stat, nextValue, codeSize );
            //WriteCode( f, stat, 256, codeSize );

            if ( curCode < 0 )
            {
                // the first value in the image
                curCode = nextValue;
            }
            else if ( codeTree->nodes[curCode].m_next[nextValue] )
            {
                // current run already in the dictionary
                curCode = codeTree->nodes[curCode].m_next[nextValue];
            }
            else
            {
                // finish the current run, write a code
                GifWriteCode( cb, stat, curCode, codeSize );

                // insert the new run into the dictionary
                codeTree->nodes[curCode].m_next[nextValue] = (uint16_t)++maxCode;

                if ( maxCode >= (1ul << codeSize) )
                {
                    // dictionary entry count has broken a size barrier,
                    // we need more bits for codes
                    codeSize++;
                }
                if ( maxCode == 4095 )
                {
                    // the dictionary is full, clear it out and begin anew
                    GifWriteCode(cb, stat, clearCode, codeSize); // clear tree

                    memset(codeTree, 0, sizeof(*codeTree));
                    codeSize = minCodeSize + 1;
                    maxCode = clearCode + 1;
                }

                curCode = nextValue;
            }
        }
    }

    // compression footer
    GifWriteCode( cb, stat, curCode, codeSize );
    GifWriteCode( cb, stat, clearCode, codeSize );
    GifWriteCode( cb, stat, clearCode+1, minCodeSize+1 );

    // write out the last partial chunk
    while ( stat.bitIndex ) GifWriteBit(stat, 0);
    if ( stat.chunkIndex ) GifWriteChunk(cb, stat);

    cb->putc(0); // image block terminator
}

// Creates a gif file.
// The input GIFWriter is assumed to be uninitialized.
// The delay value is the time between frames in hundredths of a second - note that not all viewers pay much attention to this value.
// transparent is whether to produce a transparent GIF. It only works if using GifWriteFrame8()
//     to provide images containing transparency, and it disables delta coding.
void GifBegin( GifWriter * writer, GifWriterCb * cb, size_t width, size_t height, bool looped)
{
    GIF_ASSERT(writer && ! writer->cb);
    GIF_ASSERT(width && height);

    writer->cb = cb;

    writer->width  = width;
    writer->height = height;
    writer->framePixels = width * height;
    writer->frameBytes  = width * height * sizeof(GifRGBA);
    writer->empty = true;

    writer->frame = (GifRGBA*) cb->alloc(writer->frameBytes);
    writer->spareFrame = (GifRGBA*) cb->alloc(writer->frameBytes);

    writer->codeTree = (GifLzwTree*) cb->alloc(sizeof(GifLzwTree));
    writer->errorPixels = NULL;

    writer->transpColIndex = 0;
    writer->maxAccumError = 256;

    //
    cb->puts("GIF89a");

    // screen descriptor
    cb->putc(width & 0xff);
    cb->putc((width >> 8) & 0xff);
    cb->putc(height & 0xff);
    cb->putc((height >> 8) & 0xff);

    cb->putc(0xf0);  // there is an unsorted global color table of 2 entries

    cb->putc(0);     // background color
    cb->putc(0);     // pixels are square (we need to specify this because it's 1989)

    // now the "global" palette (really just a dummy palette)
    // color 0: black
    cb->putc(0);
    cb->putc(0);
    cb->putc(0);
    // color 1: also black
    cb->putc(0);
    cb->putc(0);
    cb->putc(0);

    if (looped)
    {
        // animation header
        cb->putc(0x21); // extension
        cb->putc(0xff); // application specific
        cb->putc(11); // length 11
        cb->puts("NETSCAPE2.0"); // yes, really
        cb->putc(3); // 3 bytes of NETSCAPE2.0 data

        cb->putc(1); // JUST BECAUSE
        cb->putc(0); // loop infinitely (byte 0)
        cb->putc(0); // loop infinitely (byte 1)

        cb->putc(0); // block terminator
    }
}

// Writes out a new frame to a GIF in progress.
// The GIFWriter should have been created by GIFBegin.
// AFAIK, it is legal to use different bit depths for different frames of an image -
// this may be handy to save bits in animations that don't change much.
void GifWriteFrame( GifWriter * writer, const GifRGBA * image, size_t delay, int bitDepth = 8, bool dither = false )
{
    GIF_ASSERT(writer && writer->cb);
    GIF_ASSERT(1 <= bitDepth && bitDepth <= 8);

    GifWriterCb * cb = writer->cb;
    GifKDTree tree;

    const GifRGBA * prevFrame = writer->empty ? NULL : writer->frame;

    GifMakePalette(writer, dither ? NULL : prevFrame, image, bitDepth, &tree);

    if (dither) GifDitherImage(writer, prevFrame, image, &tree);
    else        GifThreshImage(writer, prevFrame, image, &tree);

    GifWriteLzwImage(writer, delay, &tree.pal);

    writer->empty = false;
}

// Writes the EOF code, closes the file handle, and frees temp memory used by a GIF.
// Many if not most viewers will still display a GIF properly if the EOF code is missing,
// but it's still a good idea to write it out.
void GifEnd( GifWriter * writer )
{
    GIF_ASSERT(writer && writer->cb);

    GifWriterCb * cb = writer->cb;

    cb->putc(0x3b); // end of file

    //
    cb->free(writer->errorPixels); // may by NULL
    cb->free(writer->codeTree);
    cb->free(writer->spareFrame);
    cb->free(writer->frame);

    writer->cb = NULL;
}

#endif
