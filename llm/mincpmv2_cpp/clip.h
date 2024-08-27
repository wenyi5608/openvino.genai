#ifndef CLIP_H
#define CLIP_H

#include <stddef.h>
#include <stdint.h>
#include <vector>
#include <numeric>

#include <openvino/openvino.hpp>

#ifdef LLAMA_SHARED
#    if defined(_WIN32) && !defined(__MINGW32__)
#        ifdef LLAMA_BUILD
#            define CLIP_API __declspec(dllexport)
#        else
#            define CLIP_API __declspec(dllimport)
#        endif
#    else
#        define CLIP_API __attribute__ ((visibility ("default")))
#    endif
#else
#    define CLIP_API
#endif

//#define CLIP_DEBUG_FUNCTIONS
enum projector_type {
    PROJECTOR_TYPE_RESAMPLER,
    PROJECTOR_TYPE_UNKNOWN,
};

struct clip_ctx {
    bool has_text_encoder = false;
    bool has_vision_encoder = false;
    bool has_minicpmv_projector = false;

    float image_mean[3];
    float image_std[3];
    int32_t ftype = 1;

    std::vector<uint8_t> buf_compute_meta;

    projector_type proj_type = PROJECTOR_TYPE_RESAMPLER;

    ov::InferRequest ireq_vision;
    ov::InferRequest ireq_resampler;
    size_t embed_dim = 2304;
    ov::Tensor _pos_embeds;
    size_t pos_max_size = 70;
};

// RGB uint8 image
struct clip_image_u8 {
    int nx;
    int ny;

    std::vector<uint8_t> buf;
};

// RGB float32 image (NHWC)
// Memory layout: RGBRGBRGB...
struct clip_image_f32 {
    int nx;
    int ny;

    std::vector<float> buf;
};


struct clip_image_u8_batch {
    struct clip_image_u8 * data;
    size_t size;
};

struct clip_image_f32_batch {
    struct clip_image_f32 * data;
    size_t size;
};


CLIP_API void clip_free(struct clip_ctx * ctx);

CLIP_API size_t clip_embd_nbytes(const struct clip_ctx * ctx);

CLIP_API int clip_n_patches(const struct clip_ctx* ctx);
CLIP_API int clip_n_mmproj_embd(const struct clip_ctx* ctx);

CLIP_API struct clip_image_u8  * clip_image_u8_init ();
CLIP_API struct clip_image_f32 * clip_image_f32_init();

CLIP_API void clip_image_u8_free (struct clip_image_u8  * img);
CLIP_API void clip_image_f32_free(struct clip_image_f32 * img);
CLIP_API void clip_image_u8_batch_free (struct clip_image_u8_batch  * batch);
CLIP_API void clip_image_f32_batch_free(struct clip_image_f32_batch * batch);

CLIP_API bool clip_image_load_from_file(const char * fname, struct clip_image_u8 * img);

/** interpret bytes as an image file with length bytes_length, and use the result to populate img */
CLIP_API bool clip_image_load_from_bytes(const unsigned char * bytes, size_t bytes_length, struct clip_image_u8 * img);

CLIP_API bool bicubic_resize(const clip_image_u8& img, clip_image_u8& dst, int target_width, int target_height);

/** preprocess img and store the result in res_imgs, pad_to_square may be overriden to false depending on model configuration */
CLIP_API bool clip_image_preprocess(struct clip_ctx* ctx, const struct clip_image_u8* img, struct clip_image_f32_batch* res_imgs);

CLIP_API bool clip_image_encode(struct clip_ctx* ctx, int n_threads, struct clip_image_f32* img, float* vec, std::pair<int, int> load_image_size);
CLIP_API bool clip_image_batch_encode(struct clip_ctx* ctx, int n_threads, const struct clip_image_f32_batch* imgs, float* vec, std::pair<int, int> load_image_size);

CLIP_API ov::Tensor get_2d_sincos_pos_embed(size_t embed_dim, size_t grid_h, size_t grid_w);

#endif // CLIP_H
