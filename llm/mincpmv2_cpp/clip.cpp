// NOTE: This is modified from clip.cpp only for LLaVA,
// so there might be still unnecessary artifacts hanging around
// I'll gradually clean and extend it
// Note: Even when using identical normalized image inputs (see normalize_image_u8_to_f32()) we have a significant difference in resulting embeddings compared to pytorch

#include "log.h"

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#include <cassert>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <regex>
#include <stdexcept>
#include <vector>
#include <sstream>
#include <cinttypes>
#include <limits>

#include "clip.h"

static std::string format(const char * fmt, ...) {
    va_list ap;
    va_list ap2;
    va_start(ap, fmt);
    va_copy(ap2, ap);
    int size = vsnprintf(NULL, 0, fmt, ap);
    //GGML_ASSERT(size >= 0 && size < INT_MAX); // NOLINT
    std::vector<char> buf(size + 1);
    int size2 = vsnprintf(buf.data(), size + 1, fmt, ap2);
    //GGML_ASSERT(size2 == size);
    va_end(ap2);
    va_end(ap);
    return std::string(buf.data(), buf.size());
}


struct clip_hparams {
    int32_t image_size;
    int32_t patch_size;
    int32_t hidden_size;
    int32_t n_intermediate;
    int32_t projection_dim;
    int32_t n_head;
    int32_t n_layer;

    float eps;

    char mm_patch_merge_type[32] = "flat"; // spatial_unpad or flat (default)

    int32_t image_grid_pinpoints[32];
    int32_t image_crop_resolution;
};


struct clip_image_u8 * clip_image_u8_init() {
    return new clip_image_u8();
}

struct clip_image_f32 * clip_image_f32_init() {
    return new clip_image_f32();
}

void clip_image_u8_free(struct clip_image_u8  * img) { delete img; }
void clip_image_f32_free(struct clip_image_f32 * img) { delete img; }
void clip_image_u8_batch_free(struct clip_image_u8_batch  * batch) {
    if (batch->size > 0) {
        delete[] batch->data;
        batch->size = 0;
    }
}
void clip_image_f32_batch_free(struct clip_image_f32_batch  * batch) {
    if (batch->size > 0) {
        delete[] batch->data;
        batch->size = 0;
    }
}

static void build_clip_img_from_data(const stbi_uc * data, int nx, int ny, clip_image_u8 * img) {
    img->nx = nx;
    img->ny = ny;
    img->buf.resize(3 * nx * ny);
    memcpy(img->buf.data(), data, img->buf.size());
}

bool clip_image_load_from_file(const char * fname, clip_image_u8 * img) {
    int nx, ny, nc;
    auto * data = stbi_load(fname, &nx, &ny, &nc, 3);
    if (!data) {
        LOG_TEE("%s: failed to load image '%s'\n", __func__, fname);
        return false;
    }
    build_clip_img_from_data(data, nx, ny, img);
    stbi_image_free(data);
    return true;
}



bool clip_image_load_from_bytes(const unsigned char * bytes, size_t bytes_length, struct clip_image_u8 * img) {
    int nx, ny, nc;
    auto * data = stbi_load_from_memory(bytes, bytes_length, &nx, &ny, &nc, 3);
    if (!data) {
        LOG_TEE("%s: failed to decode image bytes\n", __func__);
        return false;
    }
    build_clip_img_from_data(data, nx, ny, img);
    stbi_image_free(data);
    return true;
}

// Linear interpolation between two points
inline float clip_lerp(float s, float e, float t) {
    return s + (e - s) * t;
}
// Bilinear resize function
static void bilinear_resize(const clip_image_u8& src, clip_image_u8& dst, int target_width, int target_height) {
    dst.nx = target_width;
    dst.ny = target_height;
    dst.buf.resize(3 * target_width * target_height);

    float x_ratio = static_cast<float>(src.nx - 1) / target_width;
    float y_ratio = static_cast<float>(src.ny - 1) / target_height;

    for (int y = 0; y < target_height; y++) {
        for (int x = 0; x < target_width; x++) {
            float px = x_ratio * x;
            float py = y_ratio * y;
            int x_floor = static_cast<int>(px);
            int y_floor = static_cast<int>(py);
            float x_lerp = px - x_floor;
            float y_lerp = py - y_floor;

            for (int c = 0; c < 3; c++) {
                float top = clip_lerp(
                    static_cast<float>(src.buf[3 * (y_floor * src.nx + x_floor) + c]),
                    static_cast<float>(src.buf[3 * (y_floor * src.nx + (x_floor + 1)) + c]),
                    x_lerp
                );
                float bottom = clip_lerp(
                    static_cast<float>(src.buf[3 * ((y_floor + 1) * src.nx + x_floor) + c]),
                    static_cast<float>(src.buf[3 * ((y_floor + 1) * src.nx + (x_floor + 1)) + c]),
                    x_lerp
                );
                dst.buf[3 * (y * target_width + x) + c] = static_cast<uint8_t>(clip_lerp(top, bottom, y_lerp));
            }
        }
    }
}

// Normalize image to float32 - careful with pytorch .to(model.device, dtype=torch.float16) - this sometimes reduces precision (32>16>32), sometimes not
static void normalize_image_u8_to_f32(const clip_image_u8* src, clip_image_f32* dst, const float mean[3], const float std[3]) {
    dst->nx = src->nx;
    dst->ny = src->ny;
    dst->buf.resize(src->buf.size());

    for (size_t i = 0; i < src->buf.size(); ++i) {
        int c = i % 3; // rgb
        dst->buf[i] = (static_cast<float>(src->buf[i]) / 255.0f - mean[c]) / std[c];
    }
}

inline float clip(float x, float lower, float upper) {
    return std::max(lower, std::min(x, upper));
}

bool bicubic_resize(const clip_image_u8 &img, clip_image_u8 &dst, int target_width, int target_height) {
    const int nx = img.nx;
    const int ny = img.ny;

    dst.nx = target_width;
    dst.ny = target_height;
    dst.buf.resize(3 * target_width * target_height);

    float Cc;
    float C[5];
    float d0, d2, d3, a0, a1, a2, a3;
    int i, j, k, jj;
    int x, y;
    float dx, dy;
    float tx, ty;

    tx = (float)nx / (float)target_width;
    ty = (float)ny / (float)target_height;

    // Bicubic interpolation; adapted from ViT.cpp, inspired from :
    //    -> https://github.com/yglukhov/bicubic-interpolation-image-processing/blob/master/libimage.c#L36
    //    -> https://en.wikipedia.org/wiki/Bicubic_interpolation

    for (i = 0; i < target_height; i++) {
        for (j = 0; j < target_width; j++) {
            x = (int)(tx * j);
            y = (int)(ty * i);

            dx = tx * j - x;
            dy = ty * i - y;

            for (k = 0; k < 3; k++) {
                for (jj = 0; jj <= 3; jj++) {
                    d0 = img.buf[(clip(y - 1 + jj, 0, ny - 1) * nx + clip(x - 1, 0, nx - 1)) * 3 + k] - img.buf[(clip(y - 1 + jj, 0, ny - 1) * nx + clip(x, 0, nx - 1)) * 3 + k];
                    d2 = img.buf[(clip(y - 1 + jj, 0, ny - 1) * nx + clip(x + 1, 0, nx - 1)) * 3 + k] - img.buf[(clip(y - 1 + jj, 0, ny - 1) * nx + clip(x, 0, nx - 1)) * 3 + k];
                    d3 = img.buf[(clip(y - 1 + jj, 0, ny - 1) * nx + clip(x + 2, 0, nx - 1)) * 3 + k] - img.buf[(clip(y - 1 + jj, 0, ny - 1) * nx + clip(x, 0, nx - 1)) * 3 + k];
                    a0 = img.buf[(clip(y - 1 + jj, 0, ny - 1) * nx + clip(x, 0, nx - 1)) * 3 + k];

                    a1 = -1.0 / 3 * d0 + d2 - 1.0 / 6 * d3;
                    a2 =  1.0 / 2 * d0 +      1.0 / 2 * d2;
                    a3 = -1.0 / 6 * d0 -      1.0 / 2 * d2 + 1.0 / 6 * d3;

                    C[jj] = a0 + a1 * dx + a2 * dx * dx + a3 * dx * dx * dx;

                    d0 = C[0] - C[1];
                    d2 = C[2] - C[1];
                    d3 = C[3] - C[1];
                    a0 = C[1];
                    a1 = -1.0 / 3 * d0 + d2 - 1.0 / 6 * d3;
                    a2 =  1.0 / 2 * d0 +      1.0 / 2 * d2;
                    a3 = -1.0 / 6 * d0 -      1.0 / 2 * d2 + 1.0 / 6 * d3;
                    Cc = a0 + a1 * dy + a2 * dy * dy + a3 * dy * dy * dy;

                    const uint8_t Cc2 = std::min(std::max(std::round(Cc), 0.0f), 255.0f);
                    dst.buf[(i * target_width + j) * 3 + k] = float(Cc2);
                }
            }
        }
    }

    return true;
}

// llava-1.6 type of resize_and_pad (black)
static void resize_and_pad_image(const clip_image_u8& image, clip_image_u8 &image_output, const std::pair<int, int>& target_resolution) {
    int target_width = target_resolution.first;
    int target_height = target_resolution.second;

    float scale_w = static_cast<float>(target_width) / image.nx;
    float scale_h = static_cast<float>(target_height) / image.ny;

    int new_width, new_height;

    if (scale_w < scale_h) {
        new_width = target_width;
        new_height = std::min(static_cast<int>(std::ceil(image.ny * scale_w)), target_height);
    } else {
        new_height = target_height;
        new_width = std::min(static_cast<int>(std::ceil(image.nx * scale_h)), target_width);
    }

    clip_image_u8 resized_image;
    // bilinear_resize(image, resized_image, new_width, new_height);
    bicubic_resize(image, resized_image, new_width, new_height);

    clip_image_u8 padded_image;
    padded_image.nx = target_width;
    padded_image.ny = target_height;
    padded_image.buf.resize(3 * target_width * target_height, 0); // Initialize with black

    // Calculate padding offsets
    int pad_x = (target_width - new_width) / 2;
    int pad_y = (target_height - new_height) / 2;

    // Copy the resized image into the center of the padded buffer
    for (int y = 0; y < new_height; ++y) {
        for (int x = 0; x < new_width; ++x) {
            for (int c = 0; c < 3; ++c) {
                padded_image.buf[3 * ((y + pad_y) * target_width + (x + pad_x)) + c] = resized_image.buf[3 * (y * new_width + x) + c];
            }
        }
    }
    image_output = std::move(padded_image);
}

/**
 * Selects the best resolution from a list of possible resolutions based on the original size.
 *
 * @param original_size The original size of the image in the format (width, height).
 * @param possible_resolutions A list of possible resolutions in the format [(width1, height1), (width2, height2), ...].
 * @return The best fit resolution in the format (width, height).
 */
static std::pair<int, int> select_best_resolution(const std::pair<int, int> & original_size, const std::vector<std::pair<int, int>> & possible_resolutions) {
    int original_width = original_size.first;
    int original_height = original_size.second;
    std::pair<int, int> best_fit;
    int max_effective_resolution = 0;
    int min_wasted_resolution = std::numeric_limits<int>::max();

    for (const auto& resolution : possible_resolutions) {
        int width = resolution.first;
        int height = resolution.second;
        float scale = std::min(static_cast<float>(width) / original_width, static_cast<float>(height) / original_height);
        int downscaled_width = static_cast<int>(original_width * scale);
        int downscaled_height = static_cast<int>(original_height * scale);
        int effective_resolution = std::min(downscaled_width * downscaled_height, original_width * original_height);
        int wasted_resolution = (width * height) - effective_resolution;
        // LOG_TEE("resolution: %d %d, scale: %f, downscaled: %d %d, effective: %d, wasted: %d\n", width, height, scale, downscaled_width, downscaled_height, effective_resolution, wasted_resolution);
        if (effective_resolution > max_effective_resolution || (effective_resolution == max_effective_resolution && wasted_resolution < min_wasted_resolution)) {
            max_effective_resolution = effective_resolution;
            min_wasted_resolution = wasted_resolution;
            best_fit = resolution;
        }
    }

    return best_fit;
}

static std::vector<clip_image_u8*> divide_to_patches_u8(const clip_image_u8 & image, int patch_size) {
    std::vector<clip_image_u8*> patches;
    int width = image.nx;
    int height = image.ny;
    for (int i = 0; i < height; i += patch_size) {
        for (int j = 0; j < width; j += patch_size) {
            clip_image_u8 *patch = clip_image_u8_init();
            patch->nx = std::min(patch_size, width - j);
            patch->ny = std::min(patch_size, height - i);
            patch->buf.resize(3 * patch->nx * patch->ny);
            for (int y = 0; y < patch->ny; ++y) {
                for (int x = 0; x < patch->nx; ++x) {
                    for (int c = 0; c < 3; ++c) {
                        patch->buf[3 * (y * patch->nx + x) + c] = image.buf[3 * ((i + y) * width + (j + x)) + c];
                    }
                }
            }
            patches.push_back(patch);
        }
    }
    return patches;
}


// returns the normalized float tensor for llava-1.5, for spatial_unpad with anyres processing for llava-1.6 it returns the normalized image patch tensors as a vector
// res_imgs memory is being allocated here, previous allocations will be freed if found
bool clip_image_preprocess(struct clip_ctx* ctx, const clip_image_u8* img, clip_image_f32_batch* res_imgs) {
    bool pad_to_square = true;

    // free the previous res_imgs if any set
    if (res_imgs->size > 0) {
        clip_image_f32_batch_free(res_imgs);
    }
    res_imgs->data = nullptr;
    res_imgs->size = 0;

    // the logic below is to pad the shorter side to the longer side with a background color: rgb(122, 116, 104)
    // see https://github.com/haotian-liu/LLaVA/blob/e854a2bf85118c504f6f16bf5c3c7c92f8fa8c6b/llava/conversation.py#L113-L156

    clip_image_u8* temp = clip_image_u8_init(); // we will keep the input image data here temporarily
    temp->nx = img->nx;
    temp->ny = img->ny;
    temp->buf.resize(img->buf.size());
    memcpy(temp->buf.data(), img->buf.data(), temp->buf.size());


    const int nx = temp->nx;
    const int ny = temp->ny;
    // clip_image_save_to_bmp(*temp, "resized_vanilla.bmp");

    const int nx2 = temp->nx;
    const int ny2 = temp->ny;

    clip_image_f32* res = clip_image_f32_init();
    res->nx = nx2;
    res->ny = ny2;
    res->buf.resize(3 * nx2 * ny2);

    // const float scale = std::max(nx, ny) / (float)ctx->vision_model.hparams.image_size;

    // const int nx3 = int(nx / scale + 0.5f);
    // const int ny3 = int(ny / scale + 0.5f);

    const int nx3 = nx;
    const int ny3 = ny;

    const auto& m3 = ctx->image_mean; // {0.48145466f, 0.4578275f, 0.40821073f};
    const auto& s3 = ctx->image_std;  // {0.26862954f, 0.26130258f, 0.27577711f};

    for (int y = 0; y < ny3; y++) {
        for (int x = 0; x < nx3; x++) {
            for (int c = 0; c < 3; c++) {
                // linear interpolation
                const float sx = x;
                const float sy = y;

                const int x0 = std::max(0, (int)std::floor(sx));
                const int y0 = std::max(0, (int)std::floor(sy));

                const int x1 = std::min(x0 + 1, nx - 1);
                const int y1 = std::min(y0 + 1, ny - 1);

                const float dx = sx - x0;
                const float dy = sy - y0;

                const int j00 = 3 * (y0 * nx + x0) + c;
                const int j01 = 3 * (y0 * nx + x1) + c;
                const int j10 = 3 * (y1 * nx + x0) + c;
                const int j11 = 3 * (y1 * nx + x1) + c;

                const float v00 = temp->buf[j00];
                const float v01 = temp->buf[j01];
                const float v10 = temp->buf[j10];
                const float v11 = temp->buf[j11];

                const float v0 = v00 * (1.0f - dx) + v01 * dx;
                const float v1 = v10 * (1.0f - dx) + v11 * dx;

                const float v = v0 * (1.0f - dy) + v1 * dy;

                const uint8_t v2 = std::min(std::max(std::round(v), 0.0f), 255.0f);

                //rgb hwc ->chw
                //const int i = 3 * (y * nx3 + x) + c;
                const int i = (y * nx3 + x) + c * nx3 * ny3;

                res->buf[i] = ((float(v2) / 255.0f) - m3[c]) / s3[c];
            }
        }
    }
    clip_image_u8_free(temp);

    res_imgs->size = 1;
    res_imgs->data = new clip_image_f32[res_imgs->size];
    res_imgs->data[0] = *res;
    clip_image_f32_free(res);

    return true;
}


void clip_free(clip_ctx * ctx) {
    delete ctx;
}

int clip_n_mmproj_embd(const struct clip_ctx* ctx) {
    //embedding hidden_size minicpmv-2 2304 minicpmv-2.5 4096
    if (ctx->proj_type == PROJECTOR_TYPE_RESAMPLER) {
        return 2304;
    }
}

int clip_n_patches(const struct clip_ctx* ctx) {

    int n_patches = 1;

    //minicpmv-2 query_num 64, minicpmv-2.5 query_num 96
    if (ctx->proj_type == PROJECTOR_TYPE_RESAMPLER) {
        n_patches = 64;
    }

    return n_patches;
}

size_t clip_embd_nbytes(const struct clip_ctx * ctx) {
    return clip_n_patches(ctx) * clip_n_mmproj_embd(ctx) * sizeof(float);
}


ov::Tensor concatenate(const ov::Tensor& first, const ov::Tensor& second) {
    size_t res_d_0 = first.get_shape().at(0);
    size_t res_d_1 = first.get_shape().at(1);
    size_t res_d_2 = first.get_shape().at(2) * 2;
    ov::Tensor res{ first.get_element_type(), {res_d_0, res_d_1, res_d_2} };
    float* first_data = first.data<float>();
    float* second_data = second.data<float>();
    float* res_data = res.data<float>();
    for (size_t i = 0; i < res_d_0; ++i) {
        for (size_t j = 0; j < res_d_1; ++j) {
            size_t k = 0;
            for (; k < first.get_shape().at(2); ++k) {
                res_data[i * res_d_1 * res_d_2 + j * res_d_2 + k]
                    = first_data[i * res_d_1 * first.get_shape().at(2) + j * first.get_shape().at(2) + k];
            }
            for (size_t l = 0; l < second.get_shape().at(2); ++l, ++k) {
                res_data[i * res_d_1 * res_d_2 + j * res_d_2 + k]
                    = second_data[i * res_d_1 * second.get_shape().at(2) + j * second.get_shape().at(2) + l];
            }
        }
    }
    return res;
}

/// embed_dim: output dimension for each position
/// pos: a list of positions to be encoded: size (H, W)
/// out: (H, W, D)
ov::Tensor get_1d_sincos_pos_embed_from_grid_new(size_t embed_dim, const ov::Tensor& pos) {
    OPENVINO_ASSERT(embed_dim % 2 == 0);
    OPENVINO_ASSERT(pos.get_shape().size() == 3);
    OPENVINO_ASSERT(pos.get_shape().at(0) == 1);
    size_t d0 = pos.get_shape().at(1);
    size_t d1 = pos.get_shape().at(2);
    size_t d2 = embed_dim / 2;
    std::vector<float> omega(d2);
    for (size_t idx = 0; idx < omega.size(); ++idx) {
        omega.at(idx) = idx / (embed_dim / 2.0);
        omega.at(idx) = 1.0 / std::pow(10000, omega.at(idx));  // (D/2,)
    }
    const float* const pos_data = pos.data<float>();
    ov::Tensor out(ov::element::f32, { d0, d1, d2 });  // (H, W, D/2), outer product
    float* out_data = out.data<float>();
    for (size_t i = 0; i < d0; ++i) {
        for (size_t j = 0; j < d1; ++j) {
            for (size_t k = 0; k < d2; ++k) {
                out_data[i * d1 * d2 + j * d2 + k]
                    = pos_data[i * d1 + j] * omega[k];
            }
        }
    }

    ov::Tensor emb_sin{ out.get_element_type(), out.get_shape() };  // (H, W, D/2)
    float* emb_sin_data = emb_sin.data<float>();
    std::transform(out_data, out_data + out.get_size(), emb_sin_data, [](float arg) {
        return std::sin(arg);
        });
    ov::Tensor emb_cos{ out.get_element_type(), out.get_shape() };  // (H, W, D/2)
    float* emb_cos_data = emb_cos.data<float>();
    std::transform(out_data, out_data + out.get_size(), emb_cos_data, [](float arg) {
        return std::cos(arg);
        });
    return concatenate(emb_sin, emb_cos); // (H, W, D)
}

ov::Tensor get_2d_sincos_pos_embed_from_grid(size_t embed_dim, const ov::Tensor& grid) {
    OPENVINO_ASSERT(embed_dim % 2 == 0);

    // use half of dimensions to encode grid_h
    ov::Coordinate begin_h{ 0, 0, 0 };
    ov::Coordinate end_h{ grid.get_shape() };
    end_h.at(0) = 1;
    ov::Coordinate begin_w{ 1, 0, 0 };
    ov::Coordinate end_w{ grid.get_shape() };
    end_w.at(0) = 2;
    ov::Tensor emb_h = get_1d_sincos_pos_embed_from_grid_new(embed_dim / 2, ov::Tensor{ grid, begin_h, end_h });  // (H, W, D/2)
    ov::Tensor emb_w = get_1d_sincos_pos_embed_from_grid_new(embed_dim / 2, ov::Tensor{ grid, begin_w, end_w });  // (H, W, D/2)

    ov::Shape emb_h_shape = emb_h.get_shape();

    return concatenate(emb_h, emb_w);
}

ov::Tensor get_2d_sincos_pos_embed(size_t embed_dim, size_t grid_h, size_t grid_w) {
    size_t grid_h_size = grid_h, grid_w_size = grid_w;
    ov::Tensor grid(ov::element::f32, { 2, grid_h_size, grid_w_size });
    float* data = grid.data<float>();

    for (size_t y = 0; y < grid_h_size; ++y) {
        std::iota(data, data + grid_w_size, 0);
        data += grid_w_size;
    }

    for (size_t y = 0; y < grid_h_size; ++y) {
        std::fill(data, data + grid_w_size, y);
        data += grid_w_size;
    }

    ov::Shape grid_shape = grid.get_shape();

    return get_2d_sincos_pos_embed_from_grid(embed_dim, grid);
}

void set_2d_pos_cache(struct clip_ctx* ctx) {
    ctx->_pos_embeds = get_2d_sincos_pos_embed(ctx->embed_dim, ctx->pos_max_size, ctx->pos_max_size);
}

void adjust_pos_cache(struct clip_ctx* ctx, size_t target_size_h, size_t target_size_w) {
    if (target_size_h > ctx->pos_max_size || target_size_w > ctx->pos_max_size) {
        set_2d_pos_cache(ctx);
        ctx->pos_max_size = std::max(target_size_h, target_size_w);
    }
}

bool clip_image_encode(struct clip_ctx* ctx, const int n_threads, clip_image_f32* img, float* vec, std::pair<int, int> load_image_size = { 448, 448 }) {
    //if (!ctx->has_vision_encoder) {
    //    LOG_TEE("This gguf file seems to have no vision encoder\n");
    //    return false;
    //}

    clip_image_f32_batch imgs{};
    imgs.size = 1;
    imgs.data = img;
    return clip_image_batch_encode(ctx, n_threads, &imgs, vec, load_image_size);
}


bool clip_image_batch_encode(clip_ctx* ctx, const int n_threads, const clip_image_f32_batch* imgs, float* vec, std::pair<int, int> load_image_size = { 448, 448 }) {
    //don't support multi batch
    size_t batch_size = imgs->size;

    const size_t image_size_width = load_image_size.first;
    const size_t image_size_height = load_image_size.second;
    const size_t patch_size = 14;
    const size_t num_patches = ((image_size_width / patch_size) * (image_size_height / patch_size));
    const size_t num_positions = num_patches;

    //OpenVINO inference to get vision embedding
    ov::Shape input_shape = { batch_size, 3, image_size_height, image_size_width };
    ov::Tensor input_tensor = ov::Tensor(ov::element::f32, input_shape); // , imgs->data[0].buf.data());
    std::memcpy(input_tensor.data<float>(), imgs->data[0].buf.data(), input_tensor.get_byte_size());

    ctx->ireq_vision.set_input_tensor(input_tensor);
    ctx->ireq_vision.infer();
    //ctx->ireq_vision.start_async();
    //ctx->ireq_vision.wait();

    const ov::Tensor& vision_output_tensor = ctx->ireq_vision.get_output_tensor();

    ov::Shape out_shape = vision_output_tensor.get_shape();
    float *data = vision_output_tensor.data<float>();

    size_t bs = 1;
    size_t target_h = image_size_height / patch_size;
    size_t target_w = image_size_width / patch_size;
    adjust_pos_cache(ctx, target_h, target_w);
    size_t max_patch_len = target_h * target_w;
    ov::Tensor key_padding_mask(ov::element::boolean, { 1, max_patch_len });
    bool* mask_data = key_padding_mask.data<bool>();
    size_t embed_len = ctx->_pos_embeds.get_shape().at(2);
    ov::Tensor pos_embed(ov::element::f32, { max_patch_len, 1, embed_len });  // BLD => L * B * D
    float* pos_embed_data = pos_embed.data<float>();
    float* _pos_embed_data = ctx->_pos_embeds.data<float>();
    size_t _d0 = ctx->_pos_embeds.get_shape().at(0);
    size_t _d1 = ctx->_pos_embeds.get_shape().at(1);
    for (size_t h_idx = 0; h_idx < target_h; ++h_idx) {
        for (size_t w_idx = 0; w_idx < target_w; ++w_idx) {
            std::copy_n(
                _pos_embed_data + (h_idx * _d1 + w_idx) * embed_len,
                embed_len,
                pos_embed_data + (h_idx * target_w + w_idx) * bs * embed_len
            );
        }
    }

    std::fill_n(mask_data, max_patch_len, false);
 
    //Resampler inference with OpenVINO
    ctx->ireq_resampler.set_tensor("x", vision_output_tensor);
    ctx->ireq_resampler.set_tensor("pos_embed", pos_embed);
    ctx->ireq_resampler.set_tensor("key_padding_mask", key_padding_mask);

    ctx->ireq_resampler.infer();
    //ctx->ireq_resampler.start_async();
    //ctx->ireq_resampler.wait();
    const ov::Tensor& vision_embded_tensor = ctx->ireq_resampler.get_output_tensor();

    // copy the embeddings to the location passed by the user
    std::memcpy(vec, vision_embded_tensor.data<float>(), vision_embded_tensor.get_byte_size());

    return true;
}

