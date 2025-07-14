#!/user/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
import cupy
import re

kernel_Render_updateOutput = '''

    extern "C" __global__ void kernel_Render_updateOutput(
        const int n,
        const int samples,
        const float* xs,
        const float* ys,
        const float* K,
        const float* depth_focus,
        const float* Eta,  // vignetting, length of lens cone / blur parameter
        const float* images,
        const float* alphas,
        const float* coffs,
        float* bokeh_cum,
        float* weight_cum
    )
    {
        for (int intIndex = (blockIdx.x * blockDim.x) + threadIdx.x; intIndex < n; intIndex += blockDim.x * gridDim.x) {
            const int b = ( intIndex / SIZE_3(bokeh_cum) / SIZE_2(bokeh_cum) / 1 ) % SIZE_0(bokeh_cum);
            // const int c = ( intIndex / SIZE_3(bokeh_cum) / SIZE_2(bokeh_cum)                 ) % SIZE_1(bokeh_cum);
            const int y = ( intIndex / SIZE_3(bokeh_cum)                                 ) % SIZE_2(bokeh_cum);
            const int x = ( intIndex                                                 ) % SIZE_3(bokeh_cum);

            float num = 0;
            float sumR = 0;
            float sumG = 0;
            float sumB = 0;
            float sumA = 0;

            float radius = K[b];
            float zf = depth_focus[b];
            float eta = fabsf(Eta[b]);

            float df = 1 / zf;

            for (int i = 0; i < samples; ++i) {
                // float u = xs[i] * radius;
                // float v = ys[i] * radius;

                float unit_square = xs[i] * xs[i] + ys[i] * ys[i];

                // float df_ = df - 0.4 * (0.95 - unit_square);
                // float df_ = df - 0.4 * (0.05 + unit_square * unit_square - unit_square);
                // float df_ = df - 0.4 * (0. - unit_square * unit_square * unit_square + 2 * unit_square * unit_square - unit_square);
                float df_ = df;

                float alpha_prev = 1;

                float dist_square = (xs[i] + eta * (x - (SIZE_3(bokeh_cum)-1.0)/2.0)) * (xs[i] + eta * (x - (SIZE_3(bokeh_cum)-1.0)/2.0)) + (ys[i] + eta * (y - (SIZE_2(bokeh_cum)-1.0)/2.0)) * (ys[i] + eta * (y - (SIZE_2(bokeh_cum)-1.0)/2.0));
                // float alpha_curr = 0.5 + 0.5 * tanhf(1e1 * (1.1 - d));
                if (dist_square > 1.1) {
                    // sumR += 0. * alpha_prev;
                    // sumG += 0. * alpha_prev;
                    // sumB += 0. * alpha_prev;
                    
                    // if (Eta[b] > 0) {   // 2023/5/26
                    //     num += 1.;
                    // }
                    continue;
                }

                // if (alpha_curr > 0.99) {
                //     break;
                // }
                // else {
                //     alpha_prev *= (1 - alpha_curr); 
                // }

                for (int l = 0; l < SIZE_1(images); ++l) {
                    float A = coffs[OFFSET_3(coffs, b, l, 0)];
                    float B = coffs[OFFSET_3(coffs, b, l, 1)];
                    float C = coffs[OFFSET_3(coffs, b, l, 2)];
                    // float t = (A * u + B * v + C) / (zf_ - A * (zf_ * x - u) - B * (zf_ * y - v) + 1e-5);

                    // float x_intersect = t * (zf_ * x - u) + u;
                    // float y_intersect = t * (zf_ * y - v) + v;
                    // float z_intersect = t * zf_;

                    // float x_map = x_intersect / (z_intersect + 1e-5);
                    // float y_map = y_intersect / (z_intersect + 1e-5);

                    float tmp = (1 - A * x - B * y - C * df_) / (A * xs[i] + B * ys[i] + C / radius + 1e-5);
                    float x_map = x + tmp * xs[i];
                    float y_map = y + tmp * ys[i];

                    if (0 <= y_map && y_map < SIZE_2(bokeh_cum) && 0 <= x_map && x_map < SIZE_3(bokeh_cum)) {
                        int x1 = int(x_map);
                        int x2 = x1 + 1;
                        int y1 = int(y_map);
                        int y2 = y1 + 1;

                        int x2_ = x2;
                        int y2_ = y2;

                        if (x2 >= SIZE_3(bokeh_cum)) {
                            x2_ = SIZE_3(bokeh_cum) - 1;
                        }
                        if (y2 >= SIZE_2(bokeh_cum)) {
                            y2_ = SIZE_2(bokeh_cum) - 1;
                        }

                        float alpha_curr = 1;

                        // if (-0.4 <= x - x_map && x - x_map <= 0.4 && -0.4 <= y - y_map <= 0.4) {
                        //     alpha_curr = alphas[OFFSET_5(alphas, b, l, 0, y, x)];
                        //     if (alpha_curr < 0.01) {
                        //         continue;
                        //     }
                        //     sumR += images[OFFSET_5(images, b, l, 0, y, x)] * alpha_curr * alpha_prev;
                        //     sumG += images[OFFSET_5(images, b, l, 1, y, x)] * alpha_curr * alpha_prev;
                        //     sumB += images[OFFSET_5(images, b, l, 2, y, x)] * alpha_curr * alpha_prev;
                        // }
                        // else {
                        float f1 = (x2 - x_map) * alphas[OFFSET_5(alphas, b, l, 0, y1, x1)];
                        float f2 = (x_map - x1) * alphas[OFFSET_5(alphas, b, l, 0, y1, x2_)];
                        float f3 = (x2 - x_map) * alphas[OFFSET_5(alphas, b, l, 0, y2_, x1)];
                        float f4 = (x_map - x1) * alphas[OFFSET_5(alphas, b, l, 0, y2_, x2_)];
                        float f12 = f1 + f2;
                        float f34 = f3 + f4;
                        alpha_curr = (y2 - y_map) * f12 + (y_map - y1) * f34;
                        if (alpha_curr < 0.01) {
                            if (l == SIZE_1(images) - 1) {  // continues still in last layer
                                num += alpha_prev;
                            }
                            continue;
                        }

                        // float donutRatio = 0.25;        

                        f12 = f1 * images[OFFSET_5(images, b, l, 0, y1, x1)] + f2 * images[OFFSET_5(images, b, l, 0, y1, x2_)];
                        f34 = f3 * images[OFFSET_5(images, b, l, 0, y2_, x1)] + f4 * images[OFFSET_5(images, b, l, 0, y2_, x2_)];
                        float fR = (y2 - y_map) * f12 + (y_map - y1) * f34;
                        // fR *= 1 - donutRatio + donutRatio * tanhf(40 * (dist_square - 0.9));  // donut
                        sumR += fR * alpha_prev;

                        f12 = f1 * images[OFFSET_5(images, b, l, 1, y1, x1)] + f2 * images[OFFSET_5(images, b, l, 1, y1, x2_)];
                        f34 = f3 * images[OFFSET_5(images, b, l, 1, y2_, x1)] + f4 * images[OFFSET_5(images, b, l, 1, y2_, x2_)];
                        float fG = (y2 - y_map) * f12 + (y_map - y1) * f34;
                        // fG *= 1 - donutRatio + donutRatio * tanhf(40 * (dist_square - 0.9));  // donut
                        sumG += fG * alpha_prev;

                        f12 = f1 * images[OFFSET_5(images, b, l, 2, y1, x1)] + f2 * images[OFFSET_5(images, b, l, 2, y1, x2_)];
                        f34 = f3 * images[OFFSET_5(images, b, l, 2, y2_, x1)] + f4 * images[OFFSET_5(images, b, l, 2, y2_, x2_)];
                        float fB = (y2 - y_map) * f12 + (y_map - y1) * f34;
                        // fB *= 1 - donutRatio + donutRatio * tanhf(40 * (dist_square - 0.9));  // donut
                        sumB += fB * alpha_prev;

                        sumA += alpha_curr * alpha_prev;

                        num += alpha_curr * alpha_prev;

                        if (alpha_curr > 0.99) {
                            // num += 1;
                            break;
                        }
                        else {
                            alpha_prev *= (1 - alpha_curr); 
                        }
                    }
                }
            }
            bokeh_cum[OFFSET_4(bokeh_cum, b, 0, y, x)] = sumR;
            bokeh_cum[OFFSET_4(bokeh_cum, b, 1, y, x)] = sumG;
            bokeh_cum[OFFSET_4(bokeh_cum, b, 2, y, x)] = sumB;
            bokeh_cum[OFFSET_4(bokeh_cum, b, 3, y, x)] = sumA;
            weight_cum[OFFSET_4(weight_cum, b, 0, y, x)] = num;
        }
    }

'''


def cupy_kernel(strFunction, objVariables):
    strKernel = globals()[strFunction]

    while True:
        objMatch = re.search('(SIZE_)([0-5])(\()([^\)]*)(\))', strKernel)

        if objMatch is None:
            break
        # end

        intArg = int(objMatch.group(2))

        strTensor = objMatch.group(4)
        intSizes = objVariables[strTensor].size()

        strKernel = strKernel.replace(objMatch.group(), str(intSizes[intArg]))
    # end

    while True:
        objMatch = re.search('(OFFSET_)([0-5])(\()([^\)]+)(\))', strKernel)

        if objMatch is None:
            break
        # end

        intArgs = int(objMatch.group(2))
        strArgs = objMatch.group(4).split(',')

        strTensor = strArgs[0]
        intStrides = objVariables[strTensor].stride()
        strIndex = ['((' + strArgs[intArg + 1].replace('{', '(').replace('}', ')').strip() + ')*' + str(
            intStrides[intArg]) + ')' for intArg in range(intArgs)]

        strKernel = strKernel.replace(objMatch.group(0), '(' + str.join('+', strIndex) + ')')
    # end

    while True:
        objMatch = re.search('(VALUE_)([0-5])(\()([^\)]+)(\))', strKernel)

        if objMatch is None:
            break
        # end

        intArgs = int(objMatch.group(2))
        strArgs = objMatch.group(4).split(',')

        strTensor = strArgs[0]
        intStrides = objVariables[strTensor].stride()
        strIndex = ['((' + strArgs[intArg + 1].replace('{', '(').replace('}', ')').strip() + ')*' + str(
            intStrides[intArg]) + ')' for intArg in range(intArgs)]

        strKernel = strKernel.replace(objMatch.group(0), strTensor + '[' + str.join('+', strIndex) + ']')
    # end

    return strKernel


# end


# @cupy.util.memoize(for_each_device=True)
@cupy.memoize(for_each_device=True)
def cupy_launch(strFunction, strKernel):
    # return cupy.cuda.compile_with_cache(strKernel).get_function(strFunction)
    return cupy.RawModule(code=strKernel).get_function(strFunction)


# end


class _FunctionRender(torch.autograd.Function):
    @staticmethod
    def forward(self, images, alphas, coffs, K, depth_focus, Eta, samples_per_side):
        # self.save_for_backward(image, defocus)

        # defocus_dilate = -10000 * torch.ones_like(defocus).int()
        # bokeh = torch.zeros_like(images[:, 0])
        # bokeh_cum = torch.zeros_like(images[:, 0])
        b, _, _, h, w = images.shape
        bokeh_cum = torch.zeros(b, 4, h, w).cuda()
        weight_cum = torch.zeros_like(images[:, 0, :1])
        if isinstance(depth_focus, (int, float)):
            depth_focus = torch.tensor(depth_focus, device=bokeh_cum.device).repeat(images.shape[0]).float()
        elif isinstance(depth_focus, torch.Tensor):
            depth_focus = depth_focus[:, 0, 0, 0].float()
        if isinstance(K, (int, float)):
            K = torch.tensor(K, device=bokeh_cum.device).repeat(images.shape[0]).float()
        elif isinstance(K, torch.Tensor):
            K = K[:, 0, 0, 0].float()
        if isinstance(Eta, (int, float)):
            Eta = torch.tensor(Eta, device=bokeh_cum.device).repeat(images.shape[0]).float()
        elif isinstance(Eta, torch.Tensor):
            Eta = Eta[:, 0, 0, 0].float()

        size = samples_per_side

        xs = []
        ys = []
        x_lin = torch.linspace(-1, 1, steps=size, device=bokeh_cum.device)
        y_lin = torch.linspace(-1, 1, steps=size, device=bokeh_cum.device)
        X, Y = torch.meshgrid(x_lin, y_lin, indexing="ij")
        mask = (X**2 + Y**2) <= 1
        xs = X[mask]
        ys = Y[mask]
        samples = len(xs)

        if images.is_cuda == True:
            # n = bokeh_cum_cum.nelement()
            # n = bokeh_cum.nelement() // 3
            n = weight_cum.nelement()
            cupy_launch('kernel_Render_updateOutput', cupy_kernel('kernel_Render_updateOutput', {
                'samples': samples,
                'xs': xs,
                'ys': ys,
                'K': K,
                'depth_focus': depth_focus,
                'Eta': Eta,
                'images': images,
                'alphas': alphas,
                'coffs': coffs,
                'bokeh_cum': bokeh_cum,
                'weight_cum': weight_cum
            }))(
                grid=tuple([int((n + 512 - 1) / 512), 1, 1]),
                block=tuple([512, 1, 1]),
                args=(
                    cupy.int32(n),
                    cupy.int32(samples),
                    xs.data_ptr(),
                    ys.data_ptr(),
                    K.data_ptr(),
                    depth_focus.data_ptr(),
                    Eta.data_ptr(),
                    images.data_ptr(),
                    alphas.data_ptr(),
                    coffs.data_ptr(),
                    bokeh_cum.data_ptr(),
                    weight_cum.data_ptr()
                )
            )

        elif bokeh_cum.is_cuda == False:
            raise NotImplementedError()

        # end

        return bokeh_cum, weight_cum
    # end


# end


def FunctionRender(images, alphas, coffs, K, depth_focus, Eta, samples_per_side):
    bokeh_cum, weight_cum = _FunctionRender.apply(images, alphas, coffs, K, depth_focus, Eta, samples_per_side)

    return bokeh_cum, weight_cum


# end


class ModuleRenderRT(torch.nn.Module):
    def __init__(self):
        super(ModuleRenderRT, self).__init__()
        # self.gaussian_blur = GaussianBlur(gaussian_kernel)

    # end

    def forward(self, images, alphas, coffs, K, depth_focus, Eta, samples_per_side=51):
        bokeh_cum, weight_cum = FunctionRender(images, alphas, coffs, K, depth_focus, Eta, samples_per_side)
        return bokeh_cum, weight_cum
    # end


# end


if __name__ == '__main__':
    module = ModuleRenderRT().cuda()
    # In practice, make sure that the lower index corresponds to the nearer object in the second dimension
    # of the following three tensors, and do not let the disparities of different objects overlap with each other
    # by cautiously setting tensor "coffs"
    images = torch.zeros((1, 2, 3, 302, 302)).cuda()  # batch, num_object, C, H, W
    alphas = torch.zeros((1, 2, 1, 302, 302)).cuda()  # batch, num_object, C, H, W
    coffs = torch.zeros((1, 2, 3)).cuda()  # batch, num_object, num_param (a, b, c) (refer to Eq.7 of the paper)

    images[0, 0] = 100
    images[0, 1] = 0.5

    # alphas[0, 0, 0, 50:100, 50:100] = 1
    for i in range(5):
        for j in range(5):
            alphas[0, 0, 0, 50 * (i + 1):50 * (i + 1) + 2, 50 * (j + 1):50 * (j + 1) + 2] = 1
    alphas[0, 1] = 1

    coffs[0, 0, 0], coffs[0, 0, 1], coffs[0, 0, 2] = 0, 0, 1 / 1
    coffs[0, 1, 0], coffs[0, 1, 1], coffs[0, 1, 2] = 0, 0, 1 / 1

    K = 30  # blur radius
    zf = 1 / 0.5  # depth of focus
    eta = 0  # 6e-3
    samples_per_side = 101
    import time

    torch.cuda.synchronize()
    start = time.time()
    for i in range(1):
        bokeh_cum, weight_cum = module(images, alphas, coffs, K, zf, eta, samples_per_side)
        bokeh = bokeh_cum / weight_cum.clamp(1e-5, 1e5)
    torch.cuda.synchronize()
    print(time.time() - start)

    import cv2

    cv2.imwrite('bokeh.jpg', bokeh[0].detach().clone().permute(1, 2, 0).cpu().numpy()[..., ::-1] * 255)
