/**
 * @file seg or blur origin image
 */

function mainFunc({
    out
}) {
    const THRESHHOLD = 0.2;
    return `

    #define SIGMA SIGMA 3.0
    #define BLUR_MSIZE 8
    #define MSIZE 3
    #define kernelSize 5.0
    #define weight 1.0

    uniform int type; // 0: blurBackground 1: drawHumanseg 2: drawMask

    void main() {
        vec2 outCoord = vCoord.xy;
       
        outCoord.y = 1.0 - vCoord.y;

        vec2 sourceTextureSize = vec2(${out.width_shape}, ${out.height_shape});
        vec2 sourceTexelSize = 1.0 / sourceTextureSize;

        float kernel[MSIZE]; // 3
        kernel[0] = 0.12579369017522166;
        kernel[1] = 0.13298;
        kernel[2] = 0.12579369017522166;

        float origin_alpha = 1.0 - TEXTURE2D(texture_origin, vec2(outCoord.x, outCoord.y) / 2.0).r;
        vec4 counter = TEXTURE2D(texture_counter, outCoord.xy);
        vec4 res = vec4(0.0);

        if (type == 0) {
            // Simple Cheap Box Blur 
            float pixelSizeX = 1.0 / float(${out.width_shape});
            float pixelSizeY = 1.0 / float(${out.height_shape}); 
    
            // Horizontal Blur
            vec4 accumulation = vec4(0);
            float weightsum = 0.0;
            for (float i = -kernelSize; i <= kernelSize; i++){
                accumulation += TEXTURE2D(texture_counter, outCoord.xy + vec2(i * pixelSizeX, 0.0)) * weight;
                weightsum += weight;
            }
            // Vertical Blur
            for (float i = -kernelSize; i <= kernelSize; i++){
                accumulation += TEXTURE2D(texture_counter, outCoord.xy + vec2(0.0, i * pixelSizeY)) * weight;
                weightsum += weight;
            }
            
            res = accumulation / weightsum;
            if (origin_alpha > ${THRESHHOLD}) {
                res = counter;
            }
        }
        else if (type == 1) {
            res = counter;
            res.a = origin_alpha;
        }
        else if (type == 2) {
            if (origin_alpha > ${THRESHHOLD}) {
                res = vec4(1.0);
                res.a = origin_alpha;
            }
        }
                
        setPackedOutput(res);
    }
    `;
}

export default {
    mainFunc,
    textureFuncConf: {
        origin: [],
        counter: []
    }
};
