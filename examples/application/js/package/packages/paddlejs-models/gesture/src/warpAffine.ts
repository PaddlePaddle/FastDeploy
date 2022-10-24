const VSHADER_SOURCE = `attribute vec4 a_Position;
    attribute vec2 a_TexCoord;
    uniform mat4 translation;
    varying vec2 v_TexCoord;
    void main() {
        gl_Position = translation * a_Position;
        v_TexCoord = a_TexCoord;
    }`;

const FSHADER_SOURCE = `precision mediump float;
    uniform sampler2D u_Sampler;
    varying vec2 v_TexCoord;
    void main() {
        gl_FragColor = texture2D(u_Sampler, v_TexCoord);
    }`;

let imgData = null;
let mtrData: number[][] = null as unknown as number[][];
let gl: any = null;

function init(output) {
    const canvas = document.createElement('canvas');
    canvas.width = output.width;
    canvas.height = output.height;

    gl = canvas.getContext('webgl') || canvas.getContext('experimental-webgl');
    // 初始化之前先加载图片
    if (!initShaders(gl, VSHADER_SOURCE, FSHADER_SOURCE)) {
        throw new Error('initShaders false');
    }
}
function main(opt) {
    const {
        imageData,
        mtr,
        output
    } = opt;

    mtrData = mtr;
    imgData = imageData;

    gl.clearColor(0, 0, 0, 1);
    gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
    const n = initVertexBuffers(gl);
    initTextures(gl, n, 0);
    const outputData = new Uint8Array(output.width * output.height * 4);
    gl.readPixels(0, 0, output.width, output.height, gl.RGBA, gl.UNSIGNED_BYTE, outputData);
    return outputData;
}

function initVertexBuffers(gl) {
    // 顶点坐标和纹理图像坐标
    const vertices = new Float32Array([
        // webgl坐标，纹理坐标
        -1.0, 1.0, 0.0, 0.0, 1.0,
        -1.0, -1.0, 0.0, 0.0, 0.0,
        1.0, 1.0, 0.0, 1.0, 1.0,
        1.0, -1.0, 0.0, 1.0, 0.0
    ]);

    const FSIZE = vertices.BYTES_PER_ELEMENT;

    const vertexBuffer = gl.createBuffer();

    gl.bindBuffer(gl.ARRAY_BUFFER, vertexBuffer);
    gl.bufferData(gl.ARRAY_BUFFER, vertices, gl.STATIC_DRAW);

    const aPosition = gl.getAttribLocation(gl.program, 'a_Position');
    const aTexCoord = gl.getAttribLocation(gl.program, 'a_TexCoord');

    const xformMatrix = new Float32Array([
        mtrData[0][0], mtrData[1][0], 0.0, 0.0,
        mtrData[0][1], mtrData[1][1], 0.0, 0.0,
        0.0, 0.0, 0.0, 1.0,
        mtrData[0][2], mtrData[1][2], 0.0, 1.0

    ]);

    const translation = gl.getUniformLocation(gl.program, 'translation');
    gl.uniformMatrix4fv(translation, false, xformMatrix);

    gl.vertexAttribPointer(aPosition, 3, gl.FLOAT, false, FSIZE * 5, 0);
    gl.enableVertexAttribArray(aPosition);

    gl.vertexAttribPointer(aTexCoord, 2, gl.FLOAT, false, FSIZE * 5, FSIZE * 3);
    gl.enableVertexAttribArray(aTexCoord);

    return 4;
}


function initTextures(gl, n, index) {
    // 创建纹理对象
    const texture = gl.createTexture();
    const uSampler = gl.getUniformLocation(gl.program, 'u_Sampler');

    // WebGL纹理坐标中的纵轴方向和PNG，JPG等图片容器的Y轴方向是反的，所以先反转Y轴
    gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, 1);

    // 激活纹理单元，开启index号纹理单元
    gl.activeTexture(gl[`TEXTURE${index}`]);

    // 绑定纹理对象
    gl.bindTexture(gl.TEXTURE_2D, texture);

    // 配置纹理对象的参数
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);

    // 将纹理图像分配给纹理对象
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, imgData);

    // 将纹理单元编号传给着色器
    gl.uniform1i(uSampler, index);

    // 绘制
    gl.drawArrays(gl.TRIANGLE_STRIP, 0, n);
}

function initShaders(gl, vshader, fshader) {
    const vertexShader = createShader(gl, vshader, gl.VERTEX_SHADER);
    const fragmentShader = createShader(gl, fshader, gl.FRAGMENT_SHADER);

    const program = gl.createProgram();
    gl.attachShader(program, vertexShader);
    gl.attachShader(program, fragmentShader);

    if (!program) {
        console.log('无法创建程序对象');
        return false;
    }
    gl.linkProgram(program);
    gl.useProgram(program);
    gl.program = program;

    return true;
}

function createShader(gl, sourceCode, type) {
    // Compiles either a shader of type gl.VERTEX_SHADER or gl.FRAGMENT_SHADER
    const shader = gl.createShader(type);
    gl.shaderSource(shader, sourceCode);
    gl.compileShader(shader);

    if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
        const info = gl.getShaderInfoLog(shader);
        throw Error('Could not compile WebGL program. \n\n' + info);
    }
    return shader;
}

export default {
    main,
    init
};
