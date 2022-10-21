/* eslint-disable */
(function (root, factory) {
    if (typeof define === 'function' && define.amd) {
        // AMD. Register as an anonymous module.
        define(function () {
            return (root.cv = factory());
        });
    } else if (typeof module === 'object' && module.exports) {
        // Node. Does not work with strict CommonJS, but
        // only CommonJS-like environments that support module.exports,
        // like Node.
        module.exports = factory();
    } else {
        // Browser globals
        root.cv = factory();
    }
}(this, function () {
    var IsWechat = true;
    var cv = (function () {
        var _scriptDir = typeof document !== 'undefined' && document.currentScript ? document.currentScript.src : undefined;
        if (typeof __filename !== 'undefined') _scriptDir = _scriptDir || __filename;
        return (
            function (cv) {
                cv = cv || {};
                // IsWechat
                var wasmBinaryFile = global.wasm_url;

                var Module = typeof cv !== "undefined" ? cv : {};
                console.log('Module', Module)
                var moduleOverrides = {};
                var key;
                for (key in Module) {
                    if (Module.hasOwnProperty(key)) {
                        moduleOverrides[key] = Module[key]
                    }
                }
                var arguments_ = [];
                var thisProgram = "./this.program";
                var quit_ = function (status, toThrow) {
                    throw toThrow
                };
                var ENVIRONMENT_IS_WEB = false;
                var ENVIRONMENT_IS_WORKER = false;
                var ENVIRONMENT_IS_NODE = false;
                var ENVIRONMENT_IS_SHELL = false;
                ENVIRONMENT_IS_WEB = typeof window === "object";
                ENVIRONMENT_IS_WORKER = typeof importScripts === "function";
                ENVIRONMENT_IS_NODE = typeof process === "object" && typeof process.versions === "object" && typeof process.versions.node === "string";
                ENVIRONMENT_IS_SHELL = !ENVIRONMENT_IS_WEB && !ENVIRONMENT_IS_NODE && !ENVIRONMENT_IS_WORKER;
                var scriptDirectory = "";

                function locateFile(path) {
                    if (Module["locateFile"]) {
                        return Module["locateFile"](path, scriptDirectory)
                    }
                    return scriptDirectory + path
                }

                var read_, readAsync, readBinary, setWindowTitle;
                var nodeFS;
                var nodePath;
                if (ENVIRONMENT_IS_NODE) {
                    if (ENVIRONMENT_IS_WORKER) {
                        scriptDirectory = require("path").dirname(scriptDirectory) + "/"
                    } else {
                        scriptDirectory = __dirname + "/"
                    }
                    read_ = function shell_read(filename, binary) {
                        if (!nodeFS) nodeFS = require("fs");
                        if (!nodePath) nodePath = require("path");
                        filename = nodePath["normalize"](filename);
                        return nodeFS["readFileSync"](filename, binary ? null : "utf8")
                    };
                    readBinary = function readBinary(filename) {
                        var ret = read_(filename, true);
                        if (!ret.buffer) {
                            ret = new Uint8Array(ret)
                        }
                        assert(ret.buffer);
                        return ret
                    };
                    if (process["argv"].length > 1) {
                        thisProgram = process["argv"][1].replace(/\\/g, "/")
                    }
                    arguments_ = process["argv"].slice(2);
                    process["on"]("uncaughtException", function (ex) {
                        if (!(ex instanceof ExitStatus)) {
                            throw ex
                        }
                    });
                    process["on"]("unhandledRejection", abort);
                    quit_ = function (status) {
                        process["exit"](status)
                    };
                    Module["inspect"] = function () {
                        return "[Emscripten Module object]"
                    }
                } else if (!IsWechat && ENVIRONMENT_IS_SHELL) {
                    if (typeof read != "undefined") {
                        read_ = function shell_read(f) {
                            return read(f)
                        }
                    }
                    readBinary = function readBinary(f) {
                        var data;
                        if (typeof readbuffer === "function") {
                            return new Uint8Array(readbuffer(f))
                        }
                        data = read(f, "binary");
                        assert(typeof data === "object");
                        return data
                    };
                    if (typeof scriptArgs != "undefined") {
                        arguments_ = scriptArgs
                    } else if (typeof arguments != "undefined") {
                        arguments_ = arguments
                    }
                    if (typeof quit === "function") {
                        quit_ = function (status) {
                            quit(status)
                        }
                    }
                    if (typeof print !== "undefined") {
                        if (typeof console === "undefined") console = {};
                        console.log = print;
                        console.warn = console.error = typeof printErr !== "undefined" ? printErr : print
                    }
                } else if (IsWechat || ENVIRONMENT_IS_WEB || ENVIRONMENT_IS_WORKER) {
                    if (ENVIRONMENT_IS_WORKER) {
                        scriptDirectory = self.location.href
                    } else if (!IsWechat && document.currentScript) {
                        scriptDirectory = document.currentScript.src
                    }
                    if (_scriptDir) {
                        scriptDirectory = _scriptDir
                    }
                    if (scriptDirectory.indexOf("blob:") !== 0) {
                        scriptDirectory = scriptDirectory.substr(0, scriptDirectory.lastIndexOf("/") + 1)
                    } else {
                        scriptDirectory = ""
                    }
                    {
                        read_ = function shell_read(url) {
                            var xhr = new XMLHttpRequest;
                            xhr.open("GET", url, false);
                            xhr.send(null);
                            return xhr.responseText
                        };
                        if (ENVIRONMENT_IS_WORKER) {
                            readBinary = function readBinary(url) {
                                var xhr = new XMLHttpRequest;
                                xhr.open("GET", url, false);
                                xhr.responseType = "arraybuffer";
                                xhr.send(null);
                                return new Uint8Array(xhr.response)
                            }
                        }
                        readAsync = function readAsync(url, onload, onerror) {
                            var xhr = new XMLHttpRequest;
                            xhr.open("GET", url, true);
                            xhr.responseType = "arraybuffer";
                            xhr.onload = function xhr_onload() {
                                if (xhr.status == 200 || xhr.status == 0 && xhr.response) {
                                    onload(xhr.response);
                                    return
                                }
                                onerror()
                            };
                            xhr.onerror = onerror;
                            xhr.send(null)
                        }
                    }
                    setWindowTitle = function (title) {
                        document.title = title
                    }
                } else {
                }
                var out = Module["print"] || console.log.bind(console);
                var err = Module["printErr"] || console.warn.bind(console);
                for (key in moduleOverrides) {
                    if (moduleOverrides.hasOwnProperty(key)) {
                        Module[key] = moduleOverrides[key]
                    }
                }
                moduleOverrides = null;
                if (Module["arguments"]) arguments_ = Module["arguments"];
                if (Module["thisProgram"]) thisProgram = Module["thisProgram"];
                if (Module["quit"]) quit_ = Module["quit"];
                var STACK_ALIGN = 16;

                function dynamicAlloc(size) {
                    var ret = HEAP32[DYNAMICTOP_PTR >> 2];
                    var end = ret + size + 15 & -16;
                    HEAP32[DYNAMICTOP_PTR >> 2] = end;
                    return ret
                }

                function alignMemory(size, factor) {
                    if (!factor) factor = STACK_ALIGN;
                    return Math.ceil(size / factor) * factor
                }

                function getNativeTypeSize(type) {
                    switch (type) {
                        case"i1":
                        case"i8":
                            return 1;
                        case"i16":
                            return 2;
                        case"i32":
                            return 4;
                        case"i64":
                            return 8;
                        case"float":
                            return 4;
                        case"double":
                            return 8;
                        default: {
                            if (type[type.length - 1] === "*") {
                                return 4
                            } else if (type[0] === "i") {
                                var bits = Number(type.substr(1));
                                assert(bits % 8 === 0, "getNativeTypeSize invalid bits " + bits + ", type " + type);
                                return bits / 8
                            } else {
                                return 0
                            }
                        }
                    }
                }

                function warnOnce(text) {
                    if (!warnOnce.shown) warnOnce.shown = {};
                    if (!warnOnce.shown[text]) {
                        warnOnce.shown[text] = 1;
                        err(text)
                    }
                }

                function convertJsFunctionToWasm(func, sig) {
                    // if (typeof WebAssembly.Function === "function") {
                    //     var typeNames = {"i": "i32", "j": "i64", "f": "f32", "d": "f64"};
                    //     var type = {parameters: [], results: sig[0] == "v" ? [] : [typeNames[sig[0]]]};
                    //     for (var i = 1; i < sig.length; ++i) {
                    //         type.parameters.push(typeNames[sig[i]])
                    //     }
                    //     return new WebAssembly.Function(type, func)
                    // }
                    var typeSection = [1, 0, 1, 96];
                    var sigRet = sig.slice(0, 1);
                    var sigParam = sig.slice(1);
                    var typeCodes = {"i": 127, "j": 126, "f": 125, "d": 124};
                    typeSection.push(sigParam.length);
                    for (var i = 0; i < sigParam.length; ++i) {
                        typeSection.push(typeCodes[sigParam[i]])
                    }
                    if (sigRet == "v") {
                        typeSection.push(0)
                    } else {
                        typeSection = typeSection.concat([1, typeCodes[sigRet]])
                    }
                    typeSection[1] = typeSection.length - 2;
                    var bytes = new Uint8Array([0, 97, 115, 109, 1, 0, 0, 0].concat(typeSection, [2, 7, 1, 1, 101, 1, 102, 0, 0, 7, 5, 1, 1, 102, 0, 0]));
                    var module = new WXWebAssembly.Module(bytes);
                    var instance = new WXWebAssembly.Instance(module, {"e": {"f": func}});
                    var wrappedFunc = instance.exports["f"];
                    return wrappedFunc
                }

                var freeTableIndexes = [];
                var functionsInTableMap;

                function addFunctionWasm(func, sig) {
                    var table = wasmTable;
                    if (!functionsInTableMap) {
                        functionsInTableMap = new WeakMap;
                        for (var i = 0; i < table.length; i++) {
                            var item = table.get(i);
                            if (item) {
                                functionsInTableMap.set(item, i)
                            }
                        }
                    }
                    if (functionsInTableMap.has(func)) {
                        return functionsInTableMap.get(func)
                    }
                    var ret;
                    if (freeTableIndexes.length) {
                        ret = freeTableIndexes.pop()
                    } else {
                        ret = table.length;
                        try {
                            table.grow(1)
                        } catch (err) {
                            if (!(err instanceof RangeError)) {
                                throw err
                            }
                            throw"Unable to grow wasm table. Set ALLOW_TABLE_GROWTH."
                        }
                    }
                    try {
                        table.set(ret, func)
                    } catch (err) {
                        if (!(err instanceof TypeError)) {
                            throw err
                        }
                        var wrapped = convertJsFunctionToWasm(func, sig);
                        table.set(ret, wrapped)
                    }
                    functionsInTableMap.set(func, ret);
                    return ret
                }

                function removeFunctionWasm(index) {
                    functionsInTableMap.delete(wasmTable.get(index));
                    freeTableIndexes.push(index)
                }

                var funcWrappers = {};

                function dynCall(sig, ptr, args) {
                    if (args && args.length) {
                        return Module["dynCall_" + sig].apply(null, [ptr].concat(args))
                    } else {
                        return Module["dynCall_" + sig].call(null, ptr)
                    }
                }

                var tempRet0 = 0;
                var setTempRet0 = function (value) {
                    tempRet0 = value
                };
                var wasmBinary;
                if (Module["wasmBinary"]) wasmBinary = Module["wasmBinary"];
                var noExitRuntime;
                if (Module["noExitRuntime"]) noExitRuntime = Module["noExitRuntime"];
                if (typeof WXWebAssembly !== "object") {
                    abort("no native wasm support detected")
                }

                function setValue(ptr, value, type, noSafe) {
                    type = type || "i8";
                    if (type.charAt(type.length - 1) === "*") type = "i32";
                    switch (type) {
                        case"i1":
                            HEAP8[ptr >> 0] = value;
                            break;
                        case"i8":
                            HEAP8[ptr >> 0] = value;
                            break;
                        case"i16":
                            HEAP16[ptr >> 1] = value;
                            break;
                        case"i32":
                            HEAP32[ptr >> 2] = value;
                            break;
                        case"i64":
                            tempI64 = [value >>> 0, (tempDouble = value, +Math_abs(tempDouble) >= 1 ? tempDouble > 0 ? (Math_min(+Math_floor(tempDouble / 4294967296), 4294967295) | 0) >>> 0 : ~~+Math_ceil((tempDouble - +(~~tempDouble >>> 0)) / 4294967296) >>> 0 : 0)], HEAP32[ptr >> 2] = tempI64[0], HEAP32[ptr + 4 >> 2] = tempI64[1];
                            break;
                        case"float":
                            HEAPF32[ptr >> 2] = value;
                            break;
                        case"double":
                            HEAPF64[ptr >> 3] = value;
                            break;
                        default:
                            abort("invalid type for setValue: " + type)
                    }
                }

                var wasmMemory;
                var wasmTable = new WXWebAssembly.Table({"initial": 1538, "maximum": 1538 + 0, "element": "anyfunc"});
                var ABORT = false;
                var EXITSTATUS = 0;

                function assert(condition, text) {
                    if (!condition) {
                        abort("Assertion failed: " + text)
                    }
                }

                function getCFunc(ident) {
                    var func = Module["_" + ident];
                    assert(func, "Cannot call unknown function " + ident + ", make sure it is exported");
                    return func
                }

                function ccall(ident, returnType, argTypes, args, opts) {
                    var toC = {
                        "string": function (str) {
                            var ret = 0;
                            if (str !== null && str !== undefined && str !== 0) {
                                var len = (str.length << 2) + 1;
                                ret = stackAlloc(len);
                                stringToUTF8(str, ret, len)
                            }
                            return ret
                        }, "array": function (arr) {
                            var ret = stackAlloc(arr.length);
                            writeArrayToMemory(arr, ret);
                            return ret
                        }
                    };

                    function convertReturnValue(ret) {
                        if (returnType === "string") return UTF8ToString(ret);
                        if (returnType === "boolean") return Boolean(ret);
                        return ret
                    }

                    var func = getCFunc(ident);
                    var cArgs = [];
                    var stack = 0;
                    if (args) {
                        for (var i = 0; i < args.length; i++) {
                            var converter = toC[argTypes[i]];
                            if (converter) {
                                if (stack === 0) stack = stackSave();
                                cArgs[i] = converter(args[i])
                            } else {
                                cArgs[i] = args[i]
                            }
                        }
                    }
                    var ret = func.apply(null, cArgs);
                    ret = convertReturnValue(ret);
                    if (stack !== 0) stackRestore(stack);
                    return ret
                }

                var ALLOC_NONE = 3;

                function getMemory(size) {
                    if (!runtimeInitialized) return dynamicAlloc(size);
                    return _malloc(size)
                }

                var UTF8Decoder = typeof TextDecoder !== "undefined" ? new TextDecoder("utf8") : undefined;

                function UTF8ArrayToString(heap, idx, maxBytesToRead) {
                    var endIdx = idx + maxBytesToRead;
                    var endPtr = idx;
                    while (heap[endPtr] && !(endPtr >= endIdx)) ++endPtr;
                    if (endPtr - idx > 16 && heap.subarray && UTF8Decoder) {
                        return UTF8Decoder.decode(heap.subarray(idx, endPtr))
                    } else {
                        var str = "";
                        while (idx < endPtr) {
                            var u0 = heap[idx++];
                            if (!(u0 & 128)) {
                                str += String.fromCharCode(u0);
                                continue
                            }
                            var u1 = heap[idx++] & 63;
                            if ((u0 & 224) == 192) {
                                str += String.fromCharCode((u0 & 31) << 6 | u1);
                                continue
                            }
                            var u2 = heap[idx++] & 63;
                            if ((u0 & 240) == 224) {
                                u0 = (u0 & 15) << 12 | u1 << 6 | u2
                            } else {
                                u0 = (u0 & 7) << 18 | u1 << 12 | u2 << 6 | heap[idx++] & 63
                            }
                            if (u0 < 65536) {
                                str += String.fromCharCode(u0)
                            } else {
                                var ch = u0 - 65536;
                                str += String.fromCharCode(55296 | ch >> 10, 56320 | ch & 1023)
                            }
                        }
                    }
                    return str
                }

                function UTF8ToString(ptr, maxBytesToRead) {
                    return ptr ? UTF8ArrayToString(HEAPU8, ptr, maxBytesToRead) : ""
                }

                function stringToUTF8Array(str, heap, outIdx, maxBytesToWrite) {
                    if (!(maxBytesToWrite > 0)) return 0;
                    var startIdx = outIdx;
                    var endIdx = outIdx + maxBytesToWrite - 1;
                    for (var i = 0; i < str.length; ++i) {
                        var u = str.charCodeAt(i);
                        if (u >= 55296 && u <= 57343) {
                            var u1 = str.charCodeAt(++i);
                            u = 65536 + ((u & 1023) << 10) | u1 & 1023
                        }
                        if (u <= 127) {
                            if (outIdx >= endIdx) break;
                            heap[outIdx++] = u
                        } else if (u <= 2047) {
                            if (outIdx + 1 >= endIdx) break;
                            heap[outIdx++] = 192 | u >> 6;
                            heap[outIdx++] = 128 | u & 63
                        } else if (u <= 65535) {
                            if (outIdx + 2 >= endIdx) break;
                            heap[outIdx++] = 224 | u >> 12;
                            heap[outIdx++] = 128 | u >> 6 & 63;
                            heap[outIdx++] = 128 | u & 63
                        } else {
                            if (outIdx + 3 >= endIdx) break;
                            heap[outIdx++] = 240 | u >> 18;
                            heap[outIdx++] = 128 | u >> 12 & 63;
                            heap[outIdx++] = 128 | u >> 6 & 63;
                            heap[outIdx++] = 128 | u & 63
                        }
                    }
                    heap[outIdx] = 0;
                    return outIdx - startIdx
                }

                function stringToUTF8(str, outPtr, maxBytesToWrite) {
                    return stringToUTF8Array(str, HEAPU8, outPtr, maxBytesToWrite)
                }

                function lengthBytesUTF8(str) {
                    var len = 0;
                    for (var i = 0; i < str.length; ++i) {
                        var u = str.charCodeAt(i);
                        if (u >= 55296 && u <= 57343) u = 65536 + ((u & 1023) << 10) | str.charCodeAt(++i) & 1023;
                        if (u <= 127) ++len; else if (u <= 2047) len += 2; else if (u <= 65535) len += 3; else len += 4
                    }
                    return len
                }

                var UTF16Decoder = typeof TextDecoder !== "undefined" ? new TextDecoder("utf-16le") : undefined;

                function UTF16ToString(ptr, maxBytesToRead) {
                    var endPtr = ptr;
                    var idx = endPtr >> 1;
                    var maxIdx = idx + maxBytesToRead / 2;
                    while (!(idx >= maxIdx) && HEAPU16[idx]) ++idx;
                    endPtr = idx << 1;
                    if (endPtr - ptr > 32 && UTF16Decoder) {
                        return UTF16Decoder.decode(HEAPU8.subarray(ptr, endPtr))
                    } else {
                        var i = 0;
                        var str = "";
                        while (1) {
                            var codeUnit = HEAP16[ptr + i * 2 >> 1];
                            if (codeUnit == 0 || i == maxBytesToRead / 2) return str;
                            ++i;
                            str += String.fromCharCode(codeUnit)
                        }
                    }
                }

                function stringToUTF16(str, outPtr, maxBytesToWrite) {
                    if (maxBytesToWrite === undefined) {
                        maxBytesToWrite = 2147483647
                    }
                    if (maxBytesToWrite < 2) return 0;
                    maxBytesToWrite -= 2;
                    var startPtr = outPtr;
                    var numCharsToWrite = maxBytesToWrite < str.length * 2 ? maxBytesToWrite / 2 : str.length;
                    for (var i = 0; i < numCharsToWrite; ++i) {
                        var codeUnit = str.charCodeAt(i);
                        HEAP16[outPtr >> 1] = codeUnit;
                        outPtr += 2
                    }
                    HEAP16[outPtr >> 1] = 0;
                    return outPtr - startPtr
                }

                function lengthBytesUTF16(str) {
                    return str.length * 2
                }

                function UTF32ToString(ptr, maxBytesToRead) {
                    var i = 0;
                    var str = "";
                    while (!(i >= maxBytesToRead / 4)) {
                        var utf32 = HEAP32[ptr + i * 4 >> 2];
                        if (utf32 == 0) break;
                        ++i;
                        if (utf32 >= 65536) {
                            var ch = utf32 - 65536;
                            str += String.fromCharCode(55296 | ch >> 10, 56320 | ch & 1023)
                        } else {
                            str += String.fromCharCode(utf32)
                        }
                    }
                    return str
                }

                function stringToUTF32(str, outPtr, maxBytesToWrite) {
                    if (maxBytesToWrite === undefined) {
                        maxBytesToWrite = 2147483647
                    }
                    if (maxBytesToWrite < 4) return 0;
                    var startPtr = outPtr;
                    var endPtr = startPtr + maxBytesToWrite - 4;
                    for (var i = 0; i < str.length; ++i) {
                        var codeUnit = str.charCodeAt(i);
                        if (codeUnit >= 55296 && codeUnit <= 57343) {
                            var trailSurrogate = str.charCodeAt(++i);
                            codeUnit = 65536 + ((codeUnit & 1023) << 10) | trailSurrogate & 1023
                        }
                        HEAP32[outPtr >> 2] = codeUnit;
                        outPtr += 4;
                        if (outPtr + 4 > endPtr) break
                    }
                    HEAP32[outPtr >> 2] = 0;
                    return outPtr - startPtr
                }

                function lengthBytesUTF32(str) {
                    var len = 0;
                    for (var i = 0; i < str.length; ++i) {
                        var codeUnit = str.charCodeAt(i);
                        if (codeUnit >= 55296 && codeUnit <= 57343) ++i;
                        len += 4
                    }
                    return len
                }

                function writeArrayToMemory(array, buffer) {
                    HEAP8.set(array, buffer)
                }

                function writeAsciiToMemory(str, buffer, dontAddNull) {
                    for (var i = 0; i < str.length; ++i) {
                        HEAP8[buffer++ >> 0] = str.charCodeAt(i)
                    }
                    if (!dontAddNull) HEAP8[buffer >> 0] = 0
                }

                var WASM_PAGE_SIZE = 65536;

                function alignUp(x, multiple) {
                    if (x % multiple > 0) {
                        x += multiple - x % multiple
                    }
                    return x
                }

                var buffer, HEAP8, HEAPU8, HEAP16, HEAPU16, HEAP32, HEAPU32, HEAPF32, HEAPF64;

                function updateGlobalBufferAndViews(buf) {
                    buffer = buf;
                    Module["HEAP8"] = HEAP8 = new Int8Array(buf);
                    Module["HEAP16"] = HEAP16 = new Int16Array(buf);
                    Module["HEAP32"] = HEAP32 = new Int32Array(buf);
                    Module["HEAPU8"] = HEAPU8 = new Uint8Array(buf);
                    Module["HEAPU16"] = HEAPU16 = new Uint16Array(buf);
                    Module["HEAPU32"] = HEAPU32 = new Uint32Array(buf);
                    Module["HEAPF32"] = HEAPF32 = new Float32Array(buf);
                    Module["HEAPF64"] = HEAPF64 = new Float64Array(buf)
                }

                var STACK_BASE = 5885696, DYNAMIC_BASE = 5885696, DYNAMICTOP_PTR = 642656;
                var INITIAL_INITIAL_MEMORY = Module["INITIAL_MEMORY"] || 134217728;
                if (Module["wasmMemory"]) {
                    wasmMemory = Module["wasmMemory"]
                } else {
                    wasmMemory = new WXWebAssembly.Memory({
                        "initial": INITIAL_INITIAL_MEMORY / WASM_PAGE_SIZE,
                        "maximum": 2147483648 / WASM_PAGE_SIZE
                    })
                }
                if (wasmMemory) {
                    buffer = wasmMemory.buffer
                }
                INITIAL_INITIAL_MEMORY = buffer.byteLength;
                updateGlobalBufferAndViews(buffer);
                HEAP32[DYNAMICTOP_PTR >> 2] = DYNAMIC_BASE;

                function callRuntimeCallbacks(callbacks) {
                    while (callbacks.length > 0) {
                        var callback = callbacks.shift();
                        if (typeof callback == "function") {
                            callback(Module);
                            continue
                        }
                        var func = callback.func;
                        if (typeof func === "number") {
                            if (callback.arg === undefined) {
                                Module["dynCall_v"](func)
                            } else {
                                Module["dynCall_vi"](func, callback.arg)
                            }
                        } else {
                            func(callback.arg === undefined ? null : callback.arg)
                        }
                    }
                }

                var __ATPRERUN__ = [];
                var __ATINIT__ = [];
                var __ATMAIN__ = [];
                var __ATPOSTRUN__ = [];
                var runtimeInitialized = false;
                var runtimeExited = false;

                function preRun() {
                    if (Module["preRun"]) {
                        if (typeof Module["preRun"] == "function") Module["preRun"] = [Module["preRun"]];
                        while (Module["preRun"].length) {
                            addOnPreRun(Module["preRun"].shift())
                        }
                    }
                    callRuntimeCallbacks(__ATPRERUN__)
                }

                function initRuntime() {
                    runtimeInitialized = true;
                    if (!Module["noFSInit"] && !FS.init.initialized) FS.init();
                    TTY.init();
                    callRuntimeCallbacks(__ATINIT__)
                }

                function preMain() {
                    FS.ignorePermissions = false;
                    callRuntimeCallbacks(__ATMAIN__)
                }

                function exitRuntime() {
                    runtimeExited = true
                }

                function postRun() {
                    if (Module["postRun"]) {
                        if (typeof Module["postRun"] == "function") Module["postRun"] = [Module["postRun"]];
                        while (Module["postRun"].length) {
                            addOnPostRun(Module["postRun"].shift())
                        }
                    }
                    callRuntimeCallbacks(__ATPOSTRUN__)
                }

                function addOnPreRun(cb) {
                    __ATPRERUN__.unshift(cb)
                }

                function addOnPostRun(cb) {
                    __ATPOSTRUN__.unshift(cb)
                }

                var Math_abs = Math.abs;
                var Math_ceil = Math.ceil;
                var Math_floor = Math.floor;
                var Math_min = Math.min;
                var runDependencies = 0;
                var runDependencyWatcher = null;
                var dependenciesFulfilled = null;

                function getUniqueRunDependency(id) {
                    return id
                }

                function addRunDependency(id) {
                    runDependencies++;
                    if (Module["monitorRunDependencies"]) {
                        Module["monitorRunDependencies"](runDependencies)
                    }
                }

                function removeRunDependency(id) {
                    runDependencies--;
                    if (Module["monitorRunDependencies"]) {
                        Module["monitorRunDependencies"](runDependencies)
                    }
                    if (runDependencies == 0) {
                        if (runDependencyWatcher !== null) {
                            clearInterval(runDependencyWatcher);
                            runDependencyWatcher = null
                        }
                        if (dependenciesFulfilled) {
                            var callback = dependenciesFulfilled;
                            dependenciesFulfilled = null;
                            callback()
                        }
                    }
                }

                Module["preloadedImages"] = {};
                Module["preloadedAudios"] = {};

                function abort(what) {
                    if (Module["onAbort"]) {
                        Module["onAbort"](what)
                    }
                    what += "";
                    err(what);
                    ABORT = true;
                    EXITSTATUS = 1;
                    what = "abort(" + what + "). Build with -s ASSERTIONS=1 for more info.";
                    throw what
                }

                function hasPrefix(str, prefix) {
                    return String.prototype.startsWith ? str.startsWith(prefix) : str.indexOf(prefix) === 0
                }

                var dataURIPrefix = "data:application/octet-stream;base64,";

                function isDataURI(filename) {
                    return hasPrefix(filename, dataURIPrefix)
                }

                var fileURIPrefix = "file://";

                function isFileURI(filename) {
                    return hasPrefix(filename, fileURIPrefix)
                }

                // var wasmBinaryFile = "opencv_js.wasm";
                if (!isDataURI(wasmBinaryFile)) {
                    wasmBinaryFile = locateFile(wasmBinaryFile)
                }

                function getBinary() {
                    try {
                        if (wasmBinary) {
                            return new Uint8Array(wasmBinary)
                        }
                        if (readBinary) {
                            return readBinary(wasmBinaryFile)
                        } else {
                            throw"both async and sync fetching of the wasm failed"
                        }
                    } catch (err) {
                        abort(err)
                    }
                }

                function getBinaryPromise() {
                    if (!wasmBinary && (ENVIRONMENT_IS_WEB || ENVIRONMENT_IS_WORKER) && typeof fetch === "function" && !isFileURI(wasmBinaryFile)) {
                        return fetch(wasmBinaryFile, {credentials: "same-origin"}).then(function (response) {
                            if (!response["ok"]) {
                                throw"failed to load wasm binary file at '" + wasmBinaryFile + "'"
                            }
                            return response["arrayBuffer"]()
                        }).catch(function () {
                            return getBinary()
                        })
                    }
                    return new Promise(function (resolve, reject) {
                        resolve(getBinary())
                    })
                }

                function createWasm() {
                    var info = {"env": asmLibraryArg, "wasi_snapshot_preview1": asmLibraryArg};

                    function receiveInstance(instance, module) {
                        var exports = instance.exports;
                        Module["asm"] = exports;
                        removeRunDependency("wasm-instantiate")
                    }

                    addRunDependency("wasm-instantiate");

                    function receiveInstantiatedSource(output) {
                        receiveInstance(output["instance"])
                    }

                    function instantiateArrayBuffer(receiver) {
                        return getBinaryPromise().then(function (binary) {
                            return WXWebAssembly.instantiate(binary, info)
                        }).then(receiver, function (reason) {
                            err("failed to asynchronously prepare wasm: " + reason);
                            abort(reason)
                        })
                    }

                    function instantiateAsync() {
                        if (IsWechat) {
                            var result = WXWebAssembly.instantiate(wasmBinaryFile, info);

                            return result.then(receiveInstantiatedSource, function (reason) {
                                err("wasm streaming compile failed: " + reason);
                                err("falling back to ArrayBuffer instantiation");
                                return instantiateArrayBuffer(receiveInstantiatedSource)
                            });

                        } else {
                            if (!wasmBinary && typeof WebAssembly.instantiateStreaming === "function" && !isDataURI(wasmBinaryFile) && !isFileURI(wasmBinaryFile) && typeof fetch === "function") {
                                fetch(wasmBinaryFile, {credentials: "same-origin"}).then(function (response) {
                                    var result = WebAssembly.instantiateStreaming(response, info);
                                    return result.then(receiveInstantiatedSource, function (reason) {
                                        err("wasm streaming compile failed: " + reason);
                                        err("falling back to ArrayBuffer instantiation");
                                        return instantiateArrayBuffer(receiveInstantiatedSource)
                                    })
                                })
                            } else {
                                return instantiateArrayBuffer(receiveInstantiatedSource)
                            }
                        }
                    }

                    if (Module["instantiateWasm"]) {
                        try {
                            var exports = Module["instantiateWasm"](info, receiveInstance);
                            return exports
                        } catch (e) {
                            err("Module.instantiateWasm callback failed with error: " + e);
                            return false
                        }
                    }
                    instantiateAsync();
                    return {}
                }

                var tempDouble;
                var tempI64;
                __ATINIT__.push({
                    func: function () {
                        ___wasm_call_ctors()
                    }
                });

                function _emscripten_set_main_loop_timing(mode, value) {
                    Browser.mainLoop.timingMode = mode;
                    Browser.mainLoop.timingValue = value;
                    if (!Browser.mainLoop.func) {
                        return 1
                    }
                    if (mode == 0) {
                        Browser.mainLoop.scheduler = function Browser_mainLoop_scheduler_setTimeout() {
                            var timeUntilNextTick = Math.max(0, Browser.mainLoop.tickStartTime + value - _emscripten_get_now()) | 0;
                            setTimeout(Browser.mainLoop.runner, timeUntilNextTick)
                        };
                        Browser.mainLoop.method = "timeout"
                    } else if (mode == 1) {
                        Browser.mainLoop.scheduler = function Browser_mainLoop_scheduler_rAF() {
                            Browser.requestAnimationFrame(Browser.mainLoop.runner)
                        };
                        Browser.mainLoop.method = "rAF"
                    } else if (mode == 2) {
                        if (typeof setImmediate === "undefined") {
                            var setImmediates = [];
                            var emscriptenMainLoopMessageId = "setimmediate";
                            var Browser_setImmediate_messageHandler = function (event) {
                                if (event.data === emscriptenMainLoopMessageId || event.data.target === emscriptenMainLoopMessageId) {
                                    event.stopPropagation();
                                    setImmediates.shift()()
                                }
                            };
                            addEventListener("message", Browser_setImmediate_messageHandler, true);
                            setImmediate = function Browser_emulated_setImmediate(func) {
                                setImmediates.push(func);
                                if (ENVIRONMENT_IS_WORKER) {
                                    if (Module["setImmediates"] === undefined) Module["setImmediates"] = [];
                                    Module["setImmediates"].push(func);
                                    postMessage({target: emscriptenMainLoopMessageId})
                                } else postMessage(emscriptenMainLoopMessageId, "*")
                            }
                        }
                        Browser.mainLoop.scheduler = function Browser_mainLoop_scheduler_setImmediate() {
                            setImmediate(Browser.mainLoop.runner)
                        };
                        Browser.mainLoop.method = "immediate"
                    }
                    return 0
                }

                var _emscripten_get_now;
                if (ENVIRONMENT_IS_NODE) {
                    _emscripten_get_now = function () {
                        var t = process["hrtime"]();
                        return t[0] * 1e3 + t[1] / 1e6
                    }
                } else if (typeof dateNow !== "undefined") {
                    _emscripten_get_now = dateNow
                } else _emscripten_get_now = function () {
                    return performance.now()
                };

                function _emscripten_set_main_loop(func, fps, simulateInfiniteLoop, arg, noSetTiming) {
                    noExitRuntime = true;
                    assert(!Browser.mainLoop.func, "emscripten_set_main_loop: there can only be one main loop function at once: call emscripten_cancel_main_loop to cancel the previous one before setting a new one with different parameters.");
                    Browser.mainLoop.func = func;
                    Browser.mainLoop.arg = arg;
                    var browserIterationFunc;
                    if (typeof arg !== "undefined") {
                        browserIterationFunc = function () {
                            Module["dynCall_vi"](func, arg)
                        }
                    } else {
                        browserIterationFunc = function () {
                            Module["dynCall_v"](func)
                        }
                    }
                    var thisMainLoopId = Browser.mainLoop.currentlyRunningMainloop;
                    Browser.mainLoop.runner = function Browser_mainLoop_runner() {
                        if (ABORT) return;
                        if (Browser.mainLoop.queue.length > 0) {
                            var start = Date.now();
                            var blocker = Browser.mainLoop.queue.shift();
                            blocker.func(blocker.arg);
                            if (Browser.mainLoop.remainingBlockers) {
                                var remaining = Browser.mainLoop.remainingBlockers;
                                var next = remaining % 1 == 0 ? remaining - 1 : Math.floor(remaining);
                                if (blocker.counted) {
                                    Browser.mainLoop.remainingBlockers = next
                                } else {
                                    next = next + .5;
                                    Browser.mainLoop.remainingBlockers = (8 * remaining + next) / 9
                                }
                            }
                            console.log('main loop blocker "' + blocker.name + '" took ' + (Date.now() - start) + " ms");
                            Browser.mainLoop.updateStatus();
                            if (thisMainLoopId < Browser.mainLoop.currentlyRunningMainloop) return;
                            setTimeout(Browser.mainLoop.runner, 0);
                            return
                        }
                        if (thisMainLoopId < Browser.mainLoop.currentlyRunningMainloop) return;
                        Browser.mainLoop.currentFrameNumber = Browser.mainLoop.currentFrameNumber + 1 | 0;
                        if (Browser.mainLoop.timingMode == 1 && Browser.mainLoop.timingValue > 1 && Browser.mainLoop.currentFrameNumber % Browser.mainLoop.timingValue != 0) {
                            Browser.mainLoop.scheduler();
                            return
                        } else if (Browser.mainLoop.timingMode == 0) {
                            Browser.mainLoop.tickStartTime = _emscripten_get_now()
                        }
                        Browser.mainLoop.runIter(browserIterationFunc);
                        if (thisMainLoopId < Browser.mainLoop.currentlyRunningMainloop) return;
                        if (typeof SDL === "object" && SDL.audio && SDL.audio.queueNewAudioData) SDL.audio.queueNewAudioData();
                        Browser.mainLoop.scheduler()
                    };
                    if (!noSetTiming) {
                        if (fps && fps > 0) _emscripten_set_main_loop_timing(0, 1e3 / fps); else _emscripten_set_main_loop_timing(1, 1);
                        Browser.mainLoop.scheduler()
                    }
                    if (simulateInfiniteLoop) {
                        throw"unwind"
                    }
                }

                var Browser = {
                    mainLoop: {
                        scheduler: null,
                        method: "",
                        currentlyRunningMainloop: 0,
                        func: null,
                        arg: 0,
                        timingMode: 0,
                        timingValue: 0,
                        currentFrameNumber: 0,
                        queue: [],
                        pause: function () {
                            Browser.mainLoop.scheduler = null;
                            Browser.mainLoop.currentlyRunningMainloop++
                        },
                        resume: function () {
                            Browser.mainLoop.currentlyRunningMainloop++;
                            var timingMode = Browser.mainLoop.timingMode;
                            var timingValue = Browser.mainLoop.timingValue;
                            var func = Browser.mainLoop.func;
                            Browser.mainLoop.func = null;
                            _emscripten_set_main_loop(func, 0, false, Browser.mainLoop.arg, true);
                            _emscripten_set_main_loop_timing(timingMode, timingValue);
                            Browser.mainLoop.scheduler()
                        },
                        updateStatus: function () {
                            if (Module["setStatus"]) {
                                var message = Module["statusMessage"] || "Please wait...";
                                var remaining = Browser.mainLoop.remainingBlockers;
                                var expected = Browser.mainLoop.expectedBlockers;
                                if (remaining) {
                                    if (remaining < expected) {
                                        Module["setStatus"](message + " (" + (expected - remaining) + "/" + expected + ")")
                                    } else {
                                        Module["setStatus"](message)
                                    }
                                } else {
                                    Module["setStatus"]("")
                                }
                            }
                        },
                        runIter: function (func) {
                            if (ABORT) return;
                            if (Module["preMainLoop"]) {
                                var preRet = Module["preMainLoop"]();
                                if (preRet === false) {
                                    return
                                }
                            }
                            try {
                                func()
                            } catch (e) {
                                if (e instanceof ExitStatus) {
                                    return
                                } else {
                                    if (e && typeof e === "object" && e.stack) err("exception thrown: " + [e, e.stack]);
                                    throw e
                                }
                            }
                            if (Module["postMainLoop"]) Module["postMainLoop"]()
                        }
                    },
                    isFullscreen: false,
                    pointerLock: false,
                    moduleContextCreatedCallbacks: [],
                    workers: [],
                    init: function () {
                        if (!Module["preloadPlugins"]) Module["preloadPlugins"] = [];
                        if (Browser.initted) return;
                        Browser.initted = true;
                        try {
                            new Blob;
                            Browser.hasBlobConstructor = true
                        } catch (e) {
                            Browser.hasBlobConstructor = false;
                            console.log("warning: no blob constructor, cannot create blobs with mimetypes")
                        }
                        Browser.BlobBuilder = typeof MozBlobBuilder != "undefined" ? MozBlobBuilder : typeof WebKitBlobBuilder != "undefined" ? WebKitBlobBuilder : !Browser.hasBlobConstructor ? console.log("warning: no BlobBuilder") : null;
                        Browser.URLObject = typeof window != "undefined" ? window.URL ? window.URL : window.webkitURL : undefined;
                        if (!Module.noImageDecoding && typeof Browser.URLObject === "undefined") {
                            console.log("warning: Browser does not support creating object URLs. Built-in browser image decoding will not be available.");
                            Module.noImageDecoding = true
                        }
                        var imagePlugin = {};
                        imagePlugin["canHandle"] = function imagePlugin_canHandle(name) {
                            return !Module.noImageDecoding && /\.(jpg|jpeg|png|bmp)$/i.test(name)
                        };
                        imagePlugin["handle"] = function imagePlugin_handle(byteArray, name, onload, onerror) {
                            var b = null;
                            if (Browser.hasBlobConstructor) {
                                try {
                                    b = new Blob([byteArray], {type: Browser.getMimetype(name)});
                                    if (b.size !== byteArray.length) {
                                        b = new Blob([new Uint8Array(byteArray).buffer], {type: Browser.getMimetype(name)})
                                    }
                                } catch (e) {
                                    warnOnce("Blob constructor present but fails: " + e + "; falling back to blob builder")
                                }
                            }
                            if (!b) {
                                var bb = new Browser.BlobBuilder;
                                bb.append(new Uint8Array(byteArray).buffer);
                                b = bb.getBlob()
                            }
                            var url = Browser.URLObject.createObjectURL(b);
                            var img = new Image;
                            img.onload = function img_onload() {
                                assert(img.complete, "Image " + name + " could not be decoded");
                                var canvas = document.createElement("canvas");
                                canvas.width = img.width;
                                canvas.height = img.height;
                                var ctx = canvas.getContext("2d");
                                ctx.drawImage(img, 0, 0);
                                Module["preloadedImages"][name] = canvas;
                                Browser.URLObject.revokeObjectURL(url);
                                if (onload) onload(byteArray)
                            };
                            img.onerror = function img_onerror(event) {
                                console.log("Image " + url + " could not be decoded");
                                if (onerror) onerror()
                            };
                            img.src = url
                        };
                        Module["preloadPlugins"].push(imagePlugin);
                        var audioPlugin = {};
                        audioPlugin["canHandle"] = function audioPlugin_canHandle(name) {
                            return !Module.noAudioDecoding && name.substr(-4) in {".ogg": 1, ".wav": 1, ".mp3": 1}
                        };
                        audioPlugin["handle"] = function audioPlugin_handle(byteArray, name, onload, onerror) {
                            var done = false;

                            function finish(audio) {
                                if (done) return;
                                done = true;
                                Module["preloadedAudios"][name] = audio;
                                if (onload) onload(byteArray)
                            }

                            function fail() {
                                if (done) return;
                                done = true;
                                Module["preloadedAudios"][name] = new Audio;
                                if (onerror) onerror()
                            }

                            if (Browser.hasBlobConstructor) {
                                try {
                                    var b = new Blob([byteArray], {type: Browser.getMimetype(name)})
                                } catch (e) {
                                    return fail()
                                }
                                var url = Browser.URLObject.createObjectURL(b);
                                var audio = new Audio;
                                audio.addEventListener("canplaythrough", function () {
                                    finish(audio)
                                }, false);
                                audio.onerror = function audio_onerror(event) {
                                    if (done) return;
                                    console.log("warning: browser could not fully decode audio " + name + ", trying slower base64 approach");

                                    function encode64(data) {
                                        var BASE = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
                                        var PAD = "=";
                                        var ret = "";
                                        var leftchar = 0;
                                        var leftbits = 0;
                                        for (var i = 0; i < data.length; i++) {
                                            leftchar = leftchar << 8 | data[i];
                                            leftbits += 8;
                                            while (leftbits >= 6) {
                                                var curr = leftchar >> leftbits - 6 & 63;
                                                leftbits -= 6;
                                                ret += BASE[curr]
                                            }
                                        }
                                        if (leftbits == 2) {
                                            ret += BASE[(leftchar & 3) << 4];
                                            ret += PAD + PAD
                                        } else if (leftbits == 4) {
                                            ret += BASE[(leftchar & 15) << 2];
                                            ret += PAD
                                        }
                                        return ret
                                    }

                                    audio.src = "data:audio/x-" + name.substr(-3) + ";base64," + encode64(byteArray);
                                    finish(audio)
                                };
                                audio.src = url;
                                Browser.safeSetTimeout(function () {
                                    finish(audio)
                                }, 1e4)
                            } else {
                                return fail()
                            }
                        };
                        Module["preloadPlugins"].push(audioPlugin);

                        function pointerLockChange() {
                            Browser.pointerLock = document["pointerLockElement"] === Module["canvas"] || document["mozPointerLockElement"] === Module["canvas"] || document["webkitPointerLockElement"] === Module["canvas"] || document["msPointerLockElement"] === Module["canvas"]
                        }

                        var canvas = Module["canvas"];
                        if (canvas) {
                            canvas.requestPointerLock = canvas["requestPointerLock"] || canvas["mozRequestPointerLock"] || canvas["webkitRequestPointerLock"] || canvas["msRequestPointerLock"] || function () {
                            };
                            canvas.exitPointerLock = document["exitPointerLock"] || document["mozExitPointerLock"] || document["webkitExitPointerLock"] || document["msExitPointerLock"] || function () {
                            };
                            canvas.exitPointerLock = canvas.exitPointerLock.bind(document);
                            document.addEventListener("pointerlockchange", pointerLockChange, false);
                            document.addEventListener("mozpointerlockchange", pointerLockChange, false);
                            document.addEventListener("webkitpointerlockchange", pointerLockChange, false);
                            document.addEventListener("mspointerlockchange", pointerLockChange, false);
                            if (Module["elementPointerLock"]) {
                                canvas.addEventListener("click", function (ev) {
                                    if (!Browser.pointerLock && Module["canvas"].requestPointerLock) {
                                        Module["canvas"].requestPointerLock();
                                        ev.preventDefault()
                                    }
                                }, false)
                            }
                        }
                    },
                    createContext: function (canvas, useWebGL, setInModule, webGLContextAttributes) {
                        if (useWebGL && Module.ctx && canvas == Module.canvas) return Module.ctx;
                        var ctx;
                        var contextHandle;
                        if (useWebGL) {
                            var contextAttributes = {antialias: false, alpha: false, majorVersion: 1};
                            if (webGLContextAttributes) {
                                for (var attribute in webGLContextAttributes) {
                                    contextAttributes[attribute] = webGLContextAttributes[attribute]
                                }
                            }
                            if (typeof GL !== "undefined") {
                                contextHandle = GL.createContext(canvas, contextAttributes);
                                if (contextHandle) {
                                    ctx = GL.getContext(contextHandle).GLctx
                                }
                            }
                        } else {
                            ctx = canvas.getContext("2d")
                        }
                        if (!ctx) return null;
                        if (setInModule) {
                            if (!useWebGL) assert(typeof GLctx === "undefined", "cannot set in module if GLctx is used, but we are a non-GL context that would replace it");
                            Module.ctx = ctx;
                            if (useWebGL) GL.makeContextCurrent(contextHandle);
                            Module.useWebGL = useWebGL;
                            Browser.moduleContextCreatedCallbacks.forEach(function (callback) {
                                callback()
                            });
                            Browser.init()
                        }
                        return ctx
                    },
                    destroyContext: function (canvas, useWebGL, setInModule) {
                    },
                    fullscreenHandlersInstalled: false,
                    lockPointer: undefined,
                    resizeCanvas: undefined,
                    requestFullscreen: function (lockPointer, resizeCanvas) {
                        Browser.lockPointer = lockPointer;
                        Browser.resizeCanvas = resizeCanvas;
                        if (typeof Browser.lockPointer === "undefined") Browser.lockPointer = true;
                        if (typeof Browser.resizeCanvas === "undefined") Browser.resizeCanvas = false;
                        var canvas = Module["canvas"];

                        function fullscreenChange() {
                            Browser.isFullscreen = false;
                            var canvasContainer = canvas.parentNode;
                            if ((document["fullscreenElement"] || document["mozFullScreenElement"] || document["msFullscreenElement"] || document["webkitFullscreenElement"] || document["webkitCurrentFullScreenElement"]) === canvasContainer) {
                                canvas.exitFullscreen = Browser.exitFullscreen;
                                if (Browser.lockPointer) canvas.requestPointerLock();
                                Browser.isFullscreen = true;
                                if (Browser.resizeCanvas) {
                                    Browser.setFullscreenCanvasSize()
                                } else {
                                    Browser.updateCanvasDimensions(canvas)
                                }
                            } else {
                                canvasContainer.parentNode.insertBefore(canvas, canvasContainer);
                                canvasContainer.parentNode.removeChild(canvasContainer);
                                if (Browser.resizeCanvas) {
                                    Browser.setWindowedCanvasSize()
                                } else {
                                    Browser.updateCanvasDimensions(canvas)
                                }
                            }
                            if (Module["onFullScreen"]) Module["onFullScreen"](Browser.isFullscreen);
                            if (Module["onFullscreen"]) Module["onFullscreen"](Browser.isFullscreen)
                        }

                        if (!Browser.fullscreenHandlersInstalled) {
                            Browser.fullscreenHandlersInstalled = true;
                            document.addEventListener("fullscreenchange", fullscreenChange, false);
                            document.addEventListener("mozfullscreenchange", fullscreenChange, false);
                            document.addEventListener("webkitfullscreenchange", fullscreenChange, false);
                            document.addEventListener("MSFullscreenChange", fullscreenChange, false)
                        }
                        var canvasContainer = document.createElement("div");
                        canvas.parentNode.insertBefore(canvasContainer, canvas);
                        canvasContainer.appendChild(canvas);
                        canvasContainer.requestFullscreen = canvasContainer["requestFullscreen"] || canvasContainer["mozRequestFullScreen"] || canvasContainer["msRequestFullscreen"] || (canvasContainer["webkitRequestFullscreen"] ? function () {
                            canvasContainer["webkitRequestFullscreen"](Element["ALLOW_KEYBOARD_INPUT"])
                        } : null) || (canvasContainer["webkitRequestFullScreen"] ? function () {
                            canvasContainer["webkitRequestFullScreen"](Element["ALLOW_KEYBOARD_INPUT"])
                        } : null);
                        canvasContainer.requestFullscreen()
                    },
                    exitFullscreen: function () {
                        if (!Browser.isFullscreen) {
                            return false
                        }
                        var CFS = document["exitFullscreen"] || document["cancelFullScreen"] || document["mozCancelFullScreen"] || document["msExitFullscreen"] || document["webkitCancelFullScreen"] || function () {
                        };
                        CFS.apply(document, []);
                        return true
                    },
                    nextRAF: 0,
                    fakeRequestAnimationFrame: function (func) {
                        var now = Date.now();
                        if (Browser.nextRAF === 0) {
                            Browser.nextRAF = now + 1e3 / 60
                        } else {
                            while (now + 2 >= Browser.nextRAF) {
                                Browser.nextRAF += 1e3 / 60
                            }
                        }
                        var delay = Math.max(Browser.nextRAF - now, 0);
                        setTimeout(func, delay)
                    },
                    requestAnimationFrame: function (func) {
                        if (typeof requestAnimationFrame === "function") {
                            requestAnimationFrame(func);
                            return
                        }
                        var RAF = Browser.fakeRequestAnimationFrame;
                        RAF(func)
                    },
                    safeCallback: function (func) {
                        return function () {
                            if (!ABORT) return func.apply(null, arguments)
                        }
                    },
                    allowAsyncCallbacks: true,
                    queuedAsyncCallbacks: [],
                    pauseAsyncCallbacks: function () {
                        Browser.allowAsyncCallbacks = false
                    },
                    resumeAsyncCallbacks: function () {
                        Browser.allowAsyncCallbacks = true;
                        if (Browser.queuedAsyncCallbacks.length > 0) {
                            var callbacks = Browser.queuedAsyncCallbacks;
                            Browser.queuedAsyncCallbacks = [];
                            callbacks.forEach(function (func) {
                                func()
                            })
                        }
                    },
                    safeRequestAnimationFrame: function (func) {
                        return Browser.requestAnimationFrame(function () {
                            if (ABORT) return;
                            if (Browser.allowAsyncCallbacks) {
                                func()
                            } else {
                                Browser.queuedAsyncCallbacks.push(func)
                            }
                        })
                    },
                    safeSetTimeout: function (func, timeout) {
                        noExitRuntime = true;
                        return setTimeout(function () {
                            if (ABORT) return;
                            if (Browser.allowAsyncCallbacks) {
                                func()
                            } else {
                                Browser.queuedAsyncCallbacks.push(func)
                            }
                        }, timeout)
                    },
                    safeSetInterval: function (func, timeout) {
                        noExitRuntime = true;
                        return setInterval(function () {
                            if (ABORT) return;
                            if (Browser.allowAsyncCallbacks) {
                                func()
                            }
                        }, timeout)
                    },
                    getMimetype: function (name) {
                        return {
                            "jpg": "image/jpeg",
                            "jpeg": "image/jpeg",
                            "png": "image/png",
                            "bmp": "image/bmp",
                            "ogg": "audio/ogg",
                            "wav": "audio/wav",
                            "mp3": "audio/mpeg"
                        }[name.substr(name.lastIndexOf(".") + 1)]
                    },
                    getUserMedia: function (func) {
                        if (!window.getUserMedia) {
                            window.getUserMedia = navigator["getUserMedia"] || navigator["mozGetUserMedia"]
                        }
                        window.getUserMedia(func)
                    },
                    getMovementX: function (event) {
                        return event["movementX"] || event["mozMovementX"] || event["webkitMovementX"] || 0
                    },
                    getMovementY: function (event) {
                        return event["movementY"] || event["mozMovementY"] || event["webkitMovementY"] || 0
                    },
                    getMouseWheelDelta: function (event) {
                        var delta = 0;
                        switch (event.type) {
                            case"DOMMouseScroll":
                                delta = event.detail / 3;
                                break;
                            case"mousewheel":
                                delta = event.wheelDelta / 120;
                                break;
                            case"wheel":
                                delta = event.deltaY;
                                switch (event.deltaMode) {
                                    case 0:
                                        delta /= 100;
                                        break;
                                    case 1:
                                        delta /= 3;
                                        break;
                                    case 2:
                                        delta *= 80;
                                        break;
                                    default:
                                        throw"unrecognized mouse wheel delta mode: " + event.deltaMode
                                }
                                break;
                            default:
                                throw"unrecognized mouse wheel event: " + event.type
                        }
                        return delta
                    },
                    mouseX: 0,
                    mouseY: 0,
                    mouseMovementX: 0,
                    mouseMovementY: 0,
                    touches: {},
                    lastTouches: {},
                    calculateMouseEvent: function (event) {
                        if (Browser.pointerLock) {
                            if (event.type != "mousemove" && "mozMovementX" in event) {
                                Browser.mouseMovementX = Browser.mouseMovementY = 0
                            } else {
                                Browser.mouseMovementX = Browser.getMovementX(event);
                                Browser.mouseMovementY = Browser.getMovementY(event)
                            }
                            if (typeof SDL != "undefined") {
                                Browser.mouseX = SDL.mouseX + Browser.mouseMovementX;
                                Browser.mouseY = SDL.mouseY + Browser.mouseMovementY
                            } else {
                                Browser.mouseX += Browser.mouseMovementX;
                                Browser.mouseY += Browser.mouseMovementY
                            }
                        } else {
                            var rect = Module["canvas"].getBoundingClientRect();
                            var cw = Module["canvas"].width;
                            var ch = Module["canvas"].height;
                            var scrollX = typeof window.scrollX !== "undefined" ? window.scrollX : window.pageXOffset;
                            var scrollY = typeof window.scrollY !== "undefined" ? window.scrollY : window.pageYOffset;
                            if (event.type === "touchstart" || event.type === "touchend" || event.type === "touchmove") {
                                var touch = event.touch;
                                if (touch === undefined) {
                                    return
                                }
                                var adjustedX = touch.pageX - (scrollX + rect.left);
                                var adjustedY = touch.pageY - (scrollY + rect.top);
                                adjustedX = adjustedX * (cw / rect.width);
                                adjustedY = adjustedY * (ch / rect.height);
                                var coords = {x: adjustedX, y: adjustedY};
                                if (event.type === "touchstart") {
                                    Browser.lastTouches[touch.identifier] = coords;
                                    Browser.touches[touch.identifier] = coords
                                } else if (event.type === "touchend" || event.type === "touchmove") {
                                    var last = Browser.touches[touch.identifier];
                                    if (!last) last = coords;
                                    Browser.lastTouches[touch.identifier] = last;
                                    Browser.touches[touch.identifier] = coords
                                }
                                return
                            }
                            var x = event.pageX - (scrollX + rect.left);
                            var y = event.pageY - (scrollY + rect.top);
                            x = x * (cw / rect.width);
                            y = y * (ch / rect.height);
                            Browser.mouseMovementX = x - Browser.mouseX;
                            Browser.mouseMovementY = y - Browser.mouseY;
                            Browser.mouseX = x;
                            Browser.mouseY = y
                        }
                    },
                    asyncLoad: function (url, onload, onerror, noRunDep) {
                        var dep = !noRunDep ? getUniqueRunDependency("al " + url) : "";
                        readAsync(url, function (arrayBuffer) {
                            assert(arrayBuffer, 'Loading data file "' + url + '" failed (no arrayBuffer).');
                            onload(new Uint8Array(arrayBuffer));
                            if (dep) removeRunDependency(dep)
                        }, function (event) {
                            if (onerror) {
                                onerror()
                            } else {
                                throw'Loading data file "' + url + '" failed.'
                            }
                        });
                        if (dep) addRunDependency(dep)
                    },
                    resizeListeners: [],
                    updateResizeListeners: function () {
                        var canvas = Module["canvas"];
                        Browser.resizeListeners.forEach(function (listener) {
                            listener(canvas.width, canvas.height)
                        })
                    },
                    setCanvasSize: function (width, height, noUpdates) {
                        var canvas = Module["canvas"];
                        Browser.updateCanvasDimensions(canvas, width, height);
                        if (!noUpdates) Browser.updateResizeListeners()
                    },
                    windowedWidth: 0,
                    windowedHeight: 0,
                    setFullscreenCanvasSize: function () {
                        if (typeof SDL != "undefined") {
                            var flags = HEAPU32[SDL.screen >> 2];
                            flags = flags | 8388608;
                            HEAP32[SDL.screen >> 2] = flags
                        }
                        Browser.updateCanvasDimensions(Module["canvas"]);
                        Browser.updateResizeListeners()
                    },
                    setWindowedCanvasSize: function () {
                        if (typeof SDL != "undefined") {
                            var flags = HEAPU32[SDL.screen >> 2];
                            flags = flags & ~8388608;
                            HEAP32[SDL.screen >> 2] = flags
                        }
                        Browser.updateCanvasDimensions(Module["canvas"]);
                        Browser.updateResizeListeners()
                    },
                    updateCanvasDimensions: function (canvas, wNative, hNative) {
                        if (wNative && hNative) {
                            canvas.widthNative = wNative;
                            canvas.heightNative = hNative
                        } else {
                            wNative = canvas.widthNative;
                            hNative = canvas.heightNative
                        }
                        var w = wNative;
                        var h = hNative;
                        if (Module["forcedAspectRatio"] && Module["forcedAspectRatio"] > 0) {
                            if (w / h < Module["forcedAspectRatio"]) {
                                w = Math.round(h * Module["forcedAspectRatio"])
                            } else {
                                h = Math.round(w / Module["forcedAspectRatio"])
                            }
                        }
                        if ((document["fullscreenElement"] || document["mozFullScreenElement"] || document["msFullscreenElement"] || document["webkitFullscreenElement"] || document["webkitCurrentFullScreenElement"]) === canvas.parentNode && typeof screen != "undefined") {
                            var factor = Math.min(screen.width / w, screen.height / h);
                            w = Math.round(w * factor);
                            h = Math.round(h * factor)
                        }
                        if (Browser.resizeCanvas) {
                            if (canvas.width != w) canvas.width = w;
                            if (canvas.height != h) canvas.height = h;
                            if (typeof canvas.style != "undefined") {
                                canvas.style.removeProperty("width");
                                canvas.style.removeProperty("height")
                            }
                        } else {
                            if (canvas.width != wNative) canvas.width = wNative;
                            if (canvas.height != hNative) canvas.height = hNative;
                            if (typeof canvas.style != "undefined") {
                                if (w != wNative || h != hNative) {
                                    canvas.style.setProperty("width", w + "px", "important");
                                    canvas.style.setProperty("height", h + "px", "important")
                                } else {
                                    canvas.style.removeProperty("width");
                                    canvas.style.removeProperty("height")
                                }
                            }
                        }
                    },
                    wgetRequests: {},
                    nextWgetRequestHandle: 0,
                    getNextWgetRequestHandle: function () {
                        var handle = Browser.nextWgetRequestHandle;
                        Browser.nextWgetRequestHandle++;
                        return handle
                    }
                };

                function demangle(func) {
                    demangle.recursionGuard = (demangle.recursionGuard | 0) + 1;
                    if (demangle.recursionGuard > 1) return func;
                    var __cxa_demangle_func = Module["___cxa_demangle"] || Module["__cxa_demangle"];
                    assert(__cxa_demangle_func);
                    var stackTop = stackSave();
                    try {
                        var s = func;
                        if (s.startsWith("__Z")) s = s.substr(1);
                        var len = lengthBytesUTF8(s) + 1;
                        var buf = stackAlloc(len);
                        stringToUTF8(s, buf, len);
                        var status = stackAlloc(4);
                        var ret = __cxa_demangle_func(buf, 0, 0, status);
                        if (HEAP32[status >> 2] === 0 && ret) {
                            return UTF8ToString(ret)
                        }
                    } catch (e) {
                    } finally {
                        _free(ret);
                        stackRestore(stackTop);
                        if (demangle.recursionGuard < 2) --demangle.recursionGuard
                    }
                    return func
                }

                function demangleAll(text) {
                    var regex = /\b_Z[\w\d_]+/g;
                    return text.replace(regex, function (x) {
                        var y = demangle(x);
                        return x === y ? x : y + " [" + x + "]"
                    })
                }

                function jsStackTrace() {
                    var err = new Error;
                    if (!err.stack) {
                        try {
                            throw new Error
                        } catch (e) {
                            err = e
                        }
                        if (!err.stack) {
                            return "(no stack trace available)"
                        }
                    }
                    return err.stack.toString()
                }

                function stackTrace() {
                    var js = jsStackTrace();
                    if (Module["extraStackTrace"]) js += "\n" + Module["extraStackTrace"]();
                    return demangleAll(js)
                }

                function ___cxa_allocate_exception(size) {
                    return _malloc(size)
                }

                function _atexit(func, arg) {
                }

                function ___cxa_atexit(a0, a1) {
                    return _atexit(a0, a1)
                }

                var ___exception_infos = {};
                var ___exception_last = 0;

                function __ZSt18uncaught_exceptionv() {
                    return __ZSt18uncaught_exceptionv.uncaught_exceptions > 0
                }

                function ___cxa_throw(ptr, type, destructor) {
                    ___exception_infos[ptr] = {
                        ptr: ptr,
                        adjusted: [ptr],
                        type: type,
                        destructor: destructor,
                        refcount: 0,
                        caught: false,
                        rethrown: false
                    };
                    ___exception_last = ptr;
                    if (!("uncaught_exception" in __ZSt18uncaught_exceptionv)) {
                        __ZSt18uncaught_exceptionv.uncaught_exceptions = 1
                    } else {
                        __ZSt18uncaught_exceptionv.uncaught_exceptions++
                    }
                    throw ptr
                }

                function setErrNo(value) {
                    HEAP32[___errno_location() >> 2] = value;
                    return value
                }

                function ___map_file(pathname, size) {
                    setErrNo(63);
                    return -1
                }

                var PATH = {
                    splitPath: function (filename) {
                        var splitPathRe = /^(\/?|)([\s\S]*?)((?:\.{1,2}|[^\/]+?|)(\.[^.\/]*|))(?:[\/]*)$/;
                        return splitPathRe.exec(filename).slice(1)
                    }, normalizeArray: function (parts, allowAboveRoot) {
                        var up = 0;
                        for (var i = parts.length - 1; i >= 0; i--) {
                            var last = parts[i];
                            if (last === ".") {
                                parts.splice(i, 1)
                            } else if (last === "..") {
                                parts.splice(i, 1);
                                up++
                            } else if (up) {
                                parts.splice(i, 1);
                                up--
                            }
                        }
                        if (allowAboveRoot) {
                            for (; up; up--) {
                                parts.unshift("..")
                            }
                        }
                        return parts
                    }, normalize: function (path) {
                        var isAbsolute = path.charAt(0) === "/", trailingSlash = path.substr(-1) === "/";
                        path = PATH.normalizeArray(path.split("/").filter(function (p) {
                            return !!p
                        }), !isAbsolute).join("/");
                        if (!path && !isAbsolute) {
                            path = "."
                        }
                        if (path && trailingSlash) {
                            path += "/"
                        }
                        return (isAbsolute ? "/" : "") + path
                    }, dirname: function (path) {
                        var result = PATH.splitPath(path), root = result[0], dir = result[1];
                        if (!root && !dir) {
                            return "."
                        }
                        if (dir) {
                            dir = dir.substr(0, dir.length - 1)
                        }
                        return root + dir
                    }, basename: function (path) {
                        if (path === "/") return "/";
                        var lastSlash = path.lastIndexOf("/");
                        if (lastSlash === -1) return path;
                        return path.substr(lastSlash + 1)
                    }, extname: function (path) {
                        return PATH.splitPath(path)[3]
                    }, join: function () {
                        var paths = Array.prototype.slice.call(arguments, 0);
                        return PATH.normalize(paths.join("/"))
                    }, join2: function (l, r) {
                        return PATH.normalize(l + "/" + r)
                    }
                };
                var PATH_FS = {
                    resolve: function () {
                        var resolvedPath = "", resolvedAbsolute = false;
                        for (var i = arguments.length - 1; i >= -1 && !resolvedAbsolute; i--) {
                            var path = i >= 0 ? arguments[i] : FS.cwd();
                            if (typeof path !== "string") {
                                throw new TypeError("Arguments to path.resolve must be strings")
                            } else if (!path) {
                                return ""
                            }
                            resolvedPath = path + "/" + resolvedPath;
                            resolvedAbsolute = path.charAt(0) === "/"
                        }
                        resolvedPath = PATH.normalizeArray(resolvedPath.split("/").filter(function (p) {
                            return !!p
                        }), !resolvedAbsolute).join("/");
                        return (resolvedAbsolute ? "/" : "") + resolvedPath || "."
                    }, relative: function (from, to) {
                        from = PATH_FS.resolve(from).substr(1);
                        to = PATH_FS.resolve(to).substr(1);

                        function trim(arr) {
                            var start = 0;
                            for (; start < arr.length; start++) {
                                if (arr[start] !== "") break
                            }
                            var end = arr.length - 1;
                            for (; end >= 0; end--) {
                                if (arr[end] !== "") break
                            }
                            if (start > end) return [];
                            return arr.slice(start, end - start + 1)
                        }

                        var fromParts = trim(from.split("/"));
                        var toParts = trim(to.split("/"));
                        var length = Math.min(fromParts.length, toParts.length);
                        var samePartsLength = length;
                        for (var i = 0; i < length; i++) {
                            if (fromParts[i] !== toParts[i]) {
                                samePartsLength = i;
                                break
                            }
                        }
                        var outputParts = [];
                        for (var i = samePartsLength; i < fromParts.length; i++) {
                            outputParts.push("..")
                        }
                        outputParts = outputParts.concat(toParts.slice(samePartsLength));
                        return outputParts.join("/")
                    }
                };
                var TTY = {
                    ttys: [], init: function () {
                    }, shutdown: function () {
                    }, register: function (dev, ops) {
                        TTY.ttys[dev] = {input: [], output: [], ops: ops};
                        FS.registerDevice(dev, TTY.stream_ops)
                    }, stream_ops: {
                        open: function (stream) {
                            var tty = TTY.ttys[stream.node.rdev];
                            if (!tty) {
                                throw new FS.ErrnoError(43)
                            }
                            stream.tty = tty;
                            stream.seekable = false
                        }, close: function (stream) {
                            stream.tty.ops.flush(stream.tty)
                        }, flush: function (stream) {
                            stream.tty.ops.flush(stream.tty)
                        }, read: function (stream, buffer, offset, length, pos) {
                            if (!stream.tty || !stream.tty.ops.get_char) {
                                throw new FS.ErrnoError(60)
                            }
                            var bytesRead = 0;
                            for (var i = 0; i < length; i++) {
                                var result;
                                try {
                                    result = stream.tty.ops.get_char(stream.tty)
                                } catch (e) {
                                    throw new FS.ErrnoError(29)
                                }
                                if (result === undefined && bytesRead === 0) {
                                    throw new FS.ErrnoError(6)
                                }
                                if (result === null || result === undefined) break;
                                bytesRead++;
                                buffer[offset + i] = result
                            }
                            if (bytesRead) {
                                stream.node.timestamp = Date.now()
                            }
                            return bytesRead
                        }, write: function (stream, buffer, offset, length, pos) {
                            if (!stream.tty || !stream.tty.ops.put_char) {
                                throw new FS.ErrnoError(60)
                            }
                            try {
                                for (var i = 0; i < length; i++) {
                                    stream.tty.ops.put_char(stream.tty, buffer[offset + i])
                                }
                            } catch (e) {
                                throw new FS.ErrnoError(29)
                            }
                            if (length) {
                                stream.node.timestamp = Date.now()
                            }
                            return i
                        }
                    }, default_tty_ops: {
                        get_char: function (tty) {
                            if (!tty.input.length) {
                                var result = null;
                                if (ENVIRONMENT_IS_NODE) {
                                    var BUFSIZE = 256;
                                    var buf = Buffer.alloc ? Buffer.alloc(BUFSIZE) : new Buffer(BUFSIZE);
                                    var bytesRead = 0;
                                    try {
                                        bytesRead = nodeFS.readSync(process.stdin.fd, buf, 0, BUFSIZE, null)
                                    } catch (e) {
                                        if (e.toString().indexOf("EOF") != -1) bytesRead = 0; else throw e
                                    }
                                    if (bytesRead > 0) {
                                        result = buf.slice(0, bytesRead).toString("utf-8")
                                    } else {
                                        result = null
                                    }
                                } else if (typeof window != "undefined" && typeof window.prompt == "function") {
                                    result = window.prompt("Input: ");
                                    if (result !== null) {
                                        result += "\n"
                                    }
                                } else if (typeof readline == "function") {
                                    result = readline();
                                    if (result !== null) {
                                        result += "\n"
                                    }
                                }
                                if (!result) {
                                    return null
                                }
                                tty.input = intArrayFromString(result, true)
                            }
                            return tty.input.shift()
                        }, put_char: function (tty, val) {
                            if (val === null || val === 10) {
                                out(UTF8ArrayToString(tty.output, 0));
                                tty.output = []
                            } else {
                                if (val != 0) tty.output.push(val)
                            }
                        }, flush: function (tty) {
                            if (tty.output && tty.output.length > 0) {
                                out(UTF8ArrayToString(tty.output, 0));
                                tty.output = []
                            }
                        }
                    }, default_tty1_ops: {
                        put_char: function (tty, val) {
                            if (val === null || val === 10) {
                                err(UTF8ArrayToString(tty.output, 0));
                                tty.output = []
                            } else {
                                if (val != 0) tty.output.push(val)
                            }
                        }, flush: function (tty) {
                            if (tty.output && tty.output.length > 0) {
                                err(UTF8ArrayToString(tty.output, 0));
                                tty.output = []
                            }
                        }
                    }
                };
                var MEMFS = {
                    ops_table: null, mount: function (mount) {
                        return MEMFS.createNode(null, "/", 16384 | 511, 0)
                    }, createNode: function (parent, name, mode, dev) {
                        if (FS.isBlkdev(mode) || FS.isFIFO(mode)) {
                            throw new FS.ErrnoError(63)
                        }
                        if (!MEMFS.ops_table) {
                            MEMFS.ops_table = {
                                dir: {
                                    node: {
                                        getattr: MEMFS.node_ops.getattr,
                                        setattr: MEMFS.node_ops.setattr,
                                        lookup: MEMFS.node_ops.lookup,
                                        mknod: MEMFS.node_ops.mknod,
                                        rename: MEMFS.node_ops.rename,
                                        unlink: MEMFS.node_ops.unlink,
                                        rmdir: MEMFS.node_ops.rmdir,
                                        readdir: MEMFS.node_ops.readdir,
                                        symlink: MEMFS.node_ops.symlink
                                    }, stream: {llseek: MEMFS.stream_ops.llseek}
                                },
                                file: {
                                    node: {getattr: MEMFS.node_ops.getattr, setattr: MEMFS.node_ops.setattr},
                                    stream: {
                                        llseek: MEMFS.stream_ops.llseek,
                                        read: MEMFS.stream_ops.read,
                                        write: MEMFS.stream_ops.write,
                                        allocate: MEMFS.stream_ops.allocate,
                                        mmap: MEMFS.stream_ops.mmap,
                                        msync: MEMFS.stream_ops.msync
                                    }
                                },
                                link: {
                                    node: {
                                        getattr: MEMFS.node_ops.getattr,
                                        setattr: MEMFS.node_ops.setattr,
                                        readlink: MEMFS.node_ops.readlink
                                    }, stream: {}
                                },
                                chrdev: {
                                    node: {getattr: MEMFS.node_ops.getattr, setattr: MEMFS.node_ops.setattr},
                                    stream: FS.chrdev_stream_ops
                                }
                            }
                        }
                        var node = FS.createNode(parent, name, mode, dev);
                        if (FS.isDir(node.mode)) {
                            node.node_ops = MEMFS.ops_table.dir.node;
                            node.stream_ops = MEMFS.ops_table.dir.stream;
                            node.contents = {}
                        } else if (FS.isFile(node.mode)) {
                            node.node_ops = MEMFS.ops_table.file.node;
                            node.stream_ops = MEMFS.ops_table.file.stream;
                            node.usedBytes = 0;
                            node.contents = null
                        } else if (FS.isLink(node.mode)) {
                            node.node_ops = MEMFS.ops_table.link.node;
                            node.stream_ops = MEMFS.ops_table.link.stream
                        } else if (FS.isChrdev(node.mode)) {
                            node.node_ops = MEMFS.ops_table.chrdev.node;
                            node.stream_ops = MEMFS.ops_table.chrdev.stream
                        }
                        node.timestamp = Date.now();
                        if (parent) {
                            parent.contents[name] = node
                        }
                        return node
                    }, getFileDataAsRegularArray: function (node) {
                        if (node.contents && node.contents.subarray) {
                            var arr = [];
                            for (var i = 0; i < node.usedBytes; ++i) arr.push(node.contents[i]);
                            return arr
                        }
                        return node.contents
                    }, getFileDataAsTypedArray: function (node) {
                        if (!node.contents) return new Uint8Array(0);
                        if (node.contents.subarray) return node.contents.subarray(0, node.usedBytes);
                        return new Uint8Array(node.contents)
                    }, expandFileStorage: function (node, newCapacity) {
                        var prevCapacity = node.contents ? node.contents.length : 0;
                        if (prevCapacity >= newCapacity) return;
                        var CAPACITY_DOUBLING_MAX = 1024 * 1024;
                        newCapacity = Math.max(newCapacity, prevCapacity * (prevCapacity < CAPACITY_DOUBLING_MAX ? 2 : 1.125) >>> 0);
                        if (prevCapacity != 0) newCapacity = Math.max(newCapacity, 256);
                        var oldContents = node.contents;
                        node.contents = new Uint8Array(newCapacity);
                        if (node.usedBytes > 0) node.contents.set(oldContents.subarray(0, node.usedBytes), 0);
                        return
                    }, resizeFileStorage: function (node, newSize) {
                        if (node.usedBytes == newSize) return;
                        if (newSize == 0) {
                            node.contents = null;
                            node.usedBytes = 0;
                            return
                        }
                        if (!node.contents || node.contents.subarray) {
                            var oldContents = node.contents;
                            node.contents = new Uint8Array(newSize);
                            if (oldContents) {
                                node.contents.set(oldContents.subarray(0, Math.min(newSize, node.usedBytes)))
                            }
                            node.usedBytes = newSize;
                            return
                        }
                        if (!node.contents) node.contents = [];
                        if (node.contents.length > newSize) node.contents.length = newSize; else while (node.contents.length < newSize) node.contents.push(0);
                        node.usedBytes = newSize
                    }, node_ops: {
                        getattr: function (node) {
                            var attr = {};
                            attr.dev = FS.isChrdev(node.mode) ? node.id : 1;
                            attr.ino = node.id;
                            attr.mode = node.mode;
                            attr.nlink = 1;
                            attr.uid = 0;
                            attr.gid = 0;
                            attr.rdev = node.rdev;
                            if (FS.isDir(node.mode)) {
                                attr.size = 4096
                            } else if (FS.isFile(node.mode)) {
                                attr.size = node.usedBytes
                            } else if (FS.isLink(node.mode)) {
                                attr.size = node.link.length
                            } else {
                                attr.size = 0
                            }
                            attr.atime = new Date(node.timestamp);
                            attr.mtime = new Date(node.timestamp);
                            attr.ctime = new Date(node.timestamp);
                            attr.blksize = 4096;
                            attr.blocks = Math.ceil(attr.size / attr.blksize);
                            return attr
                        }, setattr: function (node, attr) {
                            if (attr.mode !== undefined) {
                                node.mode = attr.mode
                            }
                            if (attr.timestamp !== undefined) {
                                node.timestamp = attr.timestamp
                            }
                            if (attr.size !== undefined) {
                                MEMFS.resizeFileStorage(node, attr.size)
                            }
                        }, lookup: function (parent, name) {
                            throw FS.genericErrors[44]
                        }, mknod: function (parent, name, mode, dev) {
                            return MEMFS.createNode(parent, name, mode, dev)
                        }, rename: function (old_node, new_dir, new_name) {
                            if (FS.isDir(old_node.mode)) {
                                var new_node;
                                try {
                                    new_node = FS.lookupNode(new_dir, new_name)
                                } catch (e) {
                                }
                                if (new_node) {
                                    for (var i in new_node.contents) {
                                        throw new FS.ErrnoError(55)
                                    }
                                }
                            }
                            delete old_node.parent.contents[old_node.name];
                            old_node.name = new_name;
                            new_dir.contents[new_name] = old_node;
                            old_node.parent = new_dir
                        }, unlink: function (parent, name) {
                            delete parent.contents[name]
                        }, rmdir: function (parent, name) {
                            var node = FS.lookupNode(parent, name);
                            for (var i in node.contents) {
                                throw new FS.ErrnoError(55)
                            }
                            delete parent.contents[name]
                        }, readdir: function (node) {
                            var entries = [".", ".."];
                            for (var key in node.contents) {
                                if (!node.contents.hasOwnProperty(key)) {
                                    continue
                                }
                                entries.push(key)
                            }
                            return entries
                        }, symlink: function (parent, newname, oldpath) {
                            var node = MEMFS.createNode(parent, newname, 511 | 40960, 0);
                            node.link = oldpath;
                            return node
                        }, readlink: function (node) {
                            if (!FS.isLink(node.mode)) {
                                throw new FS.ErrnoError(28)
                            }
                            return node.link
                        }
                    }, stream_ops: {
                        read: function (stream, buffer, offset, length, position) {
                            var contents = stream.node.contents;
                            if (position >= stream.node.usedBytes) return 0;
                            var size = Math.min(stream.node.usedBytes - position, length);
                            if (size > 8 && contents.subarray) {
                                buffer.set(contents.subarray(position, position + size), offset)
                            } else {
                                for (var i = 0; i < size; i++) buffer[offset + i] = contents[position + i]
                            }
                            return size
                        }, write: function (stream, buffer, offset, length, position, canOwn) {
                            if (buffer.buffer === HEAP8.buffer) {
                                canOwn = false
                            }
                            if (!length) return 0;
                            var node = stream.node;
                            node.timestamp = Date.now();
                            if (buffer.subarray && (!node.contents || node.contents.subarray)) {
                                if (canOwn) {
                                    node.contents = buffer.subarray(offset, offset + length);
                                    node.usedBytes = length;
                                    return length
                                } else if (node.usedBytes === 0 && position === 0) {
                                    node.contents = buffer.slice(offset, offset + length);
                                    node.usedBytes = length;
                                    return length
                                } else if (position + length <= node.usedBytes) {
                                    node.contents.set(buffer.subarray(offset, offset + length), position);
                                    return length
                                }
                            }
                            MEMFS.expandFileStorage(node, position + length);
                            if (node.contents.subarray && buffer.subarray) {
                                node.contents.set(buffer.subarray(offset, offset + length), position)
                            } else {
                                for (var i = 0; i < length; i++) {
                                    node.contents[position + i] = buffer[offset + i]
                                }
                            }
                            node.usedBytes = Math.max(node.usedBytes, position + length);
                            return length
                        }, llseek: function (stream, offset, whence) {
                            var position = offset;
                            if (whence === 1) {
                                position += stream.position
                            } else if (whence === 2) {
                                if (FS.isFile(stream.node.mode)) {
                                    position += stream.node.usedBytes
                                }
                            }
                            if (position < 0) {
                                throw new FS.ErrnoError(28)
                            }
                            return position
                        }, allocate: function (stream, offset, length) {
                            MEMFS.expandFileStorage(stream.node, offset + length);
                            stream.node.usedBytes = Math.max(stream.node.usedBytes, offset + length)
                        }, mmap: function (stream, address, length, position, prot, flags) {
                            assert(address === 0);
                            if (!FS.isFile(stream.node.mode)) {
                                throw new FS.ErrnoError(43)
                            }
                            var ptr;
                            var allocated;
                            var contents = stream.node.contents;
                            if (!(flags & 2) && contents.buffer === buffer) {
                                allocated = false;
                                ptr = contents.byteOffset
                            } else {
                                if (position > 0 || position + length < contents.length) {
                                    if (contents.subarray) {
                                        contents = contents.subarray(position, position + length)
                                    } else {
                                        contents = Array.prototype.slice.call(contents, position, position + length)
                                    }
                                }
                                allocated = true;
                                ptr = FS.mmapAlloc(length);
                                if (!ptr) {
                                    throw new FS.ErrnoError(48)
                                }
                                HEAP8.set(contents, ptr)
                            }
                            return {ptr: ptr, allocated: allocated}
                        }, msync: function (stream, buffer, offset, length, mmapFlags) {
                            if (!FS.isFile(stream.node.mode)) {
                                throw new FS.ErrnoError(43)
                            }
                            if (mmapFlags & 2) {
                                return 0
                            }
                            var bytesWritten = MEMFS.stream_ops.write(stream, buffer, 0, length, offset, false);
                            return 0
                        }
                    }
                };
                var FS = {
                    root: null,
                    mounts: [],
                    devices: {},
                    streams: [],
                    nextInode: 1,
                    nameTable: null,
                    currentPath: "/",
                    initialized: false,
                    ignorePermissions: true,
                    trackingDelegate: {},
                    tracking: {openFlags: {READ: 1, WRITE: 2}},
                    ErrnoError: null,
                    genericErrors: {},
                    filesystems: null,
                    syncFSRequests: 0,
                    handleFSError: function (e) {
                        if (!(e instanceof FS.ErrnoError)) throw e + " : " + stackTrace();
                        return setErrNo(e.errno)
                    },
                    lookupPath: function (path, opts) {
                        path = PATH_FS.resolve(FS.cwd(), path);
                        opts = opts || {};
                        if (!path) return {path: "", node: null};
                        var defaults = {follow_mount: true, recurse_count: 0};
                        for (var key in defaults) {
                            if (opts[key] === undefined) {
                                opts[key] = defaults[key]
                            }
                        }
                        if (opts.recurse_count > 8) {
                            throw new FS.ErrnoError(32)
                        }
                        var parts = PATH.normalizeArray(path.split("/").filter(function (p) {
                            return !!p
                        }), false);
                        var current = FS.root;
                        var current_path = "/";
                        for (var i = 0; i < parts.length; i++) {
                            var islast = i === parts.length - 1;
                            if (islast && opts.parent) {
                                break
                            }
                            current = FS.lookupNode(current, parts[i]);
                            current_path = PATH.join2(current_path, parts[i]);
                            if (FS.isMountpoint(current)) {
                                if (!islast || islast && opts.follow_mount) {
                                    current = current.mounted.root
                                }
                            }
                            if (!islast || opts.follow) {
                                var count = 0;
                                while (FS.isLink(current.mode)) {
                                    var link = FS.readlink(current_path);
                                    current_path = PATH_FS.resolve(PATH.dirname(current_path), link);
                                    var lookup = FS.lookupPath(current_path, {recurse_count: opts.recurse_count});
                                    current = lookup.node;
                                    if (count++ > 40) {
                                        throw new FS.ErrnoError(32)
                                    }
                                }
                            }
                        }
                        return {path: current_path, node: current}
                    },
                    getPath: function (node) {
                        var path;
                        while (true) {
                            if (FS.isRoot(node)) {
                                var mount = node.mount.mountpoint;
                                if (!path) return mount;
                                return mount[mount.length - 1] !== "/" ? mount + "/" + path : mount + path
                            }
                            path = path ? node.name + "/" + path : node.name;
                            node = node.parent
                        }
                    },
                    hashName: function (parentid, name) {
                        var hash = 0;
                        for (var i = 0; i < name.length; i++) {
                            hash = (hash << 5) - hash + name.charCodeAt(i) | 0
                        }
                        return (parentid + hash >>> 0) % FS.nameTable.length
                    },
                    hashAddNode: function (node) {
                        var hash = FS.hashName(node.parent.id, node.name);
                        node.name_next = FS.nameTable[hash];
                        FS.nameTable[hash] = node
                    },
                    hashRemoveNode: function (node) {
                        var hash = FS.hashName(node.parent.id, node.name);
                        if (FS.nameTable[hash] === node) {
                            FS.nameTable[hash] = node.name_next
                        } else {
                            var current = FS.nameTable[hash];
                            while (current) {
                                if (current.name_next === node) {
                                    current.name_next = node.name_next;
                                    break
                                }
                                current = current.name_next
                            }
                        }
                    },
                    lookupNode: function (parent, name) {
                        var errCode = FS.mayLookup(parent);
                        if (errCode) {
                            throw new FS.ErrnoError(errCode, parent)
                        }
                        var hash = FS.hashName(parent.id, name);
                        for (var node = FS.nameTable[hash]; node; node = node.name_next) {
                            var nodeName = node.name;
                            if (node.parent.id === parent.id && nodeName === name) {
                                return node
                            }
                        }
                        return FS.lookup(parent, name)
                    },
                    createNode: function (parent, name, mode, rdev) {
                        var node = new FS.FSNode(parent, name, mode, rdev);
                        FS.hashAddNode(node);
                        return node
                    },
                    destroyNode: function (node) {
                        FS.hashRemoveNode(node)
                    },
                    isRoot: function (node) {
                        return node === node.parent
                    },
                    isMountpoint: function (node) {
                        return !!node.mounted
                    },
                    isFile: function (mode) {
                        return (mode & 61440) === 32768
                    },
                    isDir: function (mode) {
                        return (mode & 61440) === 16384
                    },
                    isLink: function (mode) {
                        return (mode & 61440) === 40960
                    },
                    isChrdev: function (mode) {
                        return (mode & 61440) === 8192
                    },
                    isBlkdev: function (mode) {
                        return (mode & 61440) === 24576
                    },
                    isFIFO: function (mode) {
                        return (mode & 61440) === 4096
                    },
                    isSocket: function (mode) {
                        return (mode & 49152) === 49152
                    },
                    flagModes: {
                        "r": 0,
                        "rs": 1052672,
                        "r+": 2,
                        "w": 577,
                        "wx": 705,
                        "xw": 705,
                        "w+": 578,
                        "wx+": 706,
                        "xw+": 706,
                        "a": 1089,
                        "ax": 1217,
                        "xa": 1217,
                        "a+": 1090,
                        "ax+": 1218,
                        "xa+": 1218
                    },
                    modeStringToFlags: function (str) {
                        var flags = FS.flagModes[str];
                        if (typeof flags === "undefined") {
                            throw new Error("Unknown file open mode: " + str)
                        }
                        return flags
                    },
                    flagsToPermissionString: function (flag) {
                        var perms = ["r", "w", "rw"][flag & 3];
                        if (flag & 512) {
                            perms += "w"
                        }
                        return perms
                    },
                    nodePermissions: function (node, perms) {
                        if (FS.ignorePermissions) {
                            return 0
                        }
                        if (perms.indexOf("r") !== -1 && !(node.mode & 292)) {
                            return 2
                        } else if (perms.indexOf("w") !== -1 && !(node.mode & 146)) {
                            return 2
                        } else if (perms.indexOf("x") !== -1 && !(node.mode & 73)) {
                            return 2
                        }
                        return 0
                    },
                    mayLookup: function (dir) {
                        var errCode = FS.nodePermissions(dir, "x");
                        if (errCode) return errCode;
                        if (!dir.node_ops.lookup) return 2;
                        return 0
                    },
                    mayCreate: function (dir, name) {
                        try {
                            var node = FS.lookupNode(dir, name);
                            return 20
                        } catch (e) {
                        }
                        return FS.nodePermissions(dir, "wx")
                    },
                    mayDelete: function (dir, name, isdir) {
                        var node;
                        try {
                            node = FS.lookupNode(dir, name)
                        } catch (e) {
                            return e.errno
                        }
                        var errCode = FS.nodePermissions(dir, "wx");
                        if (errCode) {
                            return errCode
                        }
                        if (isdir) {
                            if (!FS.isDir(node.mode)) {
                                return 54
                            }
                            if (FS.isRoot(node) || FS.getPath(node) === FS.cwd()) {
                                return 10
                            }
                        } else {
                            if (FS.isDir(node.mode)) {
                                return 31
                            }
                        }
                        return 0
                    },
                    mayOpen: function (node, flags) {
                        if (!node) {
                            return 44
                        }
                        if (FS.isLink(node.mode)) {
                            return 32
                        } else if (FS.isDir(node.mode)) {
                            if (FS.flagsToPermissionString(flags) !== "r" || flags & 512) {
                                return 31
                            }
                        }
                        return FS.nodePermissions(node, FS.flagsToPermissionString(flags))
                    },
                    MAX_OPEN_FDS: 4096,
                    nextfd: function (fd_start, fd_end) {
                        fd_start = fd_start || 0;
                        fd_end = fd_end || FS.MAX_OPEN_FDS;
                        for (var fd = fd_start; fd <= fd_end; fd++) {
                            if (!FS.streams[fd]) {
                                return fd
                            }
                        }
                        throw new FS.ErrnoError(33)
                    },
                    getStream: function (fd) {
                        return FS.streams[fd]
                    },
                    createStream: function (stream, fd_start, fd_end) {
                        if (!FS.FSStream) {
                            FS.FSStream = function () {
                            };
                            FS.FSStream.prototype = {
                                object: {
                                    get: function () {
                                        return this.node
                                    }, set: function (val) {
                                        this.node = val
                                    }
                                }, isRead: {
                                    get: function () {
                                        return (this.flags & 2097155) !== 1
                                    }
                                }, isWrite: {
                                    get: function () {
                                        return (this.flags & 2097155) !== 0
                                    }
                                }, isAppend: {
                                    get: function () {
                                        return this.flags & 1024
                                    }
                                }
                            }
                        }
                        var newStream = new FS.FSStream;
                        for (var p in stream) {
                            newStream[p] = stream[p]
                        }
                        stream = newStream;
                        var fd = FS.nextfd(fd_start, fd_end);
                        stream.fd = fd;
                        FS.streams[fd] = stream;
                        return stream
                    },
                    closeStream: function (fd) {
                        FS.streams[fd] = null
                    },
                    chrdev_stream_ops: {
                        open: function (stream) {
                            var device = FS.getDevice(stream.node.rdev);
                            stream.stream_ops = device.stream_ops;
                            if (stream.stream_ops.open) {
                                stream.stream_ops.open(stream)
                            }
                        }, llseek: function () {
                            throw new FS.ErrnoError(70)
                        }
                    },
                    major: function (dev) {
                        return dev >> 8
                    },
                    minor: function (dev) {
                        return dev & 255
                    },
                    makedev: function (ma, mi) {
                        return ma << 8 | mi
                    },
                    registerDevice: function (dev, ops) {
                        FS.devices[dev] = {stream_ops: ops}
                    },
                    getDevice: function (dev) {
                        return FS.devices[dev]
                    },
                    getMounts: function (mount) {
                        var mounts = [];
                        var check = [mount];
                        while (check.length) {
                            var m = check.pop();
                            mounts.push(m);
                            check.push.apply(check, m.mounts)
                        }
                        return mounts
                    },
                    syncfs: function (populate, callback) {
                        if (typeof populate === "function") {
                            callback = populate;
                            populate = false
                        }
                        FS.syncFSRequests++;
                        if (FS.syncFSRequests > 1) {
                            err("warning: " + FS.syncFSRequests + " FS.syncfs operations in flight at once, probably just doing extra work")
                        }
                        var mounts = FS.getMounts(FS.root.mount);
                        var completed = 0;

                        function doCallback(errCode) {
                            FS.syncFSRequests--;
                            return callback(errCode)
                        }

                        function done(errCode) {
                            if (errCode) {
                                if (!done.errored) {
                                    done.errored = true;
                                    return doCallback(errCode)
                                }
                                return
                            }
                            if (++completed >= mounts.length) {
                                doCallback(null)
                            }
                        }

                        mounts.forEach(function (mount) {
                            if (!mount.type.syncfs) {
                                return done(null)
                            }
                            mount.type.syncfs(mount, populate, done)
                        })
                    },
                    mount: function (type, opts, mountpoint) {
                        var root = mountpoint === "/";
                        var pseudo = !mountpoint;
                        var node;
                        if (root && FS.root) {
                            throw new FS.ErrnoError(10)
                        } else if (!root && !pseudo) {
                            var lookup = FS.lookupPath(mountpoint, {follow_mount: false});
                            mountpoint = lookup.path;
                            node = lookup.node;
                            if (FS.isMountpoint(node)) {
                                throw new FS.ErrnoError(10)
                            }
                            if (!FS.isDir(node.mode)) {
                                throw new FS.ErrnoError(54)
                            }
                        }
                        var mount = {type: type, opts: opts, mountpoint: mountpoint, mounts: []};
                        var mountRoot = type.mount(mount);
                        mountRoot.mount = mount;
                        mount.root = mountRoot;
                        if (root) {
                            FS.root = mountRoot
                        } else if (node) {
                            node.mounted = mount;
                            if (node.mount) {
                                node.mount.mounts.push(mount)
                            }
                        }
                        return mountRoot
                    },
                    unmount: function (mountpoint) {
                        var lookup = FS.lookupPath(mountpoint, {follow_mount: false});
                        if (!FS.isMountpoint(lookup.node)) {
                            throw new FS.ErrnoError(28)
                        }
                        var node = lookup.node;
                        var mount = node.mounted;
                        var mounts = FS.getMounts(mount);
                        Object.keys(FS.nameTable).forEach(function (hash) {
                            var current = FS.nameTable[hash];
                            while (current) {
                                var next = current.name_next;
                                if (mounts.indexOf(current.mount) !== -1) {
                                    FS.destroyNode(current)
                                }
                                current = next
                            }
                        });
                        node.mounted = null;
                        var idx = node.mount.mounts.indexOf(mount);
                        node.mount.mounts.splice(idx, 1)
                    },
                    lookup: function (parent, name) {
                        return parent.node_ops.lookup(parent, name)
                    },
                    mknod: function (path, mode, dev) {
                        var lookup = FS.lookupPath(path, {parent: true});
                        var parent = lookup.node;
                        var name = PATH.basename(path);
                        if (!name || name === "." || name === "..") {
                            throw new FS.ErrnoError(28)
                        }
                        var errCode = FS.mayCreate(parent, name);
                        if (errCode) {
                            throw new FS.ErrnoError(errCode)
                        }
                        if (!parent.node_ops.mknod) {
                            throw new FS.ErrnoError(63)
                        }
                        return parent.node_ops.mknod(parent, name, mode, dev)
                    },
                    create: function (path, mode) {
                        mode = mode !== undefined ? mode : 438;
                        mode &= 4095;
                        mode |= 32768;
                        return FS.mknod(path, mode, 0)
                    },
                    mkdir: function (path, mode) {
                        mode = mode !== undefined ? mode : 511;
                        mode &= 511 | 512;
                        mode |= 16384;
                        return FS.mknod(path, mode, 0)
                    },
                    mkdirTree: function (path, mode) {
                        var dirs = path.split("/");
                        var d = "";
                        for (var i = 0; i < dirs.length; ++i) {
                            if (!dirs[i]) continue;
                            d += "/" + dirs[i];
                            try {
                                FS.mkdir(d, mode)
                            } catch (e) {
                                if (e.errno != 20) throw e
                            }
                        }
                    },
                    mkdev: function (path, mode, dev) {
                        if (typeof dev === "undefined") {
                            dev = mode;
                            mode = 438
                        }
                        mode |= 8192;
                        return FS.mknod(path, mode, dev)
                    },
                    symlink: function (oldpath, newpath) {
                        if (!PATH_FS.resolve(oldpath)) {
                            throw new FS.ErrnoError(44)
                        }
                        var lookup = FS.lookupPath(newpath, {parent: true});
                        var parent = lookup.node;
                        if (!parent) {
                            throw new FS.ErrnoError(44)
                        }
                        var newname = PATH.basename(newpath);
                        var errCode = FS.mayCreate(parent, newname);
                        if (errCode) {
                            throw new FS.ErrnoError(errCode)
                        }
                        if (!parent.node_ops.symlink) {
                            throw new FS.ErrnoError(63)
                        }
                        return parent.node_ops.symlink(parent, newname, oldpath)
                    },
                    rename: function (old_path, new_path) {
                        var old_dirname = PATH.dirname(old_path);
                        var new_dirname = PATH.dirname(new_path);
                        var old_name = PATH.basename(old_path);
                        var new_name = PATH.basename(new_path);
                        var lookup, old_dir, new_dir;
                        try {
                            lookup = FS.lookupPath(old_path, {parent: true});
                            old_dir = lookup.node;
                            lookup = FS.lookupPath(new_path, {parent: true});
                            new_dir = lookup.node
                        } catch (e) {
                            throw new FS.ErrnoError(10)
                        }
                        if (!old_dir || !new_dir) throw new FS.ErrnoError(44);
                        if (old_dir.mount !== new_dir.mount) {
                            throw new FS.ErrnoError(75)
                        }
                        var old_node = FS.lookupNode(old_dir, old_name);
                        var relative = PATH_FS.relative(old_path, new_dirname);
                        if (relative.charAt(0) !== ".") {
                            throw new FS.ErrnoError(28)
                        }
                        relative = PATH_FS.relative(new_path, old_dirname);
                        if (relative.charAt(0) !== ".") {
                            throw new FS.ErrnoError(55)
                        }
                        var new_node;
                        try {
                            new_node = FS.lookupNode(new_dir, new_name)
                        } catch (e) {
                        }
                        if (old_node === new_node) {
                            return
                        }
                        var isdir = FS.isDir(old_node.mode);
                        var errCode = FS.mayDelete(old_dir, old_name, isdir);
                        if (errCode) {
                            throw new FS.ErrnoError(errCode)
                        }
                        errCode = new_node ? FS.mayDelete(new_dir, new_name, isdir) : FS.mayCreate(new_dir, new_name);
                        if (errCode) {
                            throw new FS.ErrnoError(errCode)
                        }
                        if (!old_dir.node_ops.rename) {
                            throw new FS.ErrnoError(63)
                        }
                        if (FS.isMountpoint(old_node) || new_node && FS.isMountpoint(new_node)) {
                            throw new FS.ErrnoError(10)
                        }
                        if (new_dir !== old_dir) {
                            errCode = FS.nodePermissions(old_dir, "w");
                            if (errCode) {
                                throw new FS.ErrnoError(errCode)
                            }
                        }
                        try {
                            if (FS.trackingDelegate["willMovePath"]) {
                                FS.trackingDelegate["willMovePath"](old_path, new_path)
                            }
                        } catch (e) {
                            err("FS.trackingDelegate['willMovePath']('" + old_path + "', '" + new_path + "') threw an exception: " + e.message)
                        }
                        FS.hashRemoveNode(old_node);
                        try {
                            old_dir.node_ops.rename(old_node, new_dir, new_name)
                        } catch (e) {
                            throw e
                        } finally {
                            FS.hashAddNode(old_node)
                        }
                        try {
                            if (FS.trackingDelegate["onMovePath"]) FS.trackingDelegate["onMovePath"](old_path, new_path)
                        } catch (e) {
                            err("FS.trackingDelegate['onMovePath']('" + old_path + "', '" + new_path + "') threw an exception: " + e.message)
                        }
                    },
                    rmdir: function (path) {
                        var lookup = FS.lookupPath(path, {parent: true});
                        var parent = lookup.node;
                        var name = PATH.basename(path);
                        var node = FS.lookupNode(parent, name);
                        var errCode = FS.mayDelete(parent, name, true);
                        if (errCode) {
                            throw new FS.ErrnoError(errCode)
                        }
                        if (!parent.node_ops.rmdir) {
                            throw new FS.ErrnoError(63)
                        }
                        if (FS.isMountpoint(node)) {
                            throw new FS.ErrnoError(10)
                        }
                        try {
                            if (FS.trackingDelegate["willDeletePath"]) {
                                FS.trackingDelegate["willDeletePath"](path)
                            }
                        } catch (e) {
                            err("FS.trackingDelegate['willDeletePath']('" + path + "') threw an exception: " + e.message)
                        }
                        parent.node_ops.rmdir(parent, name);
                        FS.destroyNode(node);
                        try {
                            if (FS.trackingDelegate["onDeletePath"]) FS.trackingDelegate["onDeletePath"](path)
                        } catch (e) {
                            err("FS.trackingDelegate['onDeletePath']('" + path + "') threw an exception: " + e.message)
                        }
                    },
                    readdir: function (path) {
                        var lookup = FS.lookupPath(path, {follow: true});
                        var node = lookup.node;
                        if (!node.node_ops.readdir) {
                            throw new FS.ErrnoError(54)
                        }
                        return node.node_ops.readdir(node)
                    },
                    unlink: function (path) {
                        var lookup = FS.lookupPath(path, {parent: true});
                        var parent = lookup.node;
                        var name = PATH.basename(path);
                        var node = FS.lookupNode(parent, name);
                        var errCode = FS.mayDelete(parent, name, false);
                        if (errCode) {
                            throw new FS.ErrnoError(errCode)
                        }
                        if (!parent.node_ops.unlink) {
                            throw new FS.ErrnoError(63)
                        }
                        if (FS.isMountpoint(node)) {
                            throw new FS.ErrnoError(10)
                        }
                        try {
                            if (FS.trackingDelegate["willDeletePath"]) {
                                FS.trackingDelegate["willDeletePath"](path)
                            }
                        } catch (e) {
                            err("FS.trackingDelegate['willDeletePath']('" + path + "') threw an exception: " + e.message)
                        }
                        parent.node_ops.unlink(parent, name);
                        FS.destroyNode(node);
                        try {
                            if (FS.trackingDelegate["onDeletePath"]) FS.trackingDelegate["onDeletePath"](path)
                        } catch (e) {
                            err("FS.trackingDelegate['onDeletePath']('" + path + "') threw an exception: " + e.message)
                        }
                    },
                    readlink: function (path) {
                        var lookup = FS.lookupPath(path);
                        var link = lookup.node;
                        if (!link) {
                            throw new FS.ErrnoError(44)
                        }
                        if (!link.node_ops.readlink) {
                            throw new FS.ErrnoError(28)
                        }
                        return PATH_FS.resolve(FS.getPath(link.parent), link.node_ops.readlink(link))
                    },
                    stat: function (path, dontFollow) {
                        var lookup = FS.lookupPath(path, {follow: !dontFollow});
                        var node = lookup.node;
                        if (!node) {
                            throw new FS.ErrnoError(44)
                        }
                        if (!node.node_ops.getattr) {
                            throw new FS.ErrnoError(63)
                        }
                        return node.node_ops.getattr(node)
                    },
                    lstat: function (path) {
                        return FS.stat(path, true)
                    },
                    chmod: function (path, mode, dontFollow) {
                        var node;
                        if (typeof path === "string") {
                            var lookup = FS.lookupPath(path, {follow: !dontFollow});
                            node = lookup.node
                        } else {
                            node = path
                        }
                        if (!node.node_ops.setattr) {
                            throw new FS.ErrnoError(63)
                        }
                        node.node_ops.setattr(node, {mode: mode & 4095 | node.mode & ~4095, timestamp: Date.now()})
                    },
                    lchmod: function (path, mode) {
                        FS.chmod(path, mode, true)
                    },
                    fchmod: function (fd, mode) {
                        var stream = FS.getStream(fd);
                        if (!stream) {
                            throw new FS.ErrnoError(8)
                        }
                        FS.chmod(stream.node, mode)
                    },
                    chown: function (path, uid, gid, dontFollow) {
                        var node;
                        if (typeof path === "string") {
                            var lookup = FS.lookupPath(path, {follow: !dontFollow});
                            node = lookup.node
                        } else {
                            node = path
                        }
                        if (!node.node_ops.setattr) {
                            throw new FS.ErrnoError(63)
                        }
                        node.node_ops.setattr(node, {timestamp: Date.now()})
                    },
                    lchown: function (path, uid, gid) {
                        FS.chown(path, uid, gid, true)
                    },
                    fchown: function (fd, uid, gid) {
                        var stream = FS.getStream(fd);
                        if (!stream) {
                            throw new FS.ErrnoError(8)
                        }
                        FS.chown(stream.node, uid, gid)
                    },
                    truncate: function (path, len) {
                        if (len < 0) {
                            throw new FS.ErrnoError(28)
                        }
                        var node;
                        if (typeof path === "string") {
                            var lookup = FS.lookupPath(path, {follow: true});
                            node = lookup.node
                        } else {
                            node = path
                        }
                        if (!node.node_ops.setattr) {
                            throw new FS.ErrnoError(63)
                        }
                        if (FS.isDir(node.mode)) {
                            throw new FS.ErrnoError(31)
                        }
                        if (!FS.isFile(node.mode)) {
                            throw new FS.ErrnoError(28)
                        }
                        var errCode = FS.nodePermissions(node, "w");
                        if (errCode) {
                            throw new FS.ErrnoError(errCode)
                        }
                        node.node_ops.setattr(node, {size: len, timestamp: Date.now()})
                    },
                    ftruncate: function (fd, len) {
                        var stream = FS.getStream(fd);
                        if (!stream) {
                            throw new FS.ErrnoError(8)
                        }
                        if ((stream.flags & 2097155) === 0) {
                            throw new FS.ErrnoError(28)
                        }
                        FS.truncate(stream.node, len)
                    },
                    utime: function (path, atime, mtime) {
                        var lookup = FS.lookupPath(path, {follow: true});
                        var node = lookup.node;
                        node.node_ops.setattr(node, {timestamp: Math.max(atime, mtime)})
                    },
                    open: function (path, flags, mode, fd_start, fd_end) {
                        if (path === "") {
                            throw new FS.ErrnoError(44)
                        }
                        flags = typeof flags === "string" ? FS.modeStringToFlags(flags) : flags;
                        mode = typeof mode === "undefined" ? 438 : mode;
                        if (flags & 64) {
                            mode = mode & 4095 | 32768
                        } else {
                            mode = 0
                        }
                        var node;
                        if (typeof path === "object") {
                            node = path
                        } else {
                            path = PATH.normalize(path);
                            try {
                                var lookup = FS.lookupPath(path, {follow: !(flags & 131072)});
                                node = lookup.node
                            } catch (e) {
                            }
                        }
                        var created = false;
                        if (flags & 64) {
                            if (node) {
                                if (flags & 128) {
                                    throw new FS.ErrnoError(20)
                                }
                            } else {
                                node = FS.mknod(path, mode, 0);
                                created = true
                            }
                        }
                        if (!node) {
                            throw new FS.ErrnoError(44)
                        }
                        if (FS.isChrdev(node.mode)) {
                            flags &= ~512
                        }
                        if (flags & 65536 && !FS.isDir(node.mode)) {
                            throw new FS.ErrnoError(54)
                        }
                        if (!created) {
                            var errCode = FS.mayOpen(node, flags);
                            if (errCode) {
                                throw new FS.ErrnoError(errCode)
                            }
                        }
                        if (flags & 512) {
                            FS.truncate(node, 0)
                        }
                        flags &= ~(128 | 512 | 131072);
                        var stream = FS.createStream({
                            node: node,
                            path: FS.getPath(node),
                            flags: flags,
                            seekable: true,
                            position: 0,
                            stream_ops: node.stream_ops,
                            ungotten: [],
                            error: false
                        }, fd_start, fd_end);
                        if (stream.stream_ops.open) {
                            stream.stream_ops.open(stream)
                        }
                        if (Module["logReadFiles"] && !(flags & 1)) {
                            if (!FS.readFiles) FS.readFiles = {};
                            if (!(path in FS.readFiles)) {
                                FS.readFiles[path] = 1;
                                err("FS.trackingDelegate error on read file: " + path)
                            }
                        }
                        try {
                            if (FS.trackingDelegate["onOpenFile"]) {
                                var trackingFlags = 0;
                                if ((flags & 2097155) !== 1) {
                                    trackingFlags |= FS.tracking.openFlags.READ
                                }
                                if ((flags & 2097155) !== 0) {
                                    trackingFlags |= FS.tracking.openFlags.WRITE
                                }
                                FS.trackingDelegate["onOpenFile"](path, trackingFlags)
                            }
                        } catch (e) {
                            err("FS.trackingDelegate['onOpenFile']('" + path + "', flags) threw an exception: " + e.message)
                        }
                        return stream
                    },
                    close: function (stream) {
                        if (FS.isClosed(stream)) {
                            throw new FS.ErrnoError(8)
                        }
                        if (stream.getdents) stream.getdents = null;
                        try {
                            if (stream.stream_ops.close) {
                                stream.stream_ops.close(stream)
                            }
                        } catch (e) {
                            throw e
                        } finally {
                            FS.closeStream(stream.fd)
                        }
                        stream.fd = null
                    },
                    isClosed: function (stream) {
                        return stream.fd === null
                    },
                    llseek: function (stream, offset, whence) {
                        if (FS.isClosed(stream)) {
                            throw new FS.ErrnoError(8)
                        }
                        if (!stream.seekable || !stream.stream_ops.llseek) {
                            throw new FS.ErrnoError(70)
                        }
                        if (whence != 0 && whence != 1 && whence != 2) {
                            throw new FS.ErrnoError(28)
                        }
                        stream.position = stream.stream_ops.llseek(stream, offset, whence);
                        stream.ungotten = [];
                        return stream.position
                    },
                    read: function (stream, buffer, offset, length, position) {
                        if (length < 0 || position < 0) {
                            throw new FS.ErrnoError(28)
                        }
                        if (FS.isClosed(stream)) {
                            throw new FS.ErrnoError(8)
                        }
                        if ((stream.flags & 2097155) === 1) {
                            throw new FS.ErrnoError(8)
                        }
                        if (FS.isDir(stream.node.mode)) {
                            throw new FS.ErrnoError(31)
                        }
                        if (!stream.stream_ops.read) {
                            throw new FS.ErrnoError(28)
                        }
                        var seeking = typeof position !== "undefined";
                        if (!seeking) {
                            position = stream.position
                        } else if (!stream.seekable) {
                            throw new FS.ErrnoError(70)
                        }
                        var bytesRead = stream.stream_ops.read(stream, buffer, offset, length, position);
                        if (!seeking) stream.position += bytesRead;
                        return bytesRead
                    },
                    write: function (stream, buffer, offset, length, position, canOwn) {
                        if (length < 0 || position < 0) {
                            throw new FS.ErrnoError(28)
                        }
                        if (FS.isClosed(stream)) {
                            throw new FS.ErrnoError(8)
                        }
                        if ((stream.flags & 2097155) === 0) {
                            throw new FS.ErrnoError(8)
                        }
                        if (FS.isDir(stream.node.mode)) {
                            throw new FS.ErrnoError(31)
                        }
                        if (!stream.stream_ops.write) {
                            throw new FS.ErrnoError(28)
                        }
                        if (stream.seekable && stream.flags & 1024) {
                            FS.llseek(stream, 0, 2)
                        }
                        var seeking = typeof position !== "undefined";
                        if (!seeking) {
                            position = stream.position
                        } else if (!stream.seekable) {
                            throw new FS.ErrnoError(70)
                        }
                        var bytesWritten = stream.stream_ops.write(stream, buffer, offset, length, position, canOwn);
                        if (!seeking) stream.position += bytesWritten;
                        try {
                            if (stream.path && FS.trackingDelegate["onWriteToFile"]) FS.trackingDelegate["onWriteToFile"](stream.path)
                        } catch (e) {
                            err("FS.trackingDelegate['onWriteToFile']('" + stream.path + "') threw an exception: " + e.message)
                        }
                        return bytesWritten
                    },
                    allocate: function (stream, offset, length) {
                        if (FS.isClosed(stream)) {
                            throw new FS.ErrnoError(8)
                        }
                        if (offset < 0 || length <= 0) {
                            throw new FS.ErrnoError(28)
                        }
                        if ((stream.flags & 2097155) === 0) {
                            throw new FS.ErrnoError(8)
                        }
                        if (!FS.isFile(stream.node.mode) && !FS.isDir(stream.node.mode)) {
                            throw new FS.ErrnoError(43)
                        }
                        if (!stream.stream_ops.allocate) {
                            throw new FS.ErrnoError(138)
                        }
                        stream.stream_ops.allocate(stream, offset, length)
                    },
                    mmap: function (stream, address, length, position, prot, flags) {
                        if ((prot & 2) !== 0 && (flags & 2) === 0 && (stream.flags & 2097155) !== 2) {
                            throw new FS.ErrnoError(2)
                        }
                        if ((stream.flags & 2097155) === 1) {
                            throw new FS.ErrnoError(2)
                        }
                        if (!stream.stream_ops.mmap) {
                            throw new FS.ErrnoError(43)
                        }
                        return stream.stream_ops.mmap(stream, address, length, position, prot, flags)
                    },
                    msync: function (stream, buffer, offset, length, mmapFlags) {
                        if (!stream || !stream.stream_ops.msync) {
                            return 0
                        }
                        return stream.stream_ops.msync(stream, buffer, offset, length, mmapFlags)
                    },
                    munmap: function (stream) {
                        return 0
                    },
                    ioctl: function (stream, cmd, arg) {
                        if (!stream.stream_ops.ioctl) {
                            throw new FS.ErrnoError(59)
                        }
                        return stream.stream_ops.ioctl(stream, cmd, arg)
                    },
                    readFile: function (path, opts) {
                        opts = opts || {};
                        opts.flags = opts.flags || "r";
                        opts.encoding = opts.encoding || "binary";
                        if (opts.encoding !== "utf8" && opts.encoding !== "binary") {
                            throw new Error('Invalid encoding type "' + opts.encoding + '"')
                        }
                        var ret;
                        var stream = FS.open(path, opts.flags);
                        var stat = FS.stat(path);
                        var length = stat.size;
                        var buf = new Uint8Array(length);
                        FS.read(stream, buf, 0, length, 0);
                        if (opts.encoding === "utf8") {
                            ret = UTF8ArrayToString(buf, 0)
                        } else if (opts.encoding === "binary") {
                            ret = buf
                        }
                        FS.close(stream);
                        return ret
                    },
                    writeFile: function (path, data, opts) {
                        opts = opts || {};
                        opts.flags = opts.flags || "w";
                        var stream = FS.open(path, opts.flags, opts.mode);
                        if (typeof data === "string") {
                            var buf = new Uint8Array(lengthBytesUTF8(data) + 1);
                            var actualNumBytes = stringToUTF8Array(data, buf, 0, buf.length);
                            FS.write(stream, buf, 0, actualNumBytes, undefined, opts.canOwn)
                        } else if (ArrayBuffer.isView(data)) {
                            FS.write(stream, data, 0, data.byteLength, undefined, opts.canOwn)
                        } else {
                            throw new Error("Unsupported data type")
                        }
                        FS.close(stream)
                    },
                    cwd: function () {
                        return FS.currentPath
                    },
                    chdir: function (path) {
                        var lookup = FS.lookupPath(path, {follow: true});
                        if (lookup.node === null) {
                            throw new FS.ErrnoError(44)
                        }
                        if (!FS.isDir(lookup.node.mode)) {
                            throw new FS.ErrnoError(54)
                        }
                        var errCode = FS.nodePermissions(lookup.node, "x");
                        if (errCode) {
                            throw new FS.ErrnoError(errCode)
                        }
                        FS.currentPath = lookup.path
                    },
                    createDefaultDirectories: function () {
                        FS.mkdir("/tmp");
                        FS.mkdir("/home");
                        FS.mkdir("/home/web_user")
                    },
                    createDefaultDevices: function () {
                        FS.mkdir("/dev");
                        FS.registerDevice(FS.makedev(1, 3), {
                            read: function () {
                                return 0
                            }, write: function (stream, buffer, offset, length, pos) {
                                return length
                            }
                        });
                        FS.mkdev("/dev/null", FS.makedev(1, 3));
                        TTY.register(FS.makedev(5, 0), TTY.default_tty_ops);
                        TTY.register(FS.makedev(6, 0), TTY.default_tty1_ops);
                        FS.mkdev("/dev/tty", FS.makedev(5, 0));
                        FS.mkdev("/dev/tty1", FS.makedev(6, 0));
                        var random_device;
                        if (typeof crypto === "object" && typeof crypto["getRandomValues"] === "function") {
                            var randomBuffer = new Uint8Array(1);
                            random_device = function () {
                                crypto.getRandomValues(randomBuffer);
                                return randomBuffer[0]
                            }
                        } else if (ENVIRONMENT_IS_NODE) {
                            try {
                                var crypto_module = require("crypto");
                                random_device = function () {
                                    return crypto_module["randomBytes"](1)[0]
                                }
                            } catch (e) {
                            }
                        } else {
                        }
                        if (!random_device) {
                            random_device = function () {
                                abort("random_device")
                            }
                        }
                        FS.createDevice("/dev", "random", random_device);
                        FS.createDevice("/dev", "urandom", random_device);
                        FS.mkdir("/dev/shm");
                        FS.mkdir("/dev/shm/tmp")
                    },
                    createSpecialDirectories: function () {
                        FS.mkdir("/proc");
                        FS.mkdir("/proc/self");
                        FS.mkdir("/proc/self/fd");
                        FS.mount({
                            mount: function () {
                                var node = FS.createNode("/proc/self", "fd", 16384 | 511, 73);
                                node.node_ops = {
                                    lookup: function (parent, name) {
                                        var fd = +name;
                                        var stream = FS.getStream(fd);
                                        if (!stream) throw new FS.ErrnoError(8);
                                        var ret = {
                                            parent: null,
                                            mount: {mountpoint: "fake"},
                                            node_ops: {
                                                readlink: function () {
                                                    return stream.path
                                                }
                                            }
                                        };
                                        ret.parent = ret;
                                        return ret
                                    }
                                };
                                return node
                            }
                        }, {}, "/proc/self/fd")
                    },
                    createStandardStreams: function () {
                        if (Module["stdin"]) {
                            FS.createDevice("/dev", "stdin", Module["stdin"])
                        } else {
                            FS.symlink("/dev/tty", "/dev/stdin")
                        }
                        if (Module["stdout"]) {
                            FS.createDevice("/dev", "stdout", null, Module["stdout"])
                        } else {
                            FS.symlink("/dev/tty", "/dev/stdout")
                        }
                        if (Module["stderr"]) {
                            FS.createDevice("/dev", "stderr", null, Module["stderr"])
                        } else {
                            FS.symlink("/dev/tty1", "/dev/stderr")
                        }
                        var stdin = FS.open("/dev/stdin", "r");
                        var stdout = FS.open("/dev/stdout", "w");
                        var stderr = FS.open("/dev/stderr", "w")
                    },
                    ensureErrnoError: function () {
                        if (FS.ErrnoError) return;
                        FS.ErrnoError = function ErrnoError(errno, node) {
                            this.node = node;
                            this.setErrno = function (errno) {
                                this.errno = errno
                            };
                            this.setErrno(errno);
                            this.message = "FS error"
                        };
                        FS.ErrnoError.prototype = new Error;
                        FS.ErrnoError.prototype.constructor = FS.ErrnoError;
                        [44].forEach(function (code) {
                            FS.genericErrors[code] = new FS.ErrnoError(code);
                            FS.genericErrors[code].stack = "<generic error, no stack>"
                        })
                    },
                    staticInit: function () {
                        FS.ensureErrnoError();
                        FS.nameTable = new Array(4096);
                        FS.mount(MEMFS, {}, "/");
                        FS.createDefaultDirectories();
                        FS.createDefaultDevices();
                        FS.createSpecialDirectories();
                        FS.filesystems = {"MEMFS": MEMFS}
                    },
                    init: function (input, output, error) {
                        FS.init.initialized = true;
                        FS.ensureErrnoError();
                        Module["stdin"] = input || Module["stdin"];
                        Module["stdout"] = output || Module["stdout"];
                        Module["stderr"] = error || Module["stderr"];
                        FS.createStandardStreams()
                    },
                    quit: function () {
                        FS.init.initialized = false;
                        var fflush = Module["_fflush"];
                        if (fflush) fflush(0);
                        for (var i = 0; i < FS.streams.length; i++) {
                            var stream = FS.streams[i];
                            if (!stream) {
                                continue
                            }
                            FS.close(stream)
                        }
                    },
                    getMode: function (canRead, canWrite) {
                        var mode = 0;
                        if (canRead) mode |= 292 | 73;
                        if (canWrite) mode |= 146;
                        return mode
                    },
                    joinPath: function (parts, forceRelative) {
                        var path = PATH.join.apply(null, parts);
                        if (forceRelative && path[0] == "/") path = path.substr(1);
                        return path
                    },
                    absolutePath: function (relative, base) {
                        return PATH_FS.resolve(base, relative)
                    },
                    standardizePath: function (path) {
                        return PATH.normalize(path)
                    },
                    findObject: function (path, dontResolveLastLink) {
                        var ret = FS.analyzePath(path, dontResolveLastLink);
                        if (ret.exists) {
                            return ret.object
                        } else {
                            setErrNo(ret.error);
                            return null
                        }
                    },
                    analyzePath: function (path, dontResolveLastLink) {
                        try {
                            var lookup = FS.lookupPath(path, {follow: !dontResolveLastLink});
                            path = lookup.path
                        } catch (e) {
                        }
                        var ret = {
                            isRoot: false,
                            exists: false,
                            error: 0,
                            name: null,
                            path: null,
                            object: null,
                            parentExists: false,
                            parentPath: null,
                            parentObject: null
                        };
                        try {
                            var lookup = FS.lookupPath(path, {parent: true});
                            ret.parentExists = true;
                            ret.parentPath = lookup.path;
                            ret.parentObject = lookup.node;
                            ret.name = PATH.basename(path);
                            lookup = FS.lookupPath(path, {follow: !dontResolveLastLink});
                            ret.exists = true;
                            ret.path = lookup.path;
                            ret.object = lookup.node;
                            ret.name = lookup.node.name;
                            ret.isRoot = lookup.path === "/"
                        } catch (e) {
                            ret.error = e.errno
                        }
                        return ret
                    },
                    createFolder: function (parent, name, canRead, canWrite) {
                        var path = PATH.join2(typeof parent === "string" ? parent : FS.getPath(parent), name);
                        var mode = FS.getMode(canRead, canWrite);
                        return FS.mkdir(path, mode)
                    },
                    createPath: function (parent, path, canRead, canWrite) {
                        parent = typeof parent === "string" ? parent : FS.getPath(parent);
                        var parts = path.split("/").reverse();
                        while (parts.length) {
                            var part = parts.pop();
                            if (!part) continue;
                            var current = PATH.join2(parent, part);
                            try {
                                FS.mkdir(current)
                            } catch (e) {
                            }
                            parent = current
                        }
                        return current
                    },
                    createFile: function (parent, name, properties, canRead, canWrite) {
                        var path = PATH.join2(typeof parent === "string" ? parent : FS.getPath(parent), name);
                        var mode = FS.getMode(canRead, canWrite);
                        return FS.create(path, mode)
                    },
                    createDataFile: function (parent, name, data, canRead, canWrite, canOwn) {
                        var path = name ? PATH.join2(typeof parent === "string" ? parent : FS.getPath(parent), name) : parent;
                        var mode = FS.getMode(canRead, canWrite);
                        var node = FS.create(path, mode);
                        if (data) {
                            if (typeof data === "string") {
                                var arr = new Array(data.length);
                                for (var i = 0, len = data.length; i < len; ++i) arr[i] = data.charCodeAt(i);
                                data = arr
                            }
                            FS.chmod(node, mode | 146);
                            var stream = FS.open(node, "w");
                            FS.write(stream, data, 0, data.length, 0, canOwn);
                            FS.close(stream);
                            FS.chmod(node, mode)
                        }
                        return node
                    },
                    createDevice: function (parent, name, input, output) {
                        var path = PATH.join2(typeof parent === "string" ? parent : FS.getPath(parent), name);
                        var mode = FS.getMode(!!input, !!output);
                        if (!FS.createDevice.major) FS.createDevice.major = 64;
                        var dev = FS.makedev(FS.createDevice.major++, 0);
                        FS.registerDevice(dev, {
                            open: function (stream) {
                                stream.seekable = false
                            }, close: function (stream) {
                                if (output && output.buffer && output.buffer.length) {
                                    output(10)
                                }
                            }, read: function (stream, buffer, offset, length, pos) {
                                var bytesRead = 0;
                                for (var i = 0; i < length; i++) {
                                    var result;
                                    try {
                                        result = input()
                                    } catch (e) {
                                        throw new FS.ErrnoError(29)
                                    }
                                    if (result === undefined && bytesRead === 0) {
                                        throw new FS.ErrnoError(6)
                                    }
                                    if (result === null || result === undefined) break;
                                    bytesRead++;
                                    buffer[offset + i] = result
                                }
                                if (bytesRead) {
                                    stream.node.timestamp = Date.now()
                                }
                                return bytesRead
                            }, write: function (stream, buffer, offset, length, pos) {
                                for (var i = 0; i < length; i++) {
                                    try {
                                        output(buffer[offset + i])
                                    } catch (e) {
                                        throw new FS.ErrnoError(29)
                                    }
                                }
                                if (length) {
                                    stream.node.timestamp = Date.now()
                                }
                                return i
                            }
                        });
                        return FS.mkdev(path, mode, dev)
                    },
                    createLink: function (parent, name, target, canRead, canWrite) {
                        var path = PATH.join2(typeof parent === "string" ? parent : FS.getPath(parent), name);
                        return FS.symlink(target, path)
                    },
                    forceLoadFile: function (obj) {
                        if (obj.isDevice || obj.isFolder || obj.link || obj.contents) return true;
                        var success = true;
                        if (typeof XMLHttpRequest !== "undefined") {
                            throw new Error("Lazy loading should have been performed (contents set) in createLazyFile, but it was not. Lazy loading only works in web workers. Use --embed-file or --preload-file in emcc on the main thread.")
                        } else if (read_) {
                            try {
                                obj.contents = intArrayFromString(read_(obj.url), true);
                                obj.usedBytes = obj.contents.length
                            } catch (e) {
                                success = false
                            }
                        } else {
                            throw new Error("Cannot load without read() or XMLHttpRequest.")
                        }
                        if (!success) setErrNo(29);
                        return success
                    },
                    createLazyFile: function (parent, name, url, canRead, canWrite) {
                        function LazyUint8Array() {
                            this.lengthKnown = false;
                            this.chunks = []
                        }

                        LazyUint8Array.prototype.get = function LazyUint8Array_get(idx) {
                            if (idx > this.length - 1 || idx < 0) {
                                return undefined
                            }
                            var chunkOffset = idx % this.chunkSize;
                            var chunkNum = idx / this.chunkSize | 0;
                            return this.getter(chunkNum)[chunkOffset]
                        };
                        LazyUint8Array.prototype.setDataGetter = function LazyUint8Array_setDataGetter(getter) {
                            this.getter = getter
                        };
                        LazyUint8Array.prototype.cacheLength = function LazyUint8Array_cacheLength() {
                            var xhr = new XMLHttpRequest;
                            xhr.open("HEAD", url, false);
                            xhr.send(null);
                            if (!(xhr.status >= 200 && xhr.status < 300 || xhr.status === 304)) throw new Error("Couldn't load " + url + ". Status: " + xhr.status);
                            var datalength = Number(xhr.getResponseHeader("Content-length"));
                            var header;
                            var hasByteServing = (header = xhr.getResponseHeader("Accept-Ranges")) && header === "bytes";
                            var usesGzip = (header = xhr.getResponseHeader("Content-Encoding")) && header === "gzip";
                            var chunkSize = 1024 * 1024;
                            if (!hasByteServing) chunkSize = datalength;
                            var doXHR = function (from, to) {
                                if (from > to) throw new Error("invalid range (" + from + ", " + to + ") or no bytes requested!");
                                if (to > datalength - 1) throw new Error("only " + datalength + " bytes available! programmer error!");
                                var xhr = new XMLHttpRequest;
                                xhr.open("GET", url, false);
                                if (datalength !== chunkSize) xhr.setRequestHeader("Range", "bytes=" + from + "-" + to);
                                if (typeof Uint8Array != "undefined") xhr.responseType = "arraybuffer";
                                if (xhr.overrideMimeType) {
                                    xhr.overrideMimeType("text/plain; charset=x-user-defined")
                                }
                                xhr.send(null);
                                if (!(xhr.status >= 200 && xhr.status < 300 || xhr.status === 304)) throw new Error("Couldn't load " + url + ". Status: " + xhr.status);
                                if (xhr.response !== undefined) {
                                    return new Uint8Array(xhr.response || [])
                                } else {
                                    return intArrayFromString(xhr.responseText || "", true)
                                }
                            };
                            var lazyArray = this;
                            lazyArray.setDataGetter(function (chunkNum) {
                                var start = chunkNum * chunkSize;
                                var end = (chunkNum + 1) * chunkSize - 1;
                                end = Math.min(end, datalength - 1);
                                if (typeof lazyArray.chunks[chunkNum] === "undefined") {
                                    lazyArray.chunks[chunkNum] = doXHR(start, end)
                                }
                                if (typeof lazyArray.chunks[chunkNum] === "undefined") throw new Error("doXHR failed!");
                                return lazyArray.chunks[chunkNum]
                            });
                            if (usesGzip || !datalength) {
                                chunkSize = datalength = 1;
                                datalength = this.getter(0).length;
                                chunkSize = datalength;
                                out("LazyFiles on gzip forces download of the whole file when length is accessed")
                            }
                            this._length = datalength;
                            this._chunkSize = chunkSize;
                            this.lengthKnown = true
                        };
                        if (typeof XMLHttpRequest !== "undefined") {
                            if (!ENVIRONMENT_IS_WORKER) throw"Cannot do synchronous binary XHRs outside webworkers in modern browsers. Use --embed-file or --preload-file in emcc";
                            var lazyArray = new LazyUint8Array;
                            Object.defineProperties(lazyArray, {
                                length: {
                                    get: function () {
                                        if (!this.lengthKnown) {
                                            this.cacheLength()
                                        }
                                        return this._length
                                    }
                                }, chunkSize: {
                                    get: function () {
                                        if (!this.lengthKnown) {
                                            this.cacheLength()
                                        }
                                        return this._chunkSize
                                    }
                                }
                            });
                            var properties = {isDevice: false, contents: lazyArray}
                        } else {
                            var properties = {isDevice: false, url: url}
                        }
                        var node = FS.createFile(parent, name, properties, canRead, canWrite);
                        if (properties.contents) {
                            node.contents = properties.contents
                        } else if (properties.url) {
                            node.contents = null;
                            node.url = properties.url
                        }
                        Object.defineProperties(node, {
                            usedBytes: {
                                get: function () {
                                    return this.contents.length
                                }
                            }
                        });
                        var stream_ops = {};
                        var keys = Object.keys(node.stream_ops);
                        keys.forEach(function (key) {
                            var fn = node.stream_ops[key];
                            stream_ops[key] = function forceLoadLazyFile() {
                                if (!FS.forceLoadFile(node)) {
                                    throw new FS.ErrnoError(29)
                                }
                                return fn.apply(null, arguments)
                            }
                        });
                        stream_ops.read = function stream_ops_read(stream, buffer, offset, length, position) {
                            if (!FS.forceLoadFile(node)) {
                                throw new FS.ErrnoError(29)
                            }
                            var contents = stream.node.contents;
                            if (position >= contents.length) return 0;
                            var size = Math.min(contents.length - position, length);
                            if (contents.slice) {
                                for (var i = 0; i < size; i++) {
                                    buffer[offset + i] = contents[position + i]
                                }
                            } else {
                                for (var i = 0; i < size; i++) {
                                    buffer[offset + i] = contents.get(position + i)
                                }
                            }
                            return size
                        };
                        node.stream_ops = stream_ops;
                        return node
                    },
                    createPreloadedFile: function (parent, name, url, canRead, canWrite, onload, onerror, dontCreateFile, canOwn, preFinish) {
                        Browser.init();
                        var fullname = name ? PATH_FS.resolve(PATH.join2(parent, name)) : parent;
                        var dep = getUniqueRunDependency("cp " + fullname);

                        function processData(byteArray) {
                            function finish(byteArray) {
                                if (preFinish) preFinish();
                                if (!dontCreateFile) {
                                    FS.createDataFile(parent, name, byteArray, canRead, canWrite, canOwn)
                                }
                                if (onload) onload();
                                removeRunDependency(dep)
                            }

                            var handled = false;
                            Module["preloadPlugins"].forEach(function (plugin) {
                                if (handled) return;
                                if (plugin["canHandle"](fullname)) {
                                    plugin["handle"](byteArray, fullname, finish, function () {
                                        if (onerror) onerror();
                                        removeRunDependency(dep)
                                    });
                                    handled = true
                                }
                            });
                            if (!handled) finish(byteArray)
                        }

                        addRunDependency(dep);
                        if (typeof url == "string") {
                            Browser.asyncLoad(url, function (byteArray) {
                                processData(byteArray)
                            }, onerror)
                        } else {
                            processData(url)
                        }
                    },
                    indexedDB: function () {
                        return window.indexedDB || window.mozIndexedDB || window.webkitIndexedDB || window.msIndexedDB
                    },
                    DB_NAME: function () {
                        return "EM_FS_" + window.location.pathname
                    },
                    DB_VERSION: 20,
                    DB_STORE_NAME: "FILE_DATA",
                    saveFilesToDB: function (paths, onload, onerror) {
                        onload = onload || function () {
                        };
                        onerror = onerror || function () {
                        };
                        var indexedDB = FS.indexedDB();
                        try {
                            var openRequest = indexedDB.open(FS.DB_NAME(), FS.DB_VERSION)
                        } catch (e) {
                            return onerror(e)
                        }
                        openRequest.onupgradeneeded = function openRequest_onupgradeneeded() {
                            out("creating db");
                            var db = openRequest.result;
                            db.createObjectStore(FS.DB_STORE_NAME)
                        };
                        openRequest.onsuccess = function openRequest_onsuccess() {
                            var db = openRequest.result;
                            var transaction = db.transaction([FS.DB_STORE_NAME], "readwrite");
                            var files = transaction.objectStore(FS.DB_STORE_NAME);
                            var ok = 0, fail = 0, total = paths.length;

                            function finish() {
                                if (fail == 0) onload(); else onerror()
                            }

                            paths.forEach(function (path) {
                                var putRequest = files.put(FS.analyzePath(path).object.contents, path);
                                putRequest.onsuccess = function putRequest_onsuccess() {
                                    ok++;
                                    if (ok + fail == total) finish()
                                };
                                putRequest.onerror = function putRequest_onerror() {
                                    fail++;
                                    if (ok + fail == total) finish()
                                }
                            });
                            transaction.onerror = onerror
                        };
                        openRequest.onerror = onerror
                    },
                    loadFilesFromDB: function (paths, onload, onerror) {
                        onload = onload || function () {
                        };
                        onerror = onerror || function () {
                        };
                        var indexedDB = FS.indexedDB();
                        try {
                            var openRequest = indexedDB.open(FS.DB_NAME(), FS.DB_VERSION)
                        } catch (e) {
                            return onerror(e)
                        }
                        openRequest.onupgradeneeded = onerror;
                        openRequest.onsuccess = function openRequest_onsuccess() {
                            var db = openRequest.result;
                            try {
                                var transaction = db.transaction([FS.DB_STORE_NAME], "readonly")
                            } catch (e) {
                                onerror(e);
                                return
                            }
                            var files = transaction.objectStore(FS.DB_STORE_NAME);
                            var ok = 0, fail = 0, total = paths.length;

                            function finish() {
                                if (fail == 0) onload(); else onerror()
                            }

                            paths.forEach(function (path) {
                                var getRequest = files.get(path);
                                getRequest.onsuccess = function getRequest_onsuccess() {
                                    if (FS.analyzePath(path).exists) {
                                        FS.unlink(path)
                                    }
                                    FS.createDataFile(PATH.dirname(path), PATH.basename(path), getRequest.result, true, true, true);
                                    ok++;
                                    if (ok + fail == total) finish()
                                };
                                getRequest.onerror = function getRequest_onerror() {
                                    fail++;
                                    if (ok + fail == total) finish()
                                }
                            });
                            transaction.onerror = onerror
                        };
                        openRequest.onerror = onerror
                    },
                    mmapAlloc: function (size) {
                        var alignedSize = alignMemory(size, 16384);
                        var ptr = _malloc(alignedSize);
                        while (size < alignedSize) HEAP8[ptr + size++] = 0;
                        return ptr
                    }
                };
                var SYSCALLS = {
                    mappings: {}, DEFAULT_POLLMASK: 5, umask: 511, calculateAt: function (dirfd, path) {
                        if (path[0] !== "/") {
                            var dir;
                            if (dirfd === -100) {
                                dir = FS.cwd()
                            } else {
                                var dirstream = FS.getStream(dirfd);
                                if (!dirstream) throw new FS.ErrnoError(8);
                                dir = dirstream.path
                            }
                            path = PATH.join2(dir, path)
                        }
                        return path
                    }, doStat: function (func, path, buf) {
                        try {
                            var stat = func(path)
                        } catch (e) {
                            if (e && e.node && PATH.normalize(path) !== PATH.normalize(FS.getPath(e.node))) {
                                return -54
                            }
                            throw e
                        }
                        HEAP32[buf >> 2] = stat.dev;
                        HEAP32[buf + 4 >> 2] = 0;
                        HEAP32[buf + 8 >> 2] = stat.ino;
                        HEAP32[buf + 12 >> 2] = stat.mode;
                        HEAP32[buf + 16 >> 2] = stat.nlink;
                        HEAP32[buf + 20 >> 2] = stat.uid;
                        HEAP32[buf + 24 >> 2] = stat.gid;
                        HEAP32[buf + 28 >> 2] = stat.rdev;
                        HEAP32[buf + 32 >> 2] = 0;
                        tempI64 = [stat.size >>> 0, (tempDouble = stat.size, +Math_abs(tempDouble) >= 1 ? tempDouble > 0 ? (Math_min(+Math_floor(tempDouble / 4294967296), 4294967295) | 0) >>> 0 : ~~+Math_ceil((tempDouble - +(~~tempDouble >>> 0)) / 4294967296) >>> 0 : 0)], HEAP32[buf + 40 >> 2] = tempI64[0], HEAP32[buf + 44 >> 2] = tempI64[1];
                        HEAP32[buf + 48 >> 2] = 4096;
                        HEAP32[buf + 52 >> 2] = stat.blocks;
                        HEAP32[buf + 56 >> 2] = stat.atime.getTime() / 1e3 | 0;
                        HEAP32[buf + 60 >> 2] = 0;
                        HEAP32[buf + 64 >> 2] = stat.mtime.getTime() / 1e3 | 0;
                        HEAP32[buf + 68 >> 2] = 0;
                        HEAP32[buf + 72 >> 2] = stat.ctime.getTime() / 1e3 | 0;
                        HEAP32[buf + 76 >> 2] = 0;
                        tempI64 = [stat.ino >>> 0, (tempDouble = stat.ino, +Math_abs(tempDouble) >= 1 ? tempDouble > 0 ? (Math_min(+Math_floor(tempDouble / 4294967296), 4294967295) | 0) >>> 0 : ~~+Math_ceil((tempDouble - +(~~tempDouble >>> 0)) / 4294967296) >>> 0 : 0)], HEAP32[buf + 80 >> 2] = tempI64[0], HEAP32[buf + 84 >> 2] = tempI64[1];
                        return 0
                    }, doMsync: function (addr, stream, len, flags, offset) {
                        var buffer = HEAPU8.slice(addr, addr + len);
                        FS.msync(stream, buffer, offset, len, flags)
                    }, doMkdir: function (path, mode) {
                        path = PATH.normalize(path);
                        if (path[path.length - 1] === "/") path = path.substr(0, path.length - 1);
                        FS.mkdir(path, mode, 0);
                        return 0
                    }, doMknod: function (path, mode, dev) {
                        switch (mode & 61440) {
                            case 32768:
                            case 8192:
                            case 24576:
                            case 4096:
                            case 49152:
                                break;
                            default:
                                return -28
                        }
                        FS.mknod(path, mode, dev);
                        return 0
                    }, doReadlink: function (path, buf, bufsize) {
                        if (bufsize <= 0) return -28;
                        var ret = FS.readlink(path);
                        var len = Math.min(bufsize, lengthBytesUTF8(ret));
                        var endChar = HEAP8[buf + len];
                        stringToUTF8(ret, buf, bufsize + 1);
                        HEAP8[buf + len] = endChar;
                        return len
                    }, doAccess: function (path, amode) {
                        if (amode & ~7) {
                            return -28
                        }
                        var node;
                        var lookup = FS.lookupPath(path, {follow: true});
                        node = lookup.node;
                        if (!node) {
                            return -44
                        }
                        var perms = "";
                        if (amode & 4) perms += "r";
                        if (amode & 2) perms += "w";
                        if (amode & 1) perms += "x";
                        if (perms && FS.nodePermissions(node, perms)) {
                            return -2
                        }
                        return 0
                    }, doDup: function (path, flags, suggestFD) {
                        var suggest = FS.getStream(suggestFD);
                        if (suggest) FS.close(suggest);
                        return FS.open(path, flags, 0, suggestFD, suggestFD).fd
                    }, doReadv: function (stream, iov, iovcnt, offset) {
                        var ret = 0;
                        for (var i = 0; i < iovcnt; i++) {
                            var ptr = HEAP32[iov + i * 8 >> 2];
                            var len = HEAP32[iov + (i * 8 + 4) >> 2];
                            var curr = FS.read(stream, HEAP8, ptr, len, offset);
                            if (curr < 0) return -1;
                            ret += curr;
                            if (curr < len) break
                        }
                        return ret
                    }, doWritev: function (stream, iov, iovcnt, offset) {
                        var ret = 0;
                        for (var i = 0; i < iovcnt; i++) {
                            var ptr = HEAP32[iov + i * 8 >> 2];
                            var len = HEAP32[iov + (i * 8 + 4) >> 2];
                            var curr = FS.write(stream, HEAP8, ptr, len, offset);
                            if (curr < 0) return -1;
                            ret += curr
                        }
                        return ret
                    }, varargs: undefined, get: function () {
                        SYSCALLS.varargs += 4;
                        var ret = HEAP32[SYSCALLS.varargs - 4 >> 2];
                        return ret
                    }, getStr: function (ptr) {
                        var ret = UTF8ToString(ptr);
                        return ret
                    }, getStreamFromFD: function (fd) {
                        var stream = FS.getStream(fd);
                        if (!stream) throw new FS.ErrnoError(8);
                        return stream
                    }, get64: function (low, high) {
                        return low
                    }
                };

                function syscallMunmap(addr, len) {
                    if ((addr | 0) === -1 || len === 0) {
                        return -28
                    }
                    var info = SYSCALLS.mappings[addr];
                    if (!info) return 0;
                    if (len === info.len) {
                        var stream = FS.getStream(info.fd);
                        if (info.prot & 2) {
                            SYSCALLS.doMsync(addr, stream, len, info.flags, info.offset)
                        }
                        FS.munmap(stream);
                        SYSCALLS.mappings[addr] = null;
                        if (info.allocated) {
                            _free(info.malloc)
                        }
                    }
                    return 0
                }

                function ___sys_munmap(addr, len) {
                    try {
                        return syscallMunmap(addr, len)
                    } catch (e) {
                        if (typeof FS === "undefined" || !(e instanceof FS.ErrnoError)) abort(e);
                        return -e.errno
                    }
                }

                var tupleRegistrations = {};

                function runDestructors(destructors) {
                    while (destructors.length) {
                        var ptr = destructors.pop();
                        var del = destructors.pop();
                        del(ptr)
                    }
                }

                function simpleReadValueFromPointer(pointer) {
                    return this["fromWireType"](HEAPU32[pointer >> 2])
                }

                var awaitingDependencies = {};
                var registeredTypes = {};
                var typeDependencies = {};
                var char_0 = 48;
                var char_9 = 57;

                function makeLegalFunctionName(name) {
                    if (undefined === name) {
                        return "_unknown"
                    }
                    name = name.replace(/[^a-zA-Z0-9_]/g, "$");
                    var f = name.charCodeAt(0);
                    if (f >= char_0 && f <= char_9) {
                        return "_" + name
                    } else {
                        return name
                    }
                }

                function createNamedFunction(name, body) {
                    name = makeLegalFunctionName(name);
                    if (IsWechat) {
                        var f1 = function (body) {
                            return function () {
                                "use strict";
                                return body.apply(this, arguments);
                            }
                        }
                        return f1(body)
                    } else {
                        return new Function("body", "return function " + name + "() {\n" + '    "use strict";' + "    return body.apply(this, arguments);\n" + "};\n")(body)
                    }
                }

                function extendError(baseErrorType, errorName) {
                    var errorClass = createNamedFunction(errorName, function (message) {
                        this.name = errorName;
                        this.message = message;
                        var stack = new Error(message).stack;
                        if (stack !== undefined) {
                            this.stack = this.toString() + "\n" + stack.replace(/^Error(:[^\n]*)?\n/, "")
                        }
                    });
                    errorClass.prototype = Object.create(baseErrorType.prototype);
                    errorClass.prototype.constructor = errorClass;
                    errorClass.prototype.toString = function () {
                        if (this.message === undefined) {
                            return this.name
                        } else {
                            return this.name + ": " + this.message
                        }
                    };
                    return errorClass
                }

                var InternalError = undefined;

                function throwInternalError(message) {
                    throw new InternalError(message)
                }

                function whenDependentTypesAreResolved(myTypes, dependentTypes, getTypeConverters) {
                    myTypes.forEach(function (type) {
                        typeDependencies[type] = dependentTypes
                    });

                    function onComplete(typeConverters) {
                        var myTypeConverters = getTypeConverters(typeConverters);
                        if (myTypeConverters.length !== myTypes.length) {
                            throwInternalError("Mismatched type converter count")
                        }
                        for (var i = 0; i < myTypes.length; ++i) {
                            registerType(myTypes[i], myTypeConverters[i])
                        }
                    }

                    var typeConverters = new Array(dependentTypes.length);
                    var unregisteredTypes = [];
                    var registered = 0;
                    dependentTypes.forEach(function (dt, i) {
                        if (registeredTypes.hasOwnProperty(dt)) {
                            typeConverters[i] = registeredTypes[dt]
                        } else {
                            unregisteredTypes.push(dt);
                            if (!awaitingDependencies.hasOwnProperty(dt)) {
                                awaitingDependencies[dt] = []
                            }
                            awaitingDependencies[dt].push(function () {
                                typeConverters[i] = registeredTypes[dt];
                                ++registered;
                                if (registered === unregisteredTypes.length) {
                                    onComplete(typeConverters)
                                }
                            })
                        }
                    });
                    if (0 === unregisteredTypes.length) {
                        onComplete(typeConverters)
                    }
                }

                function __embind_finalize_value_array(rawTupleType) {
                    var reg = tupleRegistrations[rawTupleType];
                    delete tupleRegistrations[rawTupleType];
                    var elements = reg.elements;
                    var elementsLength = elements.length;
                    var elementTypes = elements.map(function (elt) {
                        return elt.getterReturnType
                    }).concat(elements.map(function (elt) {
                        return elt.setterArgumentType
                    }));
                    var rawConstructor = reg.rawConstructor;
                    var rawDestructor = reg.rawDestructor;
                    whenDependentTypesAreResolved([rawTupleType], elementTypes, function (elementTypes) {
                        elements.forEach(function (elt, i) {
                            var getterReturnType = elementTypes[i];
                            var getter = elt.getter;
                            var getterContext = elt.getterContext;
                            var setterArgumentType = elementTypes[i + elementsLength];
                            var setter = elt.setter;
                            var setterContext = elt.setterContext;
                            elt.read = function (ptr) {
                                return getterReturnType["fromWireType"](getter(getterContext, ptr))
                            };
                            elt.write = function (ptr, o) {
                                var destructors = [];
                                setter(setterContext, ptr, setterArgumentType["toWireType"](destructors, o));
                                runDestructors(destructors)
                            }
                        });
                        return [{
                            name: reg.name,
                            "fromWireType": function (ptr) {
                                var rv = new Array(elementsLength);
                                for (var i = 0; i < elementsLength; ++i) {
                                    rv[i] = elements[i].read(ptr)
                                }
                                rawDestructor(ptr);
                                return rv
                            },
                            "toWireType": function (destructors, o) {
                                if (elementsLength !== o.length) {
                                    throw new TypeError("Incorrect number of tuple elements for " + reg.name + ": expected=" + elementsLength + ", actual=" + o.length)
                                }
                                var ptr = rawConstructor();
                                for (var i = 0; i < elementsLength; ++i) {
                                    elements[i].write(ptr, o[i])
                                }
                                if (destructors !== null) {
                                    destructors.push(rawDestructor, ptr)
                                }
                                return ptr
                            },
                            "argPackAdvance": 8,
                            "readValueFromPointer": simpleReadValueFromPointer,
                            destructorFunction: rawDestructor
                        }]
                    })
                }

                var structRegistrations = {};

                function __embind_finalize_value_object(structType) {
                    var reg = structRegistrations[structType];
                    delete structRegistrations[structType];
                    var rawConstructor = reg.rawConstructor;
                    var rawDestructor = reg.rawDestructor;
                    var fieldRecords = reg.fields;
                    var fieldTypes = fieldRecords.map(function (field) {
                        return field.getterReturnType
                    }).concat(fieldRecords.map(function (field) {
                        return field.setterArgumentType
                    }));
                    whenDependentTypesAreResolved([structType], fieldTypes, function (fieldTypes) {
                        var fields = {};
                        fieldRecords.forEach(function (field, i) {
                            var fieldName = field.fieldName;
                            var getterReturnType = fieldTypes[i];
                            var getter = field.getter;
                            var getterContext = field.getterContext;
                            var setterArgumentType = fieldTypes[i + fieldRecords.length];
                            var setter = field.setter;
                            var setterContext = field.setterContext;
                            fields[fieldName] = {
                                read: function (ptr) {
                                    return getterReturnType["fromWireType"](getter(getterContext, ptr))
                                }, write: function (ptr, o) {
                                    var destructors = [];
                                    setter(setterContext, ptr, setterArgumentType["toWireType"](destructors, o));
                                    runDestructors(destructors)
                                }
                            }
                        });
                        return [{
                            name: reg.name,
                            "fromWireType": function (ptr) {
                                var rv = {};
                                for (var i in fields) {
                                    rv[i] = fields[i].read(ptr)
                                }
                                rawDestructor(ptr);
                                return rv
                            },
                            "toWireType": function (destructors, o) {
                                for (var fieldName in fields) {
                                    if (!(fieldName in o)) {
                                        throw new TypeError('Missing field:  "' + fieldName + '"')
                                    }
                                }
                                var ptr = rawConstructor();
                                for (fieldName in fields) {
                                    fields[fieldName].write(ptr, o[fieldName])
                                }
                                if (destructors !== null) {
                                    destructors.push(rawDestructor, ptr)
                                }
                                return ptr
                            },
                            "argPackAdvance": 8,
                            "readValueFromPointer": simpleReadValueFromPointer,
                            destructorFunction: rawDestructor
                        }]
                    })
                }

                function getShiftFromSize(size) {
                    switch (size) {
                        case 1:
                            return 0;
                        case 2:
                            return 1;
                        case 4:
                            return 2;
                        case 8:
                            return 3;
                        default:
                            throw new TypeError("Unknown type size: " + size)
                    }
                }

                function embind_init_charCodes() {
                    var codes = new Array(256);
                    for (var i = 0; i < 256; ++i) {
                        codes[i] = String.fromCharCode(i)
                    }
                    embind_charCodes = codes
                }

                var embind_charCodes = undefined;

                function readLatin1String(ptr) {
                    var ret = "";
                    var c = ptr;
                    while (HEAPU8[c]) {
                        ret += embind_charCodes[HEAPU8[c++]]
                    }
                    return ret
                }

                var BindingError = undefined;

                function throwBindingError(message) {
                    throw new BindingError(message)
                }

                function registerType(rawType, registeredInstance, options) {
                    options = options || {};
                    if (!("argPackAdvance" in registeredInstance)) {
                        throw new TypeError("registerType registeredInstance requires argPackAdvance")
                    }
                    var name = registeredInstance.name;
                    if (!rawType) {
                        throwBindingError('type "' + name + '" must have a positive integer typeid pointer')
                    }
                    if (registeredTypes.hasOwnProperty(rawType)) {
                        if (options.ignoreDuplicateRegistrations) {
                            return
                        } else {
                            throwBindingError("Cannot register type '" + name + "' twice")
                        }
                    }
                    registeredTypes[rawType] = registeredInstance;
                    delete typeDependencies[rawType];
                    if (awaitingDependencies.hasOwnProperty(rawType)) {
                        var callbacks = awaitingDependencies[rawType];
                        delete awaitingDependencies[rawType];
                        callbacks.forEach(function (cb) {
                            cb()
                        })
                    }
                }

                function __embind_register_bool(rawType, name, size, trueValue, falseValue) {
                    var shift = getShiftFromSize(size);
                    name = readLatin1String(name);
                    registerType(rawType, {
                        name: name, "fromWireType": function (wt) {
                            return !!wt
                        }, "toWireType": function (destructors, o) {
                            return o ? trueValue : falseValue
                        }, "argPackAdvance": 8, "readValueFromPointer": function (pointer) {
                            var heap;
                            if (size === 1) {
                                heap = HEAP8
                            } else if (size === 2) {
                                heap = HEAP16
                            } else if (size === 4) {
                                heap = HEAP32
                            } else {
                                throw new TypeError("Unknown boolean type size: " + name)
                            }
                            return this["fromWireType"](heap[pointer >> shift])
                        }, destructorFunction: null
                    })
                }

                function ClassHandle_isAliasOf(other) {
                    if (!(this instanceof ClassHandle)) {
                        return false
                    }
                    if (!(other instanceof ClassHandle)) {
                        return false
                    }
                    var leftClass = this.$$.ptrType.registeredClass;
                    var left = this.$$.ptr;
                    var rightClass = other.$$.ptrType.registeredClass;
                    var right = other.$$.ptr;
                    while (leftClass.baseClass) {
                        left = leftClass.upcast(left);
                        leftClass = leftClass.baseClass
                    }
                    while (rightClass.baseClass) {
                        right = rightClass.upcast(right);
                        rightClass = rightClass.baseClass
                    }
                    return leftClass === rightClass && left === right
                }

                function shallowCopyInternalPointer(o) {
                    return {
                        count: o.count,
                        deleteScheduled: o.deleteScheduled,
                        preservePointerOnDelete: o.preservePointerOnDelete,
                        ptr: o.ptr,
                        ptrType: o.ptrType,
                        smartPtr: o.smartPtr,
                        smartPtrType: o.smartPtrType
                    }
                }

                function throwInstanceAlreadyDeleted(obj) {
                    function getInstanceTypeName(handle) {
                        return handle.$$.ptrType.registeredClass.name
                    }

                    throwBindingError(getInstanceTypeName(obj) + " instance already deleted")
                }

                var finalizationGroup = false;

                function detachFinalizer(handle) {
                }

                function runDestructor($$) {
                    if ($$.smartPtr) {
                        $$.smartPtrType.rawDestructor($$.smartPtr)
                    } else {
                        $$.ptrType.registeredClass.rawDestructor($$.ptr)
                    }
                }

                function releaseClassHandle($$) {
                    $$.count.value -= 1;
                    var toDelete = 0 === $$.count.value;
                    if (toDelete) {
                        runDestructor($$)
                    }
                }

                function attachFinalizer(handle) {
                    if ("undefined" === typeof FinalizationGroup) {
                        attachFinalizer = function (handle) {
                            return handle
                        };
                        return handle
                    }
                    finalizationGroup = new FinalizationGroup(function (iter) {
                        for (var result = iter.next(); !result.done; result = iter.next()) {
                            var $$ = result.value;
                            if (!$$.ptr) {
                                console.warn("object already deleted: " + $$.ptr)
                            } else {
                                releaseClassHandle($$)
                            }
                        }
                    });
                    attachFinalizer = function (handle) {
                        finalizationGroup.register(handle, handle.$$, handle.$$);
                        return handle
                    };
                    detachFinalizer = function (handle) {
                        finalizationGroup.unregister(handle.$$)
                    };
                    return attachFinalizer(handle)
                }

                function ClassHandle_clone() {
                    if (!this.$$.ptr) {
                        throwInstanceAlreadyDeleted(this)
                    }
                    if (this.$$.preservePointerOnDelete) {
                        this.$$.count.value += 1;
                        return this
                    } else {
                        var clone = attachFinalizer(Object.create(Object.getPrototypeOf(this), {$$: {value: shallowCopyInternalPointer(this.$$)}}));
                        clone.$$.count.value += 1;
                        clone.$$.deleteScheduled = false;
                        return clone
                    }
                }

                function ClassHandle_delete() {
                    if (!this.$$.ptr) {
                        throwInstanceAlreadyDeleted(this)
                    }
                    if (this.$$.deleteScheduled && !this.$$.preservePointerOnDelete) {
                        throwBindingError("Object already scheduled for deletion")
                    }
                    detachFinalizer(this);
                    releaseClassHandle(this.$$);
                    if (!this.$$.preservePointerOnDelete) {
                        this.$$.smartPtr = undefined;
                        this.$$.ptr = undefined
                    }
                }

                function ClassHandle_isDeleted() {
                    return !this.$$.ptr
                }

                var delayFunction = undefined;
                var deletionQueue = [];

                function flushPendingDeletes() {
                    while (deletionQueue.length) {
                        var obj = deletionQueue.pop();
                        obj.$$.deleteScheduled = false;
                        obj["delete"]()
                    }
                }

                function ClassHandle_deleteLater() {
                    if (!this.$$.ptr) {
                        throwInstanceAlreadyDeleted(this)
                    }
                    if (this.$$.deleteScheduled && !this.$$.preservePointerOnDelete) {
                        throwBindingError("Object already scheduled for deletion")
                    }
                    deletionQueue.push(this);
                    if (deletionQueue.length === 1 && delayFunction) {
                        delayFunction(flushPendingDeletes)
                    }
                    this.$$.deleteScheduled = true;
                    return this
                }

                function init_ClassHandle() {
                    ClassHandle.prototype["isAliasOf"] = ClassHandle_isAliasOf;
                    ClassHandle.prototype["clone"] = ClassHandle_clone;
                    ClassHandle.prototype["delete"] = ClassHandle_delete;
                    ClassHandle.prototype["isDeleted"] = ClassHandle_isDeleted;
                    ClassHandle.prototype["deleteLater"] = ClassHandle_deleteLater
                }

                function ClassHandle() {
                }

                var registeredPointers = {};

                function ensureOverloadTable(proto, methodName, humanName) {
                    if (undefined === proto[methodName].overloadTable) {
                        var prevFunc = proto[methodName];
                        proto[methodName] = function () {
                            if (!proto[methodName].overloadTable.hasOwnProperty(arguments.length)) {
                                throwBindingError("Function '" + humanName + "' called with an invalid number of arguments (" + arguments.length + ") - expects one of (" + proto[methodName].overloadTable + ")!")
                            }
                            return proto[methodName].overloadTable[arguments.length].apply(this, arguments)
                        };
                        proto[methodName].overloadTable = [];
                        proto[methodName].overloadTable[prevFunc.argCount] = prevFunc
                    }
                }

                function exposePublicSymbol(name, value, numArguments) {
                    if (Module.hasOwnProperty(name)) {
                        if (undefined === numArguments || undefined !== Module[name].overloadTable && undefined !== Module[name].overloadTable[numArguments]) {
                            throwBindingError("Cannot register public name '" + name + "' twice")
                        }
                        ensureOverloadTable(Module, name, name);
                        if (Module.hasOwnProperty(numArguments)) {
                            throwBindingError("Cannot register multiple overloads of a function with the same number of arguments (" + numArguments + ")!")
                        }
                        Module[name].overloadTable[numArguments] = value
                    } else {
                        Module[name] = value;
                        if (undefined !== numArguments) {
                            Module[name].numArguments = numArguments
                        }
                    }
                }

                function RegisteredClass(name, constructor, instancePrototype, rawDestructor, baseClass, getActualType, upcast, downcast) {
                    this.name = name;
                    this.constructor = constructor;
                    this.instancePrototype = instancePrototype;
                    this.rawDestructor = rawDestructor;
                    this.baseClass = baseClass;
                    this.getActualType = getActualType;
                    this.upcast = upcast;
                    this.downcast = downcast;
                    this.pureVirtualFunctions = []
                }

                function upcastPointer(ptr, ptrClass, desiredClass) {
                    while (ptrClass !== desiredClass) {
                        if (!ptrClass.upcast) {
                            throwBindingError("Expected null or instance of " + desiredClass.name + ", got an instance of " + ptrClass.name)
                        }
                        ptr = ptrClass.upcast(ptr);
                        ptrClass = ptrClass.baseClass
                    }
                    return ptr
                }

                function constNoSmartPtrRawPointerToWireType(destructors, handle) {
                    if (handle === null) {
                        if (this.isReference) {
                            throwBindingError("null is not a valid " + this.name)
                        }
                        return 0
                    }
                    if (!handle.$$) {
                        throwBindingError('Cannot pass "' + _embind_repr(handle) + '" as a ' + this.name)
                    }
                    if (!handle.$$.ptr) {
                        throwBindingError("Cannot pass deleted object as a pointer of type " + this.name)
                    }
                    var handleClass = handle.$$.ptrType.registeredClass;
                    var ptr = upcastPointer(handle.$$.ptr, handleClass, this.registeredClass);
                    return ptr
                }

                function genericPointerToWireType(destructors, handle) {
                    var ptr;
                    if (handle === null) {
                        if (this.isReference) {
                            throwBindingError("null is not a valid " + this.name)
                        }
                        if (this.isSmartPointer) {
                            ptr = this.rawConstructor();
                            if (destructors !== null) {
                                destructors.push(this.rawDestructor, ptr)
                            }
                            return ptr
                        } else {
                            return 0
                        }
                    }
                    if (!handle.$$) {
                        throwBindingError('Cannot pass "' + _embind_repr(handle) + '" as a ' + this.name)
                    }
                    if (!handle.$$.ptr) {
                        throwBindingError("Cannot pass deleted object as a pointer of type " + this.name)
                    }
                    if (!this.isConst && handle.$$.ptrType.isConst) {
                        throwBindingError("Cannot convert argument of type " + (handle.$$.smartPtrType ? handle.$$.smartPtrType.name : handle.$$.ptrType.name) + " to parameter type " + this.name)
                    }
                    var handleClass = handle.$$.ptrType.registeredClass;
                    ptr = upcastPointer(handle.$$.ptr, handleClass, this.registeredClass);
                    if (this.isSmartPointer) {
                        if (undefined === handle.$$.smartPtr) {
                            throwBindingError("Passing raw pointer to smart pointer is illegal")
                        }
                        switch (this.sharingPolicy) {
                            case 0:
                                if (handle.$$.smartPtrType === this) {
                                    ptr = handle.$$.smartPtr
                                } else {
                                    throwBindingError("Cannot convert argument of type " + (handle.$$.smartPtrType ? handle.$$.smartPtrType.name : handle.$$.ptrType.name) + " to parameter type " + this.name)
                                }
                                break;
                            case 1:
                                ptr = handle.$$.smartPtr;
                                break;
                            case 2:
                                if (handle.$$.smartPtrType === this) {
                                    ptr = handle.$$.smartPtr
                                } else {
                                    var clonedHandle = handle["clone"]();
                                    ptr = this.rawShare(ptr, __emval_register(function () {
                                        clonedHandle["delete"]()
                                    }));
                                    if (destructors !== null) {
                                        destructors.push(this.rawDestructor, ptr)
                                    }
                                }
                                break;
                            default:
                                throwBindingError("Unsupporting sharing policy")
                        }
                    }
                    return ptr
                }

                function nonConstNoSmartPtrRawPointerToWireType(destructors, handle) {
                    if (handle === null) {
                        if (this.isReference) {
                            throwBindingError("null is not a valid " + this.name)
                        }
                        return 0
                    }
                    if (!handle.$$) {
                        throwBindingError('Cannot pass "' + _embind_repr(handle) + '" as a ' + this.name)
                    }
                    if (!handle.$$.ptr) {
                        throwBindingError("Cannot pass deleted object as a pointer of type " + this.name)
                    }
                    if (handle.$$.ptrType.isConst) {
                        throwBindingError("Cannot convert argument of type " + handle.$$.ptrType.name + " to parameter type " + this.name)
                    }
                    var handleClass = handle.$$.ptrType.registeredClass;
                    var ptr = upcastPointer(handle.$$.ptr, handleClass, this.registeredClass);
                    return ptr
                }

                function RegisteredPointer_getPointee(ptr) {
                    if (this.rawGetPointee) {
                        ptr = this.rawGetPointee(ptr)
                    }
                    return ptr
                }

                function RegisteredPointer_destructor(ptr) {
                    if (this.rawDestructor) {
                        this.rawDestructor(ptr)
                    }
                }

                function RegisteredPointer_deleteObject(handle) {
                    if (handle !== null) {
                        handle["delete"]()
                    }
                }

                function downcastPointer(ptr, ptrClass, desiredClass) {
                    if (ptrClass === desiredClass) {
                        return ptr
                    }
                    if (undefined === desiredClass.baseClass) {
                        return null
                    }
                    var rv = downcastPointer(ptr, ptrClass, desiredClass.baseClass);
                    if (rv === null) {
                        return null
                    }
                    return desiredClass.downcast(rv)
                }

                function getInheritedInstanceCount() {
                    return Object.keys(registeredInstances).length
                }

                function getLiveInheritedInstances() {
                    var rv = [];
                    for (var k in registeredInstances) {
                        if (registeredInstances.hasOwnProperty(k)) {
                            rv.push(registeredInstances[k])
                        }
                    }
                    return rv
                }

                function setDelayFunction(fn) {
                    delayFunction = fn;
                    if (deletionQueue.length && delayFunction) {
                        delayFunction(flushPendingDeletes)
                    }
                }

                function init_embind() {
                    Module["getInheritedInstanceCount"] = getInheritedInstanceCount;
                    Module["getLiveInheritedInstances"] = getLiveInheritedInstances;
                    Module["flushPendingDeletes"] = flushPendingDeletes;
                    Module["setDelayFunction"] = setDelayFunction
                }

                var registeredInstances = {};

                function getBasestPointer(class_, ptr) {
                    if (ptr === undefined) {
                        throwBindingError("ptr should not be undefined")
                    }
                    while (class_.baseClass) {
                        ptr = class_.upcast(ptr);
                        class_ = class_.baseClass
                    }
                    return ptr
                }

                function getInheritedInstance(class_, ptr) {
                    ptr = getBasestPointer(class_, ptr);
                    return registeredInstances[ptr]
                }

                function makeClassHandle(prototype, record) {
                    if (!record.ptrType || !record.ptr) {
                        throwInternalError("makeClassHandle requires ptr and ptrType")
                    }
                    var hasSmartPtrType = !!record.smartPtrType;
                    var hasSmartPtr = !!record.smartPtr;
                    if (hasSmartPtrType !== hasSmartPtr) {
                        throwInternalError("Both smartPtrType and smartPtr must be specified")
                    }
                    record.count = {value: 1};
                    return attachFinalizer(Object.create(prototype, {$$: {value: record}}))
                }

                function RegisteredPointer_fromWireType(ptr) {
                    var rawPointer = this.getPointee(ptr);
                    if (!rawPointer) {
                        this.destructor(ptr);
                        return null
                    }
                    var registeredInstance = getInheritedInstance(this.registeredClass, rawPointer);
                    if (undefined !== registeredInstance) {
                        if (0 === registeredInstance.$$.count.value) {
                            registeredInstance.$$.ptr = rawPointer;
                            registeredInstance.$$.smartPtr = ptr;
                            return registeredInstance["clone"]()
                        } else {
                            var rv = registeredInstance["clone"]();
                            this.destructor(ptr);
                            return rv
                        }
                    }

                    function makeDefaultHandle() {
                        if (this.isSmartPointer) {
                            return makeClassHandle(this.registeredClass.instancePrototype, {
                                ptrType: this.pointeeType,
                                ptr: rawPointer,
                                smartPtrType: this,
                                smartPtr: ptr
                            })
                        } else {
                            return makeClassHandle(this.registeredClass.instancePrototype, {ptrType: this, ptr: ptr})
                        }
                    }

                    var actualType = this.registeredClass.getActualType(rawPointer);
                    var registeredPointerRecord = registeredPointers[actualType];
                    if (!registeredPointerRecord) {
                        return makeDefaultHandle.call(this)
                    }
                    var toType;
                    if (this.isConst) {
                        toType = registeredPointerRecord.constPointerType
                    } else {
                        toType = registeredPointerRecord.pointerType
                    }
                    var dp = downcastPointer(rawPointer, this.registeredClass, toType.registeredClass);
                    if (dp === null) {
                        return makeDefaultHandle.call(this)
                    }
                    if (this.isSmartPointer) {
                        return makeClassHandle(toType.registeredClass.instancePrototype, {
                            ptrType: toType,
                            ptr: dp,
                            smartPtrType: this,
                            smartPtr: ptr
                        })
                    } else {
                        return makeClassHandle(toType.registeredClass.instancePrototype, {ptrType: toType, ptr: dp})
                    }
                }

                function init_RegisteredPointer() {
                    RegisteredPointer.prototype.getPointee = RegisteredPointer_getPointee;
                    RegisteredPointer.prototype.destructor = RegisteredPointer_destructor;
                    RegisteredPointer.prototype["argPackAdvance"] = 8;
                    RegisteredPointer.prototype["readValueFromPointer"] = simpleReadValueFromPointer;
                    RegisteredPointer.prototype["deleteObject"] = RegisteredPointer_deleteObject;
                    RegisteredPointer.prototype["fromWireType"] = RegisteredPointer_fromWireType
                }

                function RegisteredPointer(name, registeredClass, isReference, isConst, isSmartPointer, pointeeType, sharingPolicy, rawGetPointee, rawConstructor, rawShare, rawDestructor) {
                    this.name = name;
                    this.registeredClass = registeredClass;
                    this.isReference = isReference;
                    this.isConst = isConst;
                    this.isSmartPointer = isSmartPointer;
                    this.pointeeType = pointeeType;
                    this.sharingPolicy = sharingPolicy;
                    this.rawGetPointee = rawGetPointee;
                    this.rawConstructor = rawConstructor;
                    this.rawShare = rawShare;
                    this.rawDestructor = rawDestructor;
                    if (!isSmartPointer && registeredClass.baseClass === undefined) {
                        if (isConst) {
                            this["toWireType"] = constNoSmartPtrRawPointerToWireType;
                            this.destructorFunction = null
                        } else {
                            this["toWireType"] = nonConstNoSmartPtrRawPointerToWireType;
                            this.destructorFunction = null
                        }
                    } else {
                        this["toWireType"] = genericPointerToWireType
                    }
                }

                function replacePublicSymbol(name, value, numArguments) {
                    if (!Module.hasOwnProperty(name)) {
                        throwInternalError("Replacing nonexistant public symbol")
                    }
                    if (undefined !== Module[name].overloadTable && undefined !== numArguments) {
                        Module[name].overloadTable[numArguments] = value
                    } else {
                        Module[name] = value;
                        Module[name].argCount = numArguments
                    }
                }

                function embind__requireFunction(signature, rawFunction) {
                    signature = readLatin1String(signature);

                    function makeDynCaller(dynCall) {
                        if (IsWechat) {
                            var f1 = function (dynCall, rawFunction) {
                                return function () {
                                    return dynCall(rawFunction, ...arguments);
                                }
                            }
                            return f1(dynCall, rawFunction)

                        } else {
                            var args = [];
                            for (var i = 1; i < signature.length; ++i) {
                                args.push("a" + i)
                            }
                            var name = "dynCall_" + signature + "_" + rawFunction;
                            var body = "return function " + name + "(" + args.join(", ") + ") {\n";
                            body += "    return dynCall(rawFunction" + (args.length ? ", " : "") + args.join(", ") + ");\n";
                            body += "};\n";

                            return new Function("dynCall", "rawFunction", body)(dynCall, rawFunction)
                        }
                    }

                    var dc = Module["dynCall_" + signature];
                    var fp = makeDynCaller(dc);
                    if (typeof fp !== "function") {
                        throwBindingError("unknown function pointer with signature " + signature + ": " + rawFunction)
                    }
                    return fp
                }

                var UnboundTypeError = undefined;

                function getTypeName(type) {
                    var ptr = ___getTypeName(type);
                    var rv = readLatin1String(ptr);
                    _free(ptr);
                    return rv
                }

                function throwUnboundTypeError(message, types) {
                    var unboundTypes = [];
                    var seen = {};

                    function visit(type) {
                        if (seen[type]) {
                            return
                        }
                        if (registeredTypes[type]) {
                            return
                        }
                        if (typeDependencies[type]) {
                            typeDependencies[type].forEach(visit);
                            return
                        }
                        unboundTypes.push(type);
                        seen[type] = true
                    }

                    types.forEach(visit);
                    throw new UnboundTypeError(message + ": " + unboundTypes.map(getTypeName).join([", "]))
                }

                function __embind_register_class(rawType, rawPointerType, rawConstPointerType, baseClassRawType, getActualTypeSignature, getActualType, upcastSignature, upcast, downcastSignature, downcast, name, destructorSignature, rawDestructor) {
                    name = readLatin1String(name);
                    getActualType = embind__requireFunction(getActualTypeSignature, getActualType);
                    if (upcast) {
                        upcast = embind__requireFunction(upcastSignature, upcast)
                    }
                    if (downcast) {
                        downcast = embind__requireFunction(downcastSignature, downcast)
                    }
                    rawDestructor = embind__requireFunction(destructorSignature, rawDestructor);
                    var legalFunctionName = makeLegalFunctionName(name);
                    exposePublicSymbol(legalFunctionName, function () {
                        throwUnboundTypeError("Cannot construct " + name + " due to unbound types", [baseClassRawType])
                    });
                    whenDependentTypesAreResolved([rawType, rawPointerType, rawConstPointerType], baseClassRawType ? [baseClassRawType] : [], function (base) {
                        base = base[0];
                        var baseClass;
                        var basePrototype;
                        if (baseClassRawType) {
                            baseClass = base.registeredClass;
                            basePrototype = baseClass.instancePrototype
                        } else {
                            basePrototype = ClassHandle.prototype
                        }
                        var constructor = createNamedFunction(legalFunctionName, function () {
                            if (Object.getPrototypeOf(this) !== instancePrototype) {
                                throw new BindingError("Use 'new' to construct " + name)
                            }
                            if (undefined === registeredClass.constructor_body) {
                                throw new BindingError(name + " has no accessible constructor")
                            }
                            var body = registeredClass.constructor_body[arguments.length];
                            if (undefined === body) {
                                throw new BindingError("Tried to invoke ctor of " + name + " with invalid number of parameters (" + arguments.length + ") - expected (" + Object.keys(registeredClass.constructor_body).toString() + ") parameters instead!")
                            }
                            return body.apply(this, arguments)
                        });
                        var instancePrototype = Object.create(basePrototype, {constructor: {value: constructor}});
                        constructor.prototype = instancePrototype;
                        var registeredClass = new RegisteredClass(name, constructor, instancePrototype, rawDestructor, baseClass, getActualType, upcast, downcast);
                        var referenceConverter = new RegisteredPointer(name, registeredClass, true, false, false);
                        var pointerConverter = new RegisteredPointer(name + "*", registeredClass, false, false, false);
                        var constPointerConverter = new RegisteredPointer(name + " const*", registeredClass, false, true, false);
                        registeredPointers[rawType] = {
                            pointerType: pointerConverter,
                            constPointerType: constPointerConverter
                        };
                        replacePublicSymbol(legalFunctionName, constructor);
                        return [referenceConverter, pointerConverter, constPointerConverter]
                    })
                }

                function new_(constructor, argumentList) {
                    if (!(constructor instanceof Function)) {
                        throw new TypeError("new_ called with constructor type " + typeof constructor + " which is not a function")
                    }
                    var dummy = createNamedFunction(constructor.name || "unknownFunctionName", function () {
                    });
                    dummy.prototype = constructor.prototype;
                    var obj = new dummy;
                    var r = constructor.apply(obj, argumentList);
                    return r instanceof Object ? r : obj
                }

                function craftInvokerFunction(humanName, argTypes, classType, cppInvokerFunc, cppTargetFunc) {
                    var argCount = argTypes.length;
                    if (argCount < 2) {
                        throwBindingError("argTypes array size mismatch! Must at least get return value and 'this' types!")
                    }
                    var isClassMethodFunc = argTypes[1] !== null && classType !== null;
                    var needsDestructorStack = false;
                    for (var i = 1; i < argTypes.length; ++i) {
                        if (argTypes[i] !== null && argTypes[i].destructorFunction === undefined) {
                            needsDestructorStack = true;
                            break
                        }
                    }
                    var returns = argTypes[0].name !== "void";
                    if (IsWechat) {
                        var args2 = [throwBindingError, cppInvokerFunc, cppTargetFunc, runDestructors, argTypes[0], argTypes[1]];
                        for (var i = 0; i < argCount - 2; ++i) {
                            args2.push(argTypes[i + 2])
                        }
                        if (isClassMethodFunc) {
                            for (var i = 1; i < argCount; ++i) {
                                if (argTypes[i].destructorFunction !== null) {
                                    args2.push(argTypes[i].destructorFunction)
                                }else{
                                    args2.push(null)
                                }
                            }
                        } else {
                            for (var i = 2; i < argCount; ++i) {
                                if (argTypes[i].destructorFunction !== null) {
                                    args2.push(argTypes[i].destructorFunction)
                                }else{
                                    args2.push(null)
                                }
                            }
                        }

                        function f1(throwBindingError, invoker, fn, runDestructors, retType, classParam) {
                            // argType0,argType1,argType2
                            const argsTypeOrigin = Array.prototype.slice.call(arguments, 6, 6 + argCount - 2)
                            // arg0Wired_dtor
                            const argsWired_dtorOrigin = Array.prototype.slice.call(arguments, 6 + argCount - 2)

                            return function () {
                                // arg0, arg1, arg2
                                if (arguments.length !== argCount - 2) {
                                    throwBindingError('function ' + humanName + ' called with ' + arguments.length + ' arguments, expected 0 args!');
                                }
                                var thisWired;
                                if (isClassMethodFunc) {
                                    if (needsDestructorStack) {
                                        var destructors = [];
                                        thisWired = classParam.toWireType(destructors, this);
                                    } else {
                                        thisWired = classParam.toWireType(null, this);
                                    }
                                }

                                // arg0Wired,arg1Wired,arg2Wired
                                var argsWired = [];
                                for (var i = 0; i < arguments.length; i++) {
                                    argsWired.push(argsTypeOrigin[i].toWireType(null, arguments[i]))
                                }

                                var rv;
                                if (isClassMethodFunc) {
                                    rv = invoker(fn, thisWired, ...argsWired);
                                } else {
                                    rv = invoker(fn, ...argsWired);
                                }

                                if (needsDestructorStack) {
                                    runDestructors(destructors);
                                } else {
                                    if (isClassMethodFunc) {
                                        for (var i = 1; i < argTypes.length; ++i) {
                                            if (argTypes[i].destructorFunction !== null) {
                                                argsWired_dtorOrigin[i - 1](thisWired);
                                            }
                                        }
                                    } else {
                                        for (var i = 2; i < argTypes.length; ++i) {
                                            if (argTypes[i].destructorFunction !== null) {
                                                argsWired_dtorOrigin[i - 2](argsWired[i - 2]);
                                            }
                                        }
                                    }
                                }
                                if (returns) {
                                    var ret = retType.fromWireType(rv);
                                    return ret;
                                }
                            }
                        }
                        return f1.apply(null, args2)
                    } else {
                        var argsList = "";
                        var argsListWired = "";
                        for (var i = 0; i < argCount - 2; ++i) {
                            argsList += (i !== 0 ? ", " : "") + "arg" + i;
                            argsListWired += (i !== 0 ? ", " : "") + "arg" + i + "Wired"
                        }

                        var invokerFnBody = "return function " + makeLegalFunctionName(humanName) + "(" + argsList + ") {\n" + "if (arguments.length !== " + (argCount - 2) + ") {\n" + "throwBindingError('function " + humanName + " called with ' + arguments.length + ' arguments, expected " + (argCount - 2) + " args!');\n" + "}\n";
                        if (needsDestructorStack) {
                            invokerFnBody += "var destructors = [];\n"
                        }
                        var dtorStack = needsDestructorStack ? "destructors" : "null";
                        var args1 = ["throwBindingError", "invoker", "fn", "runDestructors", "retType", "classParam"];
                        var args2 = [throwBindingError, cppInvokerFunc, cppTargetFunc, runDestructors, argTypes[0], argTypes[1]];
                        if (isClassMethodFunc) {
                            invokerFnBody += "var thisWired = classParam.toWireType(" + dtorStack + ", this);\n"
                        }
                        for (var i = 0; i < argCount - 2; ++i) {
                            invokerFnBody += "var arg" + i + "Wired = argType" + i + ".toWireType(" + dtorStack + ", arg" + i + "); // " + argTypes[i + 2].name + "\n";
                            args1.push("argType" + i);
                            args2.push(argTypes[i + 2])
                        }
                        if (isClassMethodFunc) {
                            argsListWired = "thisWired" + (argsListWired.length > 0 ? ", " : "") + argsListWired
                        }
                        invokerFnBody += (returns ? "var rv = " : "") + "invoker(fn" + (argsListWired.length > 0 ? ", " : "") + argsListWired + ");\n";
                        if (needsDestructorStack) {
                            invokerFnBody += "runDestructors(destructors);\n"
                        } else {
                            for (var i = isClassMethodFunc ? 1 : 2; i < argTypes.length; ++i) {
                                var paramName = i === 1 ? "thisWired" : "arg" + (i - 2) + "Wired";
                                if (argTypes[i].destructorFunction !== null) {
                                    invokerFnBody += paramName + "_dtor(" + paramName + "); // " + argTypes[i].name + "\n";
                                    args1.push(paramName + "_dtor");
                                    args2.push(argTypes[i].destructorFunction)
                                }
                            }
                        }
                        if (returns) {
                            invokerFnBody += "var ret = retType.fromWireType(rv);\n" + "return ret;\n"
                        } else { }
                        invokerFnBody += "}\n";
                        args1.push(invokerFnBody);
                        var invokerFunction = new_(Function, args1).apply(null, args2);
                        return invokerFunction
                    }
                }

                function heap32VectorToArray(count, firstElement) {
                    var array = [];
                    for (var i = 0; i < count; i++) {
                        array.push(HEAP32[(firstElement >> 2) + i])
                    }
                    return array
                }

                function __embind_register_class_class_function(rawClassType, methodName, argCount, rawArgTypesAddr, invokerSignature, rawInvoker, fn) {
                    var rawArgTypes = heap32VectorToArray(argCount, rawArgTypesAddr);
                    methodName = readLatin1String(methodName);
                    rawInvoker = embind__requireFunction(invokerSignature, rawInvoker);
                    whenDependentTypesAreResolved([], [rawClassType], function (classType) {
                        classType = classType[0];
                        var humanName = classType.name + "." + methodName;

                        function unboundTypesHandler() {
                            throwUnboundTypeError("Cannot call " + humanName + " due to unbound types", rawArgTypes)
                        }

                        var proto = classType.registeredClass.constructor;
                        if (undefined === proto[methodName]) {
                            unboundTypesHandler.argCount = argCount - 1;
                            proto[methodName] = unboundTypesHandler
                        } else {
                            ensureOverloadTable(proto, methodName, humanName);
                            proto[methodName].overloadTable[argCount - 1] = unboundTypesHandler
                        }
                        whenDependentTypesAreResolved([], rawArgTypes, function (argTypes) {
                            var invokerArgsArray = [argTypes[0], null].concat(argTypes.slice(1));
                            var func = craftInvokerFunction(humanName, invokerArgsArray, null, rawInvoker, fn);
                            if (undefined === proto[methodName].overloadTable) {
                                func.argCount = argCount - 1;
                                proto[methodName] = func
                            } else {
                                proto[methodName].overloadTable[argCount - 1] = func
                            }
                            return []
                        });
                        return []
                    })
                }

                function __embind_register_class_constructor(rawClassType, argCount, rawArgTypesAddr, invokerSignature, invoker, rawConstructor) {
                    assert(argCount > 0);
                    var rawArgTypes = heap32VectorToArray(argCount, rawArgTypesAddr);
                    invoker = embind__requireFunction(invokerSignature, invoker);
                    var args = [rawConstructor];
                    var destructors = [];
                    whenDependentTypesAreResolved([], [rawClassType], function (classType) {
                        classType = classType[0];
                        var humanName = "constructor " + classType.name;
                        if (undefined === classType.registeredClass.constructor_body) {
                            classType.registeredClass.constructor_body = []
                        }
                        if (undefined !== classType.registeredClass.constructor_body[argCount - 1]) {
                            throw new BindingError("Cannot register multiple constructors with identical number of parameters (" + (argCount - 1) + ") for class '" + classType.name + "'! Overload resolution is currently only performed using the parameter count, not actual type info!")
                        }
                        classType.registeredClass.constructor_body[argCount - 1] = function unboundTypeHandler() {
                            throwUnboundTypeError("Cannot construct " + classType.name + " due to unbound types", rawArgTypes)
                        };
                        whenDependentTypesAreResolved([], rawArgTypes, function (argTypes) {
                            classType.registeredClass.constructor_body[argCount - 1] = function constructor_body() {
                                if (arguments.length !== argCount - 1) {
                                    throwBindingError(humanName + " called with " + arguments.length + " arguments, expected " + (argCount - 1))
                                }
                                destructors.length = 0;
                                args.length = argCount;
                                for (var i = 1; i < argCount; ++i) {
                                    args[i] = argTypes[i]["toWireType"](destructors, arguments[i - 1])
                                }
                                var ptr = invoker.apply(null, args);
                                runDestructors(destructors);
                                return argTypes[0]["fromWireType"](ptr)
                            };
                            return []
                        });
                        return []
                    })
                }

                function __embind_register_class_function(rawClassType, methodName, argCount, rawArgTypesAddr, invokerSignature, rawInvoker, context, isPureVirtual) {
                    var rawArgTypes = heap32VectorToArray(argCount, rawArgTypesAddr);
                    methodName = readLatin1String(methodName);
                    rawInvoker = embind__requireFunction(invokerSignature, rawInvoker);
                    whenDependentTypesAreResolved([], [rawClassType], function (classType) {
                        classType = classType[0];
                        var humanName = classType.name + "." + methodName;
                        if (isPureVirtual) {
                            classType.registeredClass.pureVirtualFunctions.push(methodName)
                        }

                        function unboundTypesHandler() {
                            throwUnboundTypeError("Cannot call " + humanName + " due to unbound types", rawArgTypes)
                        }

                        var proto = classType.registeredClass.instancePrototype;
                        var method = proto[methodName];
                        if (undefined === method || undefined === method.overloadTable && method.className !== classType.name && method.argCount === argCount - 2) {
                            unboundTypesHandler.argCount = argCount - 2;
                            unboundTypesHandler.className = classType.name;
                            proto[methodName] = unboundTypesHandler
                        } else {
                            ensureOverloadTable(proto, methodName, humanName);
                            proto[methodName].overloadTable[argCount - 2] = unboundTypesHandler
                        }
                        whenDependentTypesAreResolved([], rawArgTypes, function (argTypes) {
                            var memberFunction = craftInvokerFunction(humanName, argTypes, classType, rawInvoker, context);
                            if (undefined === proto[methodName].overloadTable) {
                                memberFunction.argCount = argCount - 2;
                                proto[methodName] = memberFunction
                            } else {
                                proto[methodName].overloadTable[argCount - 2] = memberFunction
                            }
                            return []
                        });
                        return []
                    })
                }

                function validateThis(this_, classType, humanName) {
                    if (!(this_ instanceof Object)) {
                        throwBindingError(humanName + ' with invalid "this": ' + this_)
                    }
                    if (!(this_ instanceof classType.registeredClass.constructor)) {
                        throwBindingError(humanName + ' incompatible with "this" of type ' + this_.constructor.name)
                    }
                    if (!this_.$$.ptr) {
                        throwBindingError("cannot call emscripten binding method " + humanName + " on deleted object")
                    }
                    return upcastPointer(this_.$$.ptr, this_.$$.ptrType.registeredClass, classType.registeredClass)
                }

                function __embind_register_class_property(classType, fieldName, getterReturnType, getterSignature, getter, getterContext, setterArgumentType, setterSignature, setter, setterContext) {
                    fieldName = readLatin1String(fieldName);
                    getter = embind__requireFunction(getterSignature, getter);
                    whenDependentTypesAreResolved([], [classType], function (classType) {
                        classType = classType[0];
                        var humanName = classType.name + "." + fieldName;
                        var desc = {
                            get: function () {
                                throwUnboundTypeError("Cannot access " + humanName + " due to unbound types", [getterReturnType, setterArgumentType])
                            }, enumerable: true, configurable: true
                        };
                        if (setter) {
                            desc.set = function () {
                                throwUnboundTypeError("Cannot access " + humanName + " due to unbound types", [getterReturnType, setterArgumentType])
                            }
                        } else {
                            desc.set = function (v) {
                                throwBindingError(humanName + " is a read-only property")
                            }
                        }
                        Object.defineProperty(classType.registeredClass.instancePrototype, fieldName, desc);
                        whenDependentTypesAreResolved([], setter ? [getterReturnType, setterArgumentType] : [getterReturnType], function (types) {
                            var getterReturnType = types[0];
                            var desc = {
                                get: function () {
                                    var ptr = validateThis(this, classType, humanName + " getter");
                                    return getterReturnType["fromWireType"](getter(getterContext, ptr))
                                }, enumerable: true
                            };
                            if (setter) {
                                setter = embind__requireFunction(setterSignature, setter);
                                var setterArgumentType = types[1];
                                desc.set = function (v) {
                                    var ptr = validateThis(this, classType, humanName + " setter");
                                    var destructors = [];
                                    setter(setterContext, ptr, setterArgumentType["toWireType"](destructors, v));
                                    runDestructors(destructors)
                                }
                            }
                            Object.defineProperty(classType.registeredClass.instancePrototype, fieldName, desc);
                            return []
                        });
                        return []
                    })
                }

                function __embind_register_constant(name, type, value) {
                    name = readLatin1String(name);
                    whenDependentTypesAreResolved([], [type], function (type) {
                        type = type[0];
                        Module[name] = type["fromWireType"](value);
                        return []
                    })
                }

                var emval_free_list = [];
                var emval_handle_array = [{}, {value: undefined}, {value: null}, {value: true}, {value: false}];

                function __emval_decref(handle) {
                    if (handle > 4 && 0 === --emval_handle_array[handle].refcount) {
                        emval_handle_array[handle] = undefined;
                        emval_free_list.push(handle)
                    }
                }

                function count_emval_handles() {
                    var count = 0;
                    for (var i = 5; i < emval_handle_array.length; ++i) {
                        if (emval_handle_array[i] !== undefined) {
                            ++count
                        }
                    }
                    return count
                }

                function get_first_emval() {
                    for (var i = 5; i < emval_handle_array.length; ++i) {
                        if (emval_handle_array[i] !== undefined) {
                            return emval_handle_array[i]
                        }
                    }
                    return null
                }

                function init_emval() {
                    Module["count_emval_handles"] = count_emval_handles;
                    Module["get_first_emval"] = get_first_emval
                }

                function __emval_register(value) {
                    switch (value) {
                        case undefined: {
                            return 1
                        }
                        case null: {
                            return 2
                        }
                        case true: {
                            return 3
                        }
                        case false: {
                            return 4
                        }
                        default: {
                            var handle = emval_free_list.length ? emval_free_list.pop() : emval_handle_array.length;
                            emval_handle_array[handle] = {refcount: 1, value: value};
                            return handle
                        }
                    }
                }

                function __embind_register_emval(rawType, name) {
                    name = readLatin1String(name);
                    registerType(rawType, {
                        name: name,
                        "fromWireType": function (handle) {
                            var rv = emval_handle_array[handle].value;
                            __emval_decref(handle);
                            return rv
                        },
                        "toWireType": function (destructors, value) {
                            return __emval_register(value)
                        },
                        "argPackAdvance": 8,
                        "readValueFromPointer": simpleReadValueFromPointer,
                        destructorFunction: null
                    })
                }

                function _embind_repr(v) {
                    if (v === null) {
                        return "null"
                    }
                    var t = typeof v;
                    if (t === "object" || t === "array" || t === "function") {
                        return v.toString()
                    } else {
                        return "" + v
                    }
                }

                function floatReadValueFromPointer(name, shift) {
                    switch (shift) {
                        case 2:
                            return function (pointer) {
                                return this["fromWireType"](HEAPF32[pointer >> 2])
                            };
                        case 3:
                            return function (pointer) {
                                return this["fromWireType"](HEAPF64[pointer >> 3])
                            };
                        default:
                            throw new TypeError("Unknown float type: " + name)
                    }
                }

                function __embind_register_float(rawType, name, size) {
                    var shift = getShiftFromSize(size);
                    name = readLatin1String(name);
                    registerType(rawType, {
                        name: name,
                        "fromWireType": function (value) {
                            return value
                        },
                        "toWireType": function (destructors, value) {
                            if (typeof value !== "number" && typeof value !== "boolean") {
                                throw new TypeError('Cannot convert "' + _embind_repr(value) + '" to ' + this.name)
                            }
                            return value
                        },
                        "argPackAdvance": 8,
                        "readValueFromPointer": floatReadValueFromPointer(name, shift),
                        destructorFunction: null
                    })
                }

                function __embind_register_function(name, argCount, rawArgTypesAddr, signature, rawInvoker, fn) {
                    var argTypes = heap32VectorToArray(argCount, rawArgTypesAddr);
                    name = readLatin1String(name);
                    rawInvoker = embind__requireFunction(signature, rawInvoker);
                    exposePublicSymbol(name, function () {
                        throwUnboundTypeError("Cannot call " + name + " due to unbound types", argTypes)
                    }, argCount - 1);
                    whenDependentTypesAreResolved([], argTypes, function (argTypes) {
                        var invokerArgsArray = [argTypes[0], null].concat(argTypes.slice(1));
                        replacePublicSymbol(name, craftInvokerFunction(name, invokerArgsArray, null, rawInvoker, fn), argCount - 1);
                        return []
                    })
                }

                function integerReadValueFromPointer(name, shift, signed) {
                    switch (shift) {
                        case 0:
                            return signed ? function readS8FromPointer(pointer) {
                                return HEAP8[pointer]
                            } : function readU8FromPointer(pointer) {
                                return HEAPU8[pointer]
                            };
                        case 1:
                            return signed ? function readS16FromPointer(pointer) {
                                return HEAP16[pointer >> 1]
                            } : function readU16FromPointer(pointer) {
                                return HEAPU16[pointer >> 1]
                            };
                        case 2:
                            return signed ? function readS32FromPointer(pointer) {
                                return HEAP32[pointer >> 2]
                            } : function readU32FromPointer(pointer) {
                                return HEAPU32[pointer >> 2]
                            };
                        default:
                            throw new TypeError("Unknown integer type: " + name)
                    }
                }

                function __embind_register_integer(primitiveType, name, size, minRange, maxRange) {
                    name = readLatin1String(name);
                    if (maxRange === -1) {
                        maxRange = 4294967295
                    }
                    var shift = getShiftFromSize(size);
                    var fromWireType = function (value) {
                        return value
                    };
                    if (minRange === 0) {
                        var bitshift = 32 - 8 * size;
                        fromWireType = function (value) {
                            return value << bitshift >>> bitshift
                        }
                    }
                    var isUnsignedType = name.indexOf("unsigned") != -1;
                    registerType(primitiveType, {
                        name: name,
                        "fromWireType": fromWireType,
                        "toWireType": function (destructors, value) {
                            if (typeof value !== "number" && typeof value !== "boolean") {
                                throw new TypeError('Cannot convert "' + _embind_repr(value) + '" to ' + this.name)
                            }
                            if (value < minRange || value > maxRange) {
                                throw new TypeError('Passing a number "' + _embind_repr(value) + '" from JS side to C/C++ side to an argument of type "' + name + '", which is outside the valid range [' + minRange + ", " + maxRange + "]!")
                            }
                            return isUnsignedType ? value >>> 0 : value | 0
                        },
                        "argPackAdvance": 8,
                        "readValueFromPointer": integerReadValueFromPointer(name, shift, minRange !== 0),
                        destructorFunction: null
                    })
                }

                function __embind_register_memory_view(rawType, dataTypeIndex, name) {
                    var typeMapping = [Int8Array, Uint8Array, Int16Array, Uint16Array, Int32Array, Uint32Array, Float32Array, Float64Array];
                    var TA = typeMapping[dataTypeIndex];

                    function decodeMemoryView(handle) {
                        handle = handle >> 2;
                        var heap = HEAPU32;
                        var size = heap[handle];
                        var data = heap[handle + 1];
                        return new TA(buffer, data, size)
                    }

                    name = readLatin1String(name);
                    registerType(rawType, {
                        name: name,
                        "fromWireType": decodeMemoryView,
                        "argPackAdvance": 8,
                        "readValueFromPointer": decodeMemoryView
                    }, {ignoreDuplicateRegistrations: true})
                }

                function __embind_register_std_string(rawType, name) {
                    name = readLatin1String(name);
                    var stdStringIsUTF8 = name === "std::string";
                    registerType(rawType, {
                        name: name,
                        "fromWireType": function (value) {
                            var length = HEAPU32[value >> 2];
                            var str;
                            if (stdStringIsUTF8) {
                                var decodeStartPtr = value + 4;
                                for (var i = 0; i <= length; ++i) {
                                    var currentBytePtr = value + 4 + i;
                                    if (i == length || HEAPU8[currentBytePtr] == 0) {
                                        var maxRead = currentBytePtr - decodeStartPtr;
                                        var stringSegment = UTF8ToString(decodeStartPtr, maxRead);
                                        if (str === undefined) {
                                            str = stringSegment
                                        } else {
                                            str += String.fromCharCode(0);
                                            str += stringSegment
                                        }
                                        decodeStartPtr = currentBytePtr + 1
                                    }
                                }
                            } else {
                                var a = new Array(length);
                                for (var i = 0; i < length; ++i) {
                                    a[i] = String.fromCharCode(HEAPU8[value + 4 + i])
                                }
                                str = a.join("")
                            }
                            _free(value);
                            return str
                        },
                        "toWireType": function (destructors, value) {
                            if (value instanceof ArrayBuffer) {
                                value = new Uint8Array(value)
                            }
                            var getLength;
                            var valueIsOfTypeString = typeof value === "string";
                            if (!(valueIsOfTypeString || value instanceof Uint8Array || value instanceof Uint8ClampedArray || value instanceof Int8Array)) {
                                throwBindingError("Cannot pass non-string to std::string")
                            }
                            if (stdStringIsUTF8 && valueIsOfTypeString) {
                                getLength = function () {
                                    return lengthBytesUTF8(value)
                                }
                            } else {
                                getLength = function () {
                                    return value.length
                                }
                            }
                            var length = getLength();
                            var ptr = _malloc(4 + length + 1);
                            HEAPU32[ptr >> 2] = length;
                            if (stdStringIsUTF8 && valueIsOfTypeString) {
                                stringToUTF8(value, ptr + 4, length + 1)
                            } else {
                                if (valueIsOfTypeString) {
                                    for (var i = 0; i < length; ++i) {
                                        var charCode = value.charCodeAt(i);
                                        if (charCode > 255) {
                                            _free(ptr);
                                            throwBindingError("String has UTF-16 code units that do not fit in 8 bits")
                                        }
                                        HEAPU8[ptr + 4 + i] = charCode
                                    }
                                } else {
                                    for (var i = 0; i < length; ++i) {
                                        HEAPU8[ptr + 4 + i] = value[i]
                                    }
                                }
                            }
                            if (destructors !== null) {
                                destructors.push(_free, ptr)
                            }
                            return ptr
                        },
                        "argPackAdvance": 8,
                        "readValueFromPointer": simpleReadValueFromPointer,
                        destructorFunction: function (ptr) {
                            _free(ptr)
                        }
                    })
                }

                function __embind_register_std_wstring(rawType, charSize, name) {
                    name = readLatin1String(name);
                    var decodeString, encodeString, getHeap, lengthBytesUTF, shift;
                    if (charSize === 2) {
                        decodeString = UTF16ToString;
                        encodeString = stringToUTF16;
                        lengthBytesUTF = lengthBytesUTF16;
                        getHeap = function () {
                            return HEAPU16
                        };
                        shift = 1
                    } else if (charSize === 4) {
                        decodeString = UTF32ToString;
                        encodeString = stringToUTF32;
                        lengthBytesUTF = lengthBytesUTF32;
                        getHeap = function () {
                            return HEAPU32
                        };
                        shift = 2
                    }
                    registerType(rawType, {
                        name: name,
                        "fromWireType": function (value) {
                            var length = HEAPU32[value >> 2];
                            var HEAP = getHeap();
                            var str;
                            var decodeStartPtr = value + 4;
                            for (var i = 0; i <= length; ++i) {
                                var currentBytePtr = value + 4 + i * charSize;
                                if (i == length || HEAP[currentBytePtr >> shift] == 0) {
                                    var maxReadBytes = currentBytePtr - decodeStartPtr;
                                    var stringSegment = decodeString(decodeStartPtr, maxReadBytes);
                                    if (str === undefined) {
                                        str = stringSegment
                                    } else {
                                        str += String.fromCharCode(0);
                                        str += stringSegment
                                    }
                                    decodeStartPtr = currentBytePtr + charSize
                                }
                            }
                            _free(value);
                            return str
                        },
                        "toWireType": function (destructors, value) {
                            if (!(typeof value === "string")) {
                                throwBindingError("Cannot pass non-string to C++ string type " + name)
                            }
                            var length = lengthBytesUTF(value);
                            var ptr = _malloc(4 + length + charSize);
                            HEAPU32[ptr >> 2] = length >> shift;
                            encodeString(value, ptr + 4, length + charSize);
                            if (destructors !== null) {
                                destructors.push(_free, ptr)
                            }
                            return ptr
                        },
                        "argPackAdvance": 8,
                        "readValueFromPointer": simpleReadValueFromPointer,
                        destructorFunction: function (ptr) {
                            _free(ptr)
                        }
                    })
                }

                function __embind_register_value_array(rawType, name, constructorSignature, rawConstructor, destructorSignature, rawDestructor) {
                    tupleRegistrations[rawType] = {
                        name: readLatin1String(name),
                        rawConstructor: embind__requireFunction(constructorSignature, rawConstructor),
                        rawDestructor: embind__requireFunction(destructorSignature, rawDestructor),
                        elements: []
                    }
                }

                function __embind_register_value_array_element(rawTupleType, getterReturnType, getterSignature, getter, getterContext, setterArgumentType, setterSignature, setter, setterContext) {
                    tupleRegistrations[rawTupleType].elements.push({
                        getterReturnType: getterReturnType,
                        getter: embind__requireFunction(getterSignature, getter),
                        getterContext: getterContext,
                        setterArgumentType: setterArgumentType,
                        setter: embind__requireFunction(setterSignature, setter),
                        setterContext: setterContext
                    })
                }

                function __embind_register_value_object(rawType, name, constructorSignature, rawConstructor, destructorSignature, rawDestructor) {
                    structRegistrations[rawType] = {
                        name: readLatin1String(name),
                        rawConstructor: embind__requireFunction(constructorSignature, rawConstructor),
                        rawDestructor: embind__requireFunction(destructorSignature, rawDestructor),
                        fields: []
                    }
                }

                function __embind_register_value_object_field(structType, fieldName, getterReturnType, getterSignature, getter, getterContext, setterArgumentType, setterSignature, setter, setterContext) {
                    structRegistrations[structType].fields.push({
                        fieldName: readLatin1String(fieldName),
                        getterReturnType: getterReturnType,
                        getter: embind__requireFunction(getterSignature, getter),
                        getterContext: getterContext,
                        setterArgumentType: setterArgumentType,
                        setter: embind__requireFunction(setterSignature, setter),
                        setterContext: setterContext
                    })
                }

                function __embind_register_void(rawType, name) {
                    name = readLatin1String(name);
                    registerType(rawType, {
                        isVoid: true, name: name, "argPackAdvance": 0, "fromWireType": function () {
                            return undefined
                        }, "toWireType": function (destructors, o) {
                            return undefined
                        }
                    })
                }

                var emval_symbols = {};

                function getStringOrSymbol(address) {
                    var symbol = emval_symbols[address];
                    if (symbol === undefined) {
                        return readLatin1String(address)
                    } else {
                        return symbol
                    }
                }

                var emval_methodCallers = [];

                function requireHandle(handle) {
                    if (!handle) {
                        throwBindingError("Cannot use deleted val. handle = " + handle)
                    }
                    return emval_handle_array[handle].value
                }

                function __emval_call_void_method(caller, handle, methodName, args) {
                    caller = emval_methodCallers[caller];
                    handle = requireHandle(handle);
                    methodName = getStringOrSymbol(methodName);
                    caller(handle, methodName, null, args)
                }

                function __emval_addMethodCaller(caller) {
                    var id = emval_methodCallers.length;
                    emval_methodCallers.push(caller);
                    return id
                }

                function requireRegisteredType(rawType, humanName) {
                    var impl = registeredTypes[rawType];
                    if (undefined === impl) {
                        throwBindingError(humanName + " has unknown type " + getTypeName(rawType))
                    }
                    return impl
                }

                function __emval_lookupTypes(argCount, argTypes) {
                    var a = new Array(argCount);
                    for (var i = 0; i < argCount; ++i) {
                        a[i] = requireRegisteredType(HEAP32[(argTypes >> 2) + i], "parameter " + i)
                    }
                    return a
                }

                function __emval_get_method_caller(argCount, argTypes) {
                    var types = __emval_lookupTypes(argCount, argTypes);
                    var retType = types[0];
                    var signatureName = retType.name + "_$" + types.slice(1).map(function (t) {
                        return t.name
                    }).join("_") + "$";
                    var params = ["retType"];
                    var args = [retType];
                    var argsList = "";
                    for (var i = 0; i < argCount - 1; ++i) {
                        argsList += (i !== 0 ? ", " : "") + "arg" + i;
                        params.push("argType" + i);
                        args.push(types[1 + i])
                    }
                    var functionName = makeLegalFunctionName("methodCaller_" + signatureName);
                    var functionBody = "return function " + functionName + "(handle, name, destructors, args) {\n";
                    var offset = 0;
                    for (var i = 0; i < argCount - 1; ++i) {
                        functionBody += "    var arg" + i + " = argType" + i + ".readValueFromPointer(args" + (offset ? "+" + offset : "") + ");\n";
                        offset += types[i + 1]["argPackAdvance"]
                    }
                    functionBody += "    var rv = handle[name](" + argsList + ");\n";
                    for (var i = 0; i < argCount - 1; ++i) {
                        if (types[i + 1]["deleteObject"]) {
                            functionBody += "    argType" + i + ".deleteObject(arg" + i + ");\n"
                        }
                    }
                    if (!retType.isVoid) {
                        functionBody += "    return retType.toWireType(destructors, rv);\n"
                    }
                    functionBody += "};\n";
                    params.push(functionBody);
                    var invokerFunction = new_(Function, params).apply(null, args);
                    return __emval_addMethodCaller(invokerFunction)
                }

                function __emval_incref(handle) {
                    if (handle > 4) {
                        emval_handle_array[handle].refcount += 1
                    }
                }

                function __emval_new_array() {
                    return __emval_register([])
                }

                function __emval_take_value(type, argv) {
                    type = requireRegisteredType(type, "_emval_take_value");
                    var v = type["readValueFromPointer"](argv);
                    return __emval_register(v)
                }

                function _abort() {
                    abort()
                }

                function _emscripten_get_sbrk_ptr() {
                    return 642656
                }

                function _emscripten_memcpy_big(dest, src, num) {
                    HEAPU8.copyWithin(dest, src, src + num)
                }

                function _emscripten_get_heap_size() {
                    return HEAPU8.length
                }

                function emscripten_realloc_buffer(size) {
                    try {
                        wasmMemory.grow(size - buffer.byteLength + 65535 >>> 16);
                        updateGlobalBufferAndViews(wasmMemory.buffer);
                        return 1
                    } catch (e) {
                    }
                }

                function _emscripten_resize_heap(requestedSize) {
                    requestedSize = requestedSize >>> 0;
                    var oldSize = _emscripten_get_heap_size();
                    var PAGE_MULTIPLE = 65536;
                    var maxHeapSize = 2147483648;
                    if (requestedSize > maxHeapSize) {
                        return false
                    }
                    var minHeapSize = 16777216;
                    for (var cutDown = 1; cutDown <= 4; cutDown *= 2) {
                        var overGrownHeapSize = oldSize * (1 + .2 / cutDown);
                        overGrownHeapSize = Math.min(overGrownHeapSize, requestedSize + 100663296);
                        var newSize = Math.min(maxHeapSize, alignUp(Math.max(minHeapSize, requestedSize, overGrownHeapSize), PAGE_MULTIPLE));
                        var replacement = emscripten_realloc_buffer(newSize);
                        if (replacement) {
                            return true
                        }
                    }
                    return false
                }

                var ENV = {};

                function __getExecutableName() {
                    return thisProgram || "./this.program"
                }

                function getEnvStrings() {
                    if (!getEnvStrings.strings) {
                        var lang = (typeof navigator === "object" && navigator.languages && navigator.languages[0] || "C").replace("-", "_") + ".UTF-8";
                        var env = {
                            "USER": "web_user",
                            "LOGNAME": "web_user",
                            "PATH": "/",
                            "PWD": "/",
                            "HOME": "/home/web_user",
                            "LANG": lang,
                            "_": __getExecutableName()
                        };
                        for (var x in ENV) {
                            env[x] = ENV[x]
                        }
                        var strings = [];
                        for (var x in env) {
                            strings.push(x + "=" + env[x])
                        }
                        getEnvStrings.strings = strings
                    }
                    return getEnvStrings.strings
                }

                function _environ_get(__environ, environ_buf) {
                    var bufSize = 0;
                    getEnvStrings().forEach(function (string, i) {
                        var ptr = environ_buf + bufSize;
                        HEAP32[__environ + i * 4 >> 2] = ptr;
                        writeAsciiToMemory(string, ptr);
                        bufSize += string.length + 1
                    });
                    return 0
                }

                function _environ_sizes_get(penviron_count, penviron_buf_size) {
                    var strings = getEnvStrings();
                    HEAP32[penviron_count >> 2] = strings.length;
                    var bufSize = 0;
                    strings.forEach(function (string) {
                        bufSize += string.length + 1
                    });
                    HEAP32[penviron_buf_size >> 2] = bufSize;
                    return 0
                }

                function _fd_close(fd) {
                    try {
                        var stream = SYSCALLS.getStreamFromFD(fd);
                        FS.close(stream);
                        return 0
                    } catch (e) {
                        if (typeof FS === "undefined" || !(e instanceof FS.ErrnoError)) abort(e);
                        return e.errno
                    }
                }

                function _fd_read(fd, iov, iovcnt, pnum) {
                    try {
                        var stream = SYSCALLS.getStreamFromFD(fd);
                        var num = SYSCALLS.doReadv(stream, iov, iovcnt);
                        HEAP32[pnum >> 2] = num;
                        return 0
                    } catch (e) {
                        if (typeof FS === "undefined" || !(e instanceof FS.ErrnoError)) abort(e);
                        return e.errno
                    }
                }

                function _fd_seek(fd, offset_low, offset_high, whence, newOffset) {
                    try {
                        var stream = SYSCALLS.getStreamFromFD(fd);
                        var HIGH_OFFSET = 4294967296;
                        var offset = offset_high * HIGH_OFFSET + (offset_low >>> 0);
                        var DOUBLE_LIMIT = 9007199254740992;
                        if (offset <= -DOUBLE_LIMIT || offset >= DOUBLE_LIMIT) {
                            return -61
                        }
                        FS.llseek(stream, offset, whence);
                        tempI64 = [stream.position >>> 0, (tempDouble = stream.position, +Math_abs(tempDouble) >= 1 ? tempDouble > 0 ? (Math_min(+Math_floor(tempDouble / 4294967296), 4294967295) | 0) >>> 0 : ~~+Math_ceil((tempDouble - +(~~tempDouble >>> 0)) / 4294967296) >>> 0 : 0)], HEAP32[newOffset >> 2] = tempI64[0], HEAP32[newOffset + 4 >> 2] = tempI64[1];
                        if (stream.getdents && offset === 0 && whence === 0) stream.getdents = null;
                        return 0
                    } catch (e) {
                        if (typeof FS === "undefined" || !(e instanceof FS.ErrnoError)) abort(e);
                        return e.errno
                    }
                }

                function _fd_write(fd, iov, iovcnt, pnum) {
                    try {
                        var stream = SYSCALLS.getStreamFromFD(fd);
                        var num = SYSCALLS.doWritev(stream, iov, iovcnt);
                        HEAP32[pnum >> 2] = num;
                        return 0
                    } catch (e) {
                        if (typeof FS === "undefined" || !(e instanceof FS.ErrnoError)) abort(e);
                        return e.errno
                    }
                }

                function _pthread_mutexattr_destroy() {
                }

                function _pthread_mutexattr_init() {
                }

                function _pthread_mutexattr_settype() {
                }

                function _setTempRet0($i) {
                    setTempRet0($i | 0)
                }

                function __isLeapYear(year) {
                    return year % 4 === 0 && (year % 100 !== 0 || year % 400 === 0)
                }

                function __arraySum(array, index) {
                    var sum = 0;
                    for (var i = 0; i <= index; sum += array[i++]) {
                    }
                    return sum
                }

                var __MONTH_DAYS_LEAP = [31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31];
                var __MONTH_DAYS_REGULAR = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31];

                function __addDays(date, days) {
                    var newDate = new Date(date.getTime());
                    while (days > 0) {
                        var leap = __isLeapYear(newDate.getFullYear());
                        var currentMonth = newDate.getMonth();
                        var daysInCurrentMonth = (leap ? __MONTH_DAYS_LEAP : __MONTH_DAYS_REGULAR)[currentMonth];
                        if (days > daysInCurrentMonth - newDate.getDate()) {
                            days -= daysInCurrentMonth - newDate.getDate() + 1;
                            newDate.setDate(1);
                            if (currentMonth < 11) {
                                newDate.setMonth(currentMonth + 1)
                            } else {
                                newDate.setMonth(0);
                                newDate.setFullYear(newDate.getFullYear() + 1)
                            }
                        } else {
                            newDate.setDate(newDate.getDate() + days);
                            return newDate
                        }
                    }
                    return newDate
                }

                function _strftime(s, maxsize, format, tm) {
                    var tm_zone = HEAP32[tm + 40 >> 2];
                    var date = {
                        tm_sec: HEAP32[tm >> 2],
                        tm_min: HEAP32[tm + 4 >> 2],
                        tm_hour: HEAP32[tm + 8 >> 2],
                        tm_mday: HEAP32[tm + 12 >> 2],
                        tm_mon: HEAP32[tm + 16 >> 2],
                        tm_year: HEAP32[tm + 20 >> 2],
                        tm_wday: HEAP32[tm + 24 >> 2],
                        tm_yday: HEAP32[tm + 28 >> 2],
                        tm_isdst: HEAP32[tm + 32 >> 2],
                        tm_gmtoff: HEAP32[tm + 36 >> 2],
                        tm_zone: tm_zone ? UTF8ToString(tm_zone) : ""
                    };
                    var pattern = UTF8ToString(format);
                    var EXPANSION_RULES_1 = {
                        "%c": "%a %b %d %H:%M:%S %Y",
                        "%D": "%m/%d/%y",
                        "%F": "%Y-%m-%d",
                        "%h": "%b",
                        "%r": "%I:%M:%S %p",
                        "%R": "%H:%M",
                        "%T": "%H:%M:%S",
                        "%x": "%m/%d/%y",
                        "%X": "%H:%M:%S",
                        "%Ec": "%c",
                        "%EC": "%C",
                        "%Ex": "%m/%d/%y",
                        "%EX": "%H:%M:%S",
                        "%Ey": "%y",
                        "%EY": "%Y",
                        "%Od": "%d",
                        "%Oe": "%e",
                        "%OH": "%H",
                        "%OI": "%I",
                        "%Om": "%m",
                        "%OM": "%M",
                        "%OS": "%S",
                        "%Ou": "%u",
                        "%OU": "%U",
                        "%OV": "%V",
                        "%Ow": "%w",
                        "%OW": "%W",
                        "%Oy": "%y"
                    };
                    for (var rule in EXPANSION_RULES_1) {
                        pattern = pattern.replace(new RegExp(rule, "g"), EXPANSION_RULES_1[rule])
                    }
                    var WEEKDAYS = ["Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"];
                    var MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"];

                    function leadingSomething(value, digits, character) {
                        var str = typeof value === "number" ? value.toString() : value || "";
                        while (str.length < digits) {
                            str = character[0] + str
                        }
                        return str
                    }

                    function leadingNulls(value, digits) {
                        return leadingSomething(value, digits, "0")
                    }

                    function compareByDay(date1, date2) {
                        function sgn(value) {
                            return value < 0 ? -1 : value > 0 ? 1 : 0
                        }

                        var compare;
                        if ((compare = sgn(date1.getFullYear() - date2.getFullYear())) === 0) {
                            if ((compare = sgn(date1.getMonth() - date2.getMonth())) === 0) {
                                compare = sgn(date1.getDate() - date2.getDate())
                            }
                        }
                        return compare
                    }

                    function getFirstWeekStartDate(janFourth) {
                        switch (janFourth.getDay()) {
                            case 0:
                                return new Date(janFourth.getFullYear() - 1, 11, 29);
                            case 1:
                                return janFourth;
                            case 2:
                                return new Date(janFourth.getFullYear(), 0, 3);
                            case 3:
                                return new Date(janFourth.getFullYear(), 0, 2);
                            case 4:
                                return new Date(janFourth.getFullYear(), 0, 1);
                            case 5:
                                return new Date(janFourth.getFullYear() - 1, 11, 31);
                            case 6:
                                return new Date(janFourth.getFullYear() - 1, 11, 30)
                        }
                    }

                    function getWeekBasedYear(date) {
                        var thisDate = __addDays(new Date(date.tm_year + 1900, 0, 1), date.tm_yday);
                        var janFourthThisYear = new Date(thisDate.getFullYear(), 0, 4);
                        var janFourthNextYear = new Date(thisDate.getFullYear() + 1, 0, 4);
                        var firstWeekStartThisYear = getFirstWeekStartDate(janFourthThisYear);
                        var firstWeekStartNextYear = getFirstWeekStartDate(janFourthNextYear);
                        if (compareByDay(firstWeekStartThisYear, thisDate) <= 0) {
                            if (compareByDay(firstWeekStartNextYear, thisDate) <= 0) {
                                return thisDate.getFullYear() + 1
                            } else {
                                return thisDate.getFullYear()
                            }
                        } else {
                            return thisDate.getFullYear() - 1
                        }
                    }

                    var EXPANSION_RULES_2 = {
                        "%a": function (date) {
                            return WEEKDAYS[date.tm_wday].substring(0, 3)
                        }, "%A": function (date) {
                            return WEEKDAYS[date.tm_wday]
                        }, "%b": function (date) {
                            return MONTHS[date.tm_mon].substring(0, 3)
                        }, "%B": function (date) {
                            return MONTHS[date.tm_mon]
                        }, "%C": function (date) {
                            var year = date.tm_year + 1900;
                            return leadingNulls(year / 100 | 0, 2)
                        }, "%d": function (date) {
                            return leadingNulls(date.tm_mday, 2)
                        }, "%e": function (date) {
                            return leadingSomething(date.tm_mday, 2, " ")
                        }, "%g": function (date) {
                            return getWeekBasedYear(date).toString().substring(2)
                        }, "%G": function (date) {
                            return getWeekBasedYear(date)
                        }, "%H": function (date) {
                            return leadingNulls(date.tm_hour, 2)
                        }, "%I": function (date) {
                            var twelveHour = date.tm_hour;
                            if (twelveHour == 0) twelveHour = 12; else if (twelveHour > 12) twelveHour -= 12;
                            return leadingNulls(twelveHour, 2)
                        }, "%j": function (date) {
                            return leadingNulls(date.tm_mday + __arraySum(__isLeapYear(date.tm_year + 1900) ? __MONTH_DAYS_LEAP : __MONTH_DAYS_REGULAR, date.tm_mon - 1), 3)
                        }, "%m": function (date) {
                            return leadingNulls(date.tm_mon + 1, 2)
                        }, "%M": function (date) {
                            return leadingNulls(date.tm_min, 2)
                        }, "%n": function () {
                            return "\n"
                        }, "%p": function (date) {
                            if (date.tm_hour >= 0 && date.tm_hour < 12) {
                                return "AM"
                            } else {
                                return "PM"
                            }
                        }, "%S": function (date) {
                            return leadingNulls(date.tm_sec, 2)
                        }, "%t": function () {
                            return "\t"
                        }, "%u": function (date) {
                            return date.tm_wday || 7
                        }, "%U": function (date) {
                            var janFirst = new Date(date.tm_year + 1900, 0, 1);
                            var firstSunday = janFirst.getDay() === 0 ? janFirst : __addDays(janFirst, 7 - janFirst.getDay());
                            var endDate = new Date(date.tm_year + 1900, date.tm_mon, date.tm_mday);
                            if (compareByDay(firstSunday, endDate) < 0) {
                                var februaryFirstUntilEndMonth = __arraySum(__isLeapYear(endDate.getFullYear()) ? __MONTH_DAYS_LEAP : __MONTH_DAYS_REGULAR, endDate.getMonth() - 1) - 31;
                                var firstSundayUntilEndJanuary = 31 - firstSunday.getDate();
                                var days = firstSundayUntilEndJanuary + februaryFirstUntilEndMonth + endDate.getDate();
                                return leadingNulls(Math.ceil(days / 7), 2)
                            }
                            return compareByDay(firstSunday, janFirst) === 0 ? "01" : "00"
                        }, "%V": function (date) {
                            var janFourthThisYear = new Date(date.tm_year + 1900, 0, 4);
                            var janFourthNextYear = new Date(date.tm_year + 1901, 0, 4);
                            var firstWeekStartThisYear = getFirstWeekStartDate(janFourthThisYear);
                            var firstWeekStartNextYear = getFirstWeekStartDate(janFourthNextYear);
                            var endDate = __addDays(new Date(date.tm_year + 1900, 0, 1), date.tm_yday);
                            if (compareByDay(endDate, firstWeekStartThisYear) < 0) {
                                return "53"
                            }
                            if (compareByDay(firstWeekStartNextYear, endDate) <= 0) {
                                return "01"
                            }
                            var daysDifference;
                            if (firstWeekStartThisYear.getFullYear() < date.tm_year + 1900) {
                                daysDifference = date.tm_yday + 32 - firstWeekStartThisYear.getDate()
                            } else {
                                daysDifference = date.tm_yday + 1 - firstWeekStartThisYear.getDate()
                            }
                            return leadingNulls(Math.ceil(daysDifference / 7), 2)
                        }, "%w": function (date) {
                            return date.tm_wday
                        }, "%W": function (date) {
                            var janFirst = new Date(date.tm_year, 0, 1);
                            var firstMonday = janFirst.getDay() === 1 ? janFirst : __addDays(janFirst, janFirst.getDay() === 0 ? 1 : 7 - janFirst.getDay() + 1);
                            var endDate = new Date(date.tm_year + 1900, date.tm_mon, date.tm_mday);
                            if (compareByDay(firstMonday, endDate) < 0) {
                                var februaryFirstUntilEndMonth = __arraySum(__isLeapYear(endDate.getFullYear()) ? __MONTH_DAYS_LEAP : __MONTH_DAYS_REGULAR, endDate.getMonth() - 1) - 31;
                                var firstMondayUntilEndJanuary = 31 - firstMonday.getDate();
                                var days = firstMondayUntilEndJanuary + februaryFirstUntilEndMonth + endDate.getDate();
                                return leadingNulls(Math.ceil(days / 7), 2)
                            }
                            return compareByDay(firstMonday, janFirst) === 0 ? "01" : "00"
                        }, "%y": function (date) {
                            return (date.tm_year + 1900).toString().substring(2)
                        }, "%Y": function (date) {
                            return date.tm_year + 1900
                        }, "%z": function (date) {
                            var off = date.tm_gmtoff;
                            var ahead = off >= 0;
                            off = Math.abs(off) / 60;
                            off = off / 60 * 100 + off % 60;
                            return (ahead ? "+" : "-") + String("0000" + off).slice(-4)
                        }, "%Z": function (date) {
                            return date.tm_zone
                        }, "%%": function () {
                            return "%"
                        }
                    };
                    for (var rule in EXPANSION_RULES_2) {
                        if (pattern.indexOf(rule) >= 0) {
                            pattern = pattern.replace(new RegExp(rule, "g"), EXPANSION_RULES_2[rule](date))
                        }
                    }
                    var bytes = intArrayFromString(pattern, false);
                    if (bytes.length > maxsize) {
                        return 0
                    }
                    writeArrayToMemory(bytes, s);
                    return bytes.length - 1
                }

                function _strftime_l(s, maxsize, format, tm) {
                    return _strftime(s, maxsize, format, tm)
                }

                Module["requestFullscreen"] = function Module_requestFullscreen(lockPointer, resizeCanvas) {
                    Browser.requestFullscreen(lockPointer, resizeCanvas)
                };
                Module["requestAnimationFrame"] = function Module_requestAnimationFrame(func) {
                    Browser.requestAnimationFrame(func)
                };
                Module["setCanvasSize"] = function Module_setCanvasSize(width, height, noUpdates) {
                    Browser.setCanvasSize(width, height, noUpdates)
                };
                Module["pauseMainLoop"] = function Module_pauseMainLoop() {
                    Browser.mainLoop.pause()
                };
                Module["resumeMainLoop"] = function Module_resumeMainLoop() {
                    Browser.mainLoop.resume()
                };
                Module["getUserMedia"] = function Module_getUserMedia() {
                    Browser.getUserMedia()
                };
                Module["createContext"] = function Module_createContext(canvas, useWebGL, setInModule, webGLContextAttributes) {
                    return Browser.createContext(canvas, useWebGL, setInModule, webGLContextAttributes)
                };
                var FSNode = function (parent, name, mode, rdev) {
                    if (!parent) {
                        parent = this
                    }
                    this.parent = parent;
                    this.mount = parent.mount;
                    this.mounted = null;
                    this.id = FS.nextInode++;
                    this.name = name;
                    this.mode = mode;
                    this.node_ops = {};
                    this.stream_ops = {};
                    this.rdev = rdev
                };
                var readMode = 292 | 73;
                var writeMode = 146;
                Object.defineProperties(FSNode.prototype, {
                    read: {
                        get: function () {
                            return (this.mode & readMode) === readMode
                        }, set: function (val) {
                            val ? this.mode |= readMode : this.mode &= ~readMode
                        }
                    }, write: {
                        get: function () {
                            return (this.mode & writeMode) === writeMode
                        }, set: function (val) {
                            val ? this.mode |= writeMode : this.mode &= ~writeMode
                        }
                    }, isFolder: {
                        get: function () {
                            return FS.isDir(this.mode)
                        }
                    }, isDevice: {
                        get: function () {
                            return FS.isChrdev(this.mode)
                        }
                    }
                });
                FS.FSNode = FSNode;
                FS.staticInit();
                Module["FS_createFolder"] = FS.createFolder;
                Module["FS_createPath"] = FS.createPath;
                Module["FS_createDataFile"] = FS.createDataFile;
                Module["FS_createPreloadedFile"] = FS.createPreloadedFile;
                Module["FS_createLazyFile"] = FS.createLazyFile;
                Module["FS_createLink"] = FS.createLink;
                Module["FS_createDevice"] = FS.createDevice;
                Module["FS_unlink"] = FS.unlink;
                InternalError = Module["InternalError"] = extendError(Error, "InternalError");
                embind_init_charCodes();
                BindingError = Module["BindingError"] = extendError(Error, "BindingError");
                init_ClassHandle();
                init_RegisteredPointer();
                init_embind();
                UnboundTypeError = Module["UnboundTypeError"] = extendError(Error, "UnboundTypeError");
                init_emval();
                var ASSERTIONS = false;

                function intArrayFromString(stringy, dontAddNull, length) {
                    var len = length > 0 ? length : lengthBytesUTF8(stringy) + 1;
                    var u8array = new Array(len);
                    var numBytesWritten = stringToUTF8Array(stringy, u8array, 0, u8array.length);
                    if (dontAddNull) u8array.length = numBytesWritten;
                    return u8array
                }

                var asmLibraryArg = {
                    "__cxa_allocate_exception": ___cxa_allocate_exception,
                    "__cxa_atexit": ___cxa_atexit,
                    "__cxa_throw": ___cxa_throw,
                    "__map_file": ___map_file,
                    "__sys_munmap": ___sys_munmap,
                    "_embind_finalize_value_array": __embind_finalize_value_array,
                    "_embind_finalize_value_object": __embind_finalize_value_object,
                    "_embind_register_bool": __embind_register_bool,
                    "_embind_register_class": __embind_register_class,
                    "_embind_register_class_class_function": __embind_register_class_class_function,
                    "_embind_register_class_constructor": __embind_register_class_constructor,
                    "_embind_register_class_function": __embind_register_class_function,
                    "_embind_register_class_property": __embind_register_class_property,
                    "_embind_register_constant": __embind_register_constant,
                    "_embind_register_emval": __embind_register_emval,
                    "_embind_register_float": __embind_register_float,
                    "_embind_register_function": __embind_register_function,
                    "_embind_register_integer": __embind_register_integer,
                    "_embind_register_memory_view": __embind_register_memory_view,
                    "_embind_register_std_string": __embind_register_std_string,
                    "_embind_register_std_wstring": __embind_register_std_wstring,
                    "_embind_register_value_array": __embind_register_value_array,
                    "_embind_register_value_array_element": __embind_register_value_array_element,
                    "_embind_register_value_object": __embind_register_value_object,
                    "_embind_register_value_object_field": __embind_register_value_object_field,
                    "_embind_register_void": __embind_register_void,
                    "_emval_call_void_method": __emval_call_void_method,
                    "_emval_decref": __emval_decref,
                    "_emval_get_method_caller": __emval_get_method_caller,
                    "_emval_incref": __emval_incref,
                    "_emval_new_array": __emval_new_array,
                    "_emval_take_value": __emval_take_value,
                    "abort": _abort,
                    "emscripten_get_sbrk_ptr": _emscripten_get_sbrk_ptr,
                    "emscripten_memcpy_big": _emscripten_memcpy_big,
                    "emscripten_resize_heap": _emscripten_resize_heap,
                    "environ_get": _environ_get,
                    "environ_sizes_get": _environ_sizes_get,
                    "fd_close": _fd_close,
                    "fd_read": _fd_read,
                    "fd_seek": _fd_seek,
                    "fd_write": _fd_write,
                    "memory": wasmMemory,
                    "pthread_mutexattr_destroy": _pthread_mutexattr_destroy,
                    "pthread_mutexattr_init": _pthread_mutexattr_init,
                    "pthread_mutexattr_settype": _pthread_mutexattr_settype,
                    "setTempRet0": _setTempRet0,
                    "strftime_l": _strftime_l,
                    "table": wasmTable
                };
                var asm = createWasm();
                var ___wasm_call_ctors = Module["___wasm_call_ctors"] = function () {
                    return (___wasm_call_ctors = Module["___wasm_call_ctors"] = Module["asm"]["__wasm_call_ctors"]).apply(null, arguments)
                };
                var _malloc = Module["_malloc"] = function () {
                    return (_malloc = Module["_malloc"] = Module["asm"]["malloc"]).apply(null, arguments)
                };
                var _free = Module["_free"] = function () {
                    return (_free = Module["_free"] = Module["asm"]["free"]).apply(null, arguments)
                };
                var ___errno_location = Module["___errno_location"] = function () {
                    return (___errno_location = Module["___errno_location"] = Module["asm"]["__errno_location"]).apply(null, arguments)
                };
                var ___getTypeName = Module["___getTypeName"] = function () {
                    return (___getTypeName = Module["___getTypeName"] = Module["asm"]["__getTypeName"]).apply(null, arguments)
                };
                var ___embind_register_native_and_builtin_types = Module["___embind_register_native_and_builtin_types"] = function () {
                    return (___embind_register_native_and_builtin_types = Module["___embind_register_native_and_builtin_types"] = Module["asm"]["__embind_register_native_and_builtin_types"]).apply(null, arguments)
                };
                var _setThrew = Module["_setThrew"] = function () {
                    return (_setThrew = Module["_setThrew"] = Module["asm"]["setThrew"]).apply(null, arguments)
                };
                var stackSave = Module["stackSave"] = function () {
                    return (stackSave = Module["stackSave"] = Module["asm"]["stackSave"]).apply(null, arguments)
                };
                var stackRestore = Module["stackRestore"] = function () {
                    return (stackRestore = Module["stackRestore"] = Module["asm"]["stackRestore"]).apply(null, arguments)
                };
                var stackAlloc = Module["stackAlloc"] = function () {
                    return (stackAlloc = Module["stackAlloc"] = Module["asm"]["stackAlloc"]).apply(null, arguments)
                };
                var ___cxa_demangle = Module["___cxa_demangle"] = function () {
                    return (___cxa_demangle = Module["___cxa_demangle"] = Module["asm"]["__cxa_demangle"]).apply(null, arguments)
                };
                var _emscripten_main_thread_process_queued_calls = Module["_emscripten_main_thread_process_queued_calls"] = function () {
                    return (_emscripten_main_thread_process_queued_calls = Module["_emscripten_main_thread_process_queued_calls"] = Module["asm"]["emscripten_main_thread_process_queued_calls"]).apply(null, arguments)
                };
                var __growWasmMemory = Module["__growWasmMemory"] = function () {
                    return (__growWasmMemory = Module["__growWasmMemory"] = Module["asm"]["__growWasmMemory"]).apply(null, arguments)
                };
                var dynCall_ii = Module["dynCall_ii"] = function () {
                    return (dynCall_ii = Module["dynCall_ii"] = Module["asm"]["dynCall_ii"]).apply(null, arguments)
                };
                var dynCall_vi = Module["dynCall_vi"] = function () {
                    return (dynCall_vi = Module["dynCall_vi"] = Module["asm"]["dynCall_vi"]).apply(null, arguments)
                };
                var dynCall_i = Module["dynCall_i"] = function () {
                    return (dynCall_i = Module["dynCall_i"] = Module["asm"]["dynCall_i"]).apply(null, arguments)
                };
                var dynCall_iii = Module["dynCall_iii"] = function () {
                    return (dynCall_iii = Module["dynCall_iii"] = Module["asm"]["dynCall_iii"]).apply(null, arguments)
                };
                var dynCall_iiii = Module["dynCall_iiii"] = function () {
                    return (dynCall_iiii = Module["dynCall_iiii"] = Module["asm"]["dynCall_iiii"]).apply(null, arguments)
                };
                var dynCall_iiiii = Module["dynCall_iiiii"] = function () {
                    return (dynCall_iiiii = Module["dynCall_iiiii"] = Module["asm"]["dynCall_iiiii"]).apply(null, arguments)
                };
                var dynCall_iiiiii = Module["dynCall_iiiiii"] = function () {
                    return (dynCall_iiiiii = Module["dynCall_iiiiii"] = Module["asm"]["dynCall_iiiiii"]).apply(null, arguments)
                };
                var dynCall_iiiiiii = Module["dynCall_iiiiiii"] = function () {
                    return (dynCall_iiiiiii = Module["dynCall_iiiiiii"] = Module["asm"]["dynCall_iiiiiii"]).apply(null, arguments)
                };
                var dynCall_viii = Module["dynCall_viii"] = function () {
                    return (dynCall_viii = Module["dynCall_viii"] = Module["asm"]["dynCall_viii"]).apply(null, arguments)
                };
                var dynCall_viiii = Module["dynCall_viiii"] = function () {
                    return (dynCall_viiii = Module["dynCall_viiii"] = Module["asm"]["dynCall_viiii"]).apply(null, arguments)
                };
                var dynCall_vii = Module["dynCall_vii"] = function () {
                    return (dynCall_vii = Module["dynCall_vii"] = Module["asm"]["dynCall_vii"]).apply(null, arguments)
                };
                var dynCall_viiidd = Module["dynCall_viiidd"] = function () {
                    return (dynCall_viiidd = Module["dynCall_viiidd"] = Module["asm"]["dynCall_viiidd"]).apply(null, arguments)
                };
                var dynCall_viiiidd = Module["dynCall_viiiidd"] = function () {
                    return (dynCall_viiiidd = Module["dynCall_viiiidd"] = Module["asm"]["dynCall_viiiidd"]).apply(null, arguments)
                };
                var dynCall_viiid = Module["dynCall_viiid"] = function () {
                    return (dynCall_viiid = Module["dynCall_viiid"] = Module["asm"]["dynCall_viiid"]).apply(null, arguments)
                };
                var dynCall_viiiid = Module["dynCall_viiiid"] = function () {
                    return (dynCall_viiiid = Module["dynCall_viiiid"] = Module["asm"]["dynCall_viiiid"]).apply(null, arguments)
                };
                var dynCall_viiiii = Module["dynCall_viiiii"] = function () {
                    return (dynCall_viiiii = Module["dynCall_viiiii"] = Module["asm"]["dynCall_viiiii"]).apply(null, arguments)
                };
                var dynCall_dii = Module["dynCall_dii"] = function () {
                    return (dynCall_dii = Module["dynCall_dii"] = Module["asm"]["dynCall_dii"]).apply(null, arguments)
                };
                var dynCall_diii = Module["dynCall_diii"] = function () {
                    return (dynCall_diii = Module["dynCall_diii"] = Module["asm"]["dynCall_diii"]).apply(null, arguments)
                };
                var dynCall_iiiid = Module["dynCall_iiiid"] = function () {
                    return (dynCall_iiiid = Module["dynCall_iiiid"] = Module["asm"]["dynCall_iiiid"]).apply(null, arguments)
                };
                var dynCall_fiii = Module["dynCall_fiii"] = function () {
                    return (dynCall_fiii = Module["dynCall_fiii"] = Module["asm"]["dynCall_fiii"]).apply(null, arguments)
                };
                var dynCall_fiiii = Module["dynCall_fiiii"] = function () {
                    return (dynCall_fiiii = Module["dynCall_fiiii"] = Module["asm"]["dynCall_fiiii"]).apply(null, arguments)
                };
                var dynCall_fiiiii = Module["dynCall_fiiiii"] = function () {
                    return (dynCall_fiiiii = Module["dynCall_fiiiii"] = Module["asm"]["dynCall_fiiiii"]).apply(null, arguments)
                };
                var dynCall_diiiii = Module["dynCall_diiiii"] = function () {
                    return (dynCall_diiiii = Module["dynCall_diiiii"] = Module["asm"]["dynCall_diiiii"]).apply(null, arguments)
                };
                var dynCall_diiii = Module["dynCall_diiii"] = function () {
                    return (dynCall_diiii = Module["dynCall_diiii"] = Module["asm"]["dynCall_diiii"]).apply(null, arguments)
                };
                var dynCall_viid = Module["dynCall_viid"] = function () {
                    return (dynCall_viid = Module["dynCall_viid"] = Module["asm"]["dynCall_viid"]).apply(null, arguments)
                };
                var dynCall_fii = Module["dynCall_fii"] = function () {
                    return (dynCall_fii = Module["dynCall_fii"] = Module["asm"]["dynCall_fii"]).apply(null, arguments)
                };
                var dynCall_viif = Module["dynCall_viif"] = function () {
                    return (dynCall_viif = Module["dynCall_viif"] = Module["asm"]["dynCall_viif"]).apply(null, arguments)
                };
                var dynCall_viiif = Module["dynCall_viiif"] = function () {
                    return (dynCall_viiif = Module["dynCall_viiif"] = Module["asm"]["dynCall_viiif"]).apply(null, arguments)
                };
                var dynCall_iiiif = Module["dynCall_iiiif"] = function () {
                    return (dynCall_iiiif = Module["dynCall_iiiif"] = Module["asm"]["dynCall_iiiif"]).apply(null, arguments)
                };
                var dynCall_viiiiiii = Module["dynCall_viiiiiii"] = function () {
                    return (dynCall_viiiiiii = Module["dynCall_viiiiiii"] = Module["asm"]["dynCall_viiiiiii"]).apply(null, arguments)
                };
                var dynCall_viiiiii = Module["dynCall_viiiiii"] = function () {
                    return (dynCall_viiiiii = Module["dynCall_viiiiii"] = Module["asm"]["dynCall_viiiiii"]).apply(null, arguments)
                };
                var dynCall_iiidd = Module["dynCall_iiidd"] = function () {
                    return (dynCall_iiidd = Module["dynCall_iiidd"] = Module["asm"]["dynCall_iiidd"]).apply(null, arguments)
                };
                var dynCall_viidd = Module["dynCall_viidd"] = function () {
                    return (dynCall_viidd = Module["dynCall_viidd"] = Module["asm"]["dynCall_viidd"]).apply(null, arguments)
                };
                var dynCall_viiiiddi = Module["dynCall_viiiiddi"] = function () {
                    return (dynCall_viiiiddi = Module["dynCall_viiiiddi"] = Module["asm"]["dynCall_viiiiddi"]).apply(null, arguments)
                };
                var dynCall_viiiddi = Module["dynCall_viiiddi"] = function () {
                    return (dynCall_viiiddi = Module["dynCall_viiiddi"] = Module["asm"]["dynCall_viiiddi"]).apply(null, arguments)
                };
                var dynCall_viiiiiiii = Module["dynCall_viiiiiiii"] = function () {
                    return (dynCall_viiiiiiii = Module["dynCall_viiiiiiii"] = Module["asm"]["dynCall_viiiiiiii"]).apply(null, arguments)
                };
                var dynCall_viiiiiiiii = Module["dynCall_viiiiiiiii"] = function () {
                    return (dynCall_viiiiiiiii = Module["dynCall_viiiiiiiii"] = Module["asm"]["dynCall_viiiiiiiii"]).apply(null, arguments)
                };
                var dynCall_viiiiiiiddi = Module["dynCall_viiiiiiiddi"] = function () {
                    return (dynCall_viiiiiiiddi = Module["dynCall_viiiiiiiddi"] = Module["asm"]["dynCall_viiiiiiiddi"]).apply(null, arguments)
                };
                var dynCall_viiiiiiiiiiddi = Module["dynCall_viiiiiiiiiiddi"] = function () {
                    return (dynCall_viiiiiiiiiiddi = Module["dynCall_viiiiiiiiiiddi"] = Module["asm"]["dynCall_viiiiiiiiiiddi"]).apply(null, arguments)
                };
                var dynCall_iiiiiiiii = Module["dynCall_iiiiiiiii"] = function () {
                    return (dynCall_iiiiiiiii = Module["dynCall_iiiiiiiii"] = Module["asm"]["dynCall_iiiiiiiii"]).apply(null, arguments)
                };
                var dynCall_viiiiiiiiii = Module["dynCall_viiiiiiiiii"] = function () {
                    return (dynCall_viiiiiiiiii = Module["dynCall_viiiiiiiiii"] = Module["asm"]["dynCall_viiiiiiiiii"]).apply(null, arguments)
                };
                var dynCall_viidi = Module["dynCall_viidi"] = function () {
                    return (dynCall_viidi = Module["dynCall_viidi"] = Module["asm"]["dynCall_viidi"]).apply(null, arguments)
                };
                var dynCall_vidii = Module["dynCall_vidii"] = function () {
                    return (dynCall_vidii = Module["dynCall_vidii"] = Module["asm"]["dynCall_vidii"]).apply(null, arguments)
                };
                var dynCall_viijii = Module["dynCall_viijii"] = function () {
                    return (dynCall_viijii = Module["dynCall_viijii"] = Module["asm"]["dynCall_viijii"]).apply(null, arguments)
                };
                var dynCall_v = Module["dynCall_v"] = function () {
                    return (dynCall_v = Module["dynCall_v"] = Module["asm"]["dynCall_v"]).apply(null, arguments)
                };
                var dynCall_viiiiiiiiidd = Module["dynCall_viiiiiiiiidd"] = function () {
                    return (dynCall_viiiiiiiiidd = Module["dynCall_viiiiiiiiidd"] = Module["asm"]["dynCall_viiiiiiiiidd"]).apply(null, arguments)
                };
                var dynCall_jiji = Module["dynCall_jiji"] = function () {
                    return (dynCall_jiji = Module["dynCall_jiji"] = Module["asm"]["dynCall_jiji"]).apply(null, arguments)
                };
                var dynCall_iidiiii = Module["dynCall_iidiiii"] = function () {
                    return (dynCall_iidiiii = Module["dynCall_iidiiii"] = Module["asm"]["dynCall_iidiiii"]).apply(null, arguments)
                };
                var dynCall_iiiiij = Module["dynCall_iiiiij"] = function () {
                    return (dynCall_iiiiij = Module["dynCall_iiiiij"] = Module["asm"]["dynCall_iiiiij"]).apply(null, arguments)
                };
                var dynCall_iiiiid = Module["dynCall_iiiiid"] = function () {
                    return (dynCall_iiiiid = Module["dynCall_iiiiid"] = Module["asm"]["dynCall_iiiiid"]).apply(null, arguments)
                };
                var dynCall_iiiiijj = Module["dynCall_iiiiijj"] = function () {
                    return (dynCall_iiiiijj = Module["dynCall_iiiiijj"] = Module["asm"]["dynCall_iiiiijj"]).apply(null, arguments)
                };
                var dynCall_iiiiiiii = Module["dynCall_iiiiiiii"] = function () {
                    return (dynCall_iiiiiiii = Module["dynCall_iiiiiiii"] = Module["asm"]["dynCall_iiiiiiii"]).apply(null, arguments)
                };
                var dynCall_iiiiiijj = Module["dynCall_iiiiiijj"] = function () {
                    return (dynCall_iiiiiijj = Module["dynCall_iiiiiijj"] = Module["asm"]["dynCall_iiiiiijj"]).apply(null, arguments)
                };
                Module["getMemory"] = getMemory;
                Module["addRunDependency"] = addRunDependency;
                Module["removeRunDependency"] = removeRunDependency;
                Module["FS_createFolder"] = FS.createFolder;
                Module["FS_createPath"] = FS.createPath;
                Module["FS_createDataFile"] = FS.createDataFile;
                Module["FS_createPreloadedFile"] = FS.createPreloadedFile;
                Module["FS_createLazyFile"] = FS.createLazyFile;
                Module["FS_createLink"] = FS.createLink;
                Module["FS_createDevice"] = FS.createDevice;
                Module["FS_unlink"] = FS.unlink;
                var calledRun;

                function ExitStatus(status) {
                    this.name = "ExitStatus";
                    this.message = "Program terminated with exit(" + status + ")";
                    this.status = status
                }

                dependenciesFulfilled = function runCaller() {
                    if (!calledRun) run();
                    if (!calledRun) dependenciesFulfilled = runCaller
                };

                function run(args) {
                    args = args || arguments_;
                    if (runDependencies > 0) {
                        return
                    }
                    preRun();
                    if (runDependencies > 0) return;

                    function doRun() {
                        if (calledRun) return;
                        calledRun = true;
                        Module["calledRun"] = true;
                        if (ABORT) return;
                        initRuntime();
                        preMain();
                        // readyPromiseResolve(Module);
                        if (Module["onRuntimeInitialized"]) Module["onRuntimeInitialized"]();
                        postRun()
                    }

                    if (Module["setStatus"]) {
                        Module["setStatus"]("Running...");
                        setTimeout(function () {
                            setTimeout(function () {
                                Module["setStatus"]("")
                            }, 1);
                            doRun()
                        }, 1)
                    } else {
                        doRun()
                    }
                }

                Module["run"] = run;
                if (Module["preInit"]) {
                    if (typeof Module["preInit"] == "function") Module["preInit"] = [Module["preInit"]];
                    while (Module["preInit"].length > 0) {
                        Module["preInit"].pop()()
                    }
                }
                noExitRuntime = true;
                run();
                if (!IsWechat) {
                    // 在微信里不执行，请见另一个Module["imread"]
                    Module["imread"] = function (imageSource) {
                        var img = null;
                        if (typeof imageSource === "string") {
                            img = document.getElementById(imageSource)
                        } else {
                            img = imageSource
                        }
                        var canvas = null;
                        var ctx = null;
                        if (img instanceof HTMLImageElement) {
                            canvas = document.createElement("canvas");
                            canvas.width = img.width;
                            canvas.height = img.height;
                            ctx = canvas.getContext("2d");
                            ctx.drawImage(img, 0, 0, img.width, img.height)
                        } else if (img instanceof HTMLCanvasElement) {
                            canvas = img;
                            ctx = canvas.getContext("2d")
                        } else {
                            throw new Error("Please input the valid canvas or img id.");
                            return
                        }
                        var imgData = ctx.getImageData(0, 0, canvas.width, canvas.height);
                        return cv.matFromImageData(imgData)

                    };
                    // 在微信里不执行，请见另一个Module["imshow"]
                    Module["imshow"] = function (canvasSource, mat) {
                        var canvas = null;
                        if (typeof canvasSource === "string") {
                            canvas = document.getElementById(canvasSource)
                        } else {
                            canvas = canvasSource
                        }
                        if (!(canvas instanceof HTMLCanvasElement)) {
                            throw new Error("Please input the valid canvas element or id.");
                            return
                        }
                        if (!(mat instanceof cv.Mat)) {
                            throw new Error("Please input the valid cv.Mat instance.");
                            return
                        }
                        var img = new cv.Mat;
                        var depth = mat.type() % 8;
                        var scale = depth <= cv.CV_8S ? 1 : depth <= cv.CV_32S ? 1 / 256 : 255;
                        var shift = depth === cv.CV_8S || depth === cv.CV_16S ? 128 : 0;
                        mat.convertTo(img, cv.CV_8U, scale, shift);
                        switch (img.type()) {
                            case cv.CV_8UC1:
                                cv.cvtColor(img, img, cv.COLOR_GRAY2RGBA);
                                break;
                            case cv.CV_8UC3:
                                cv.cvtColor(img, img, cv.COLOR_RGB2RGBA);
                                break;
                            case cv.CV_8UC4:
                                break;
                            default:
                                throw new Error("Bad number of channels (Source image must have 1, 3 or 4 channels)");
                                return
                        }
                        var imgData = new ImageData(new Uint8ClampedArray(img.data), img.cols, img.rows);
                        var ctx = canvas.getContext("2d");
                        ctx.clearRect(0, 0, canvas.width, canvas.height);
                        canvas.width = imgData.width;
                        canvas.height = imgData.height;
                        ctx.putImageData(imgData, 0, 0);
                        img.delete()

                    };
                } else {
                    Module["imread"] = function (imgData) {
                        return cv.matFromImageData(imgData)
                    };

                    Module["imshow"] = function (canvas, mat) {
                        if (!(mat instanceof cv.Mat)) {
                            throw new Error("Please input the valid cv.Mat instance.");
                            return
                        }
                        var img = new cv.Mat;
                        var depth = mat.type() % 8;
                        var scale = depth <= cv.CV_8S ? 1 : depth <= cv.CV_32S ? 1 / 256 : 255;
                        var shift = depth === cv.CV_8S || depth === cv.CV_16S ? 128 : 0;

                        mat.convertTo(img, cv.CV_8U, scale, shift);

                        switch (img.type()) {
                            case cv.CV_8UC1:
                                cv.cvtColor(img, img, cv.COLOR_GRAY2RGBA);
                                break;
                            case cv.CV_8UC3:
                                cv.cvtColor(img, img, cv.COLOR_RGB2RGBA);
                                break;
                            case cv.CV_8UC4:
                                break;
                            default:
                                throw new Error("Bad number of channels (Source image must have 1, 3 or 4 channels)");
                                return
                        }

                        var ctx = canvas.getContext("2d");
                        ctx.clearRect(0, 0, canvas.width, canvas.height);

                        // 创建ImageData对象
                        var imgData = ctx.createImageData(img.cols, img.rows);
                        // imgData.data是只读对象，但是imgData.data.set()方法可修改imgData.data。
                        imgData.data.set(new Uint8ClampedArray(img.data))
                        // 画布canvas的宽度和高度，不能比图像imgData小。
                        canvas.width = imgData.width;
                        canvas.height = imgData.height;
                        // 需要传递ImageData类型，但小程序无法通过new ImageData()创建该类型。
                        ctx.putImageData(imgData, 0, 0);
                        img.delete()
                    };
                }
                Module["VideoCapture"] = function (videoSource) {
                    var video = null;
                    if (typeof videoSource === "string") {
                        video = document.getElementById(videoSource)
                    } else {
                        video = videoSource
                    }
                    if (!(video instanceof HTMLVideoElement)) {
                        throw new Error("Please input the valid video element or id.");
                        return
                    }
                    var canvas = document.createElement("canvas");
                    canvas.width = video.width;
                    canvas.height = video.height;
                    var ctx = canvas.getContext("2d");
                    this.video = video;
                    this.read = function (frame) {
                        if (!(frame instanceof cv.Mat)) {
                            throw new Error("Please input the valid cv.Mat instance.");
                            return
                        }
                        if (frame.type() !== cv.CV_8UC4) {
                            throw new Error("Bad type of input mat: the type should be cv.CV_8UC4.");
                            return
                        }
                        if (frame.cols !== video.width || frame.rows !== video.height) {
                            throw new Error("Bad size of input mat: the size should be same as the video.");
                            return
                        }
                        ctx.drawImage(video, 0, 0, video.width, video.height);
                        frame.data.set(ctx.getImageData(0, 0, video.width, video.height).data)
                    }
                };

                function Range(start, end) {
                    this.start = typeof start === "undefined" ? 0 : start;
                    this.end = typeof end === "undefined" ? 0 : end
                }

                Module["Range"] = Range;

                function Point(x, y) {
                    this.x = typeof x === "undefined" ? 0 : x;
                    this.y = typeof y === "undefined" ? 0 : y
                }

                Module["Point"] = Point;

                function Size(width, height) {
                    this.width = typeof width === "undefined" ? 0 : width;
                    this.height = typeof height === "undefined" ? 0 : height
                }

                Module["Size"] = Size;

                function Rect() {
                    switch (arguments.length) {
                        case 0: {
                            this.x = 0;
                            this.y = 0;
                            this.width = 0;
                            this.height = 0;
                            break
                        }
                        case 1: {
                            var rect = arguments[0];
                            this.x = rect.x;
                            this.y = rect.y;
                            this.width = rect.width;
                            this.height = rect.height;
                            break
                        }
                        case 2: {
                            var point = arguments[0];
                            var size = arguments[1];
                            this.x = point.x;
                            this.y = point.y;
                            this.width = size.width;
                            this.height = size.height;
                            break
                        }
                        case 4: {
                            this.x = arguments[0];
                            this.y = arguments[1];
                            this.width = arguments[2];
                            this.height = arguments[3];
                            break
                        }
                        default: {
                            throw new Error("Invalid arguments")
                        }
                    }
                }

                Module["Rect"] = Rect;

                function RotatedRect() {
                    switch (arguments.length) {
                        case 0: {
                            this.center = {x: 0, y: 0};
                            this.size = {width: 0, height: 0};
                            this.angle = 0;
                            break
                        }
                        case 3: {
                            this.center = arguments[0];
                            this.size = arguments[1];
                            this.angle = arguments[2];
                            break
                        }
                        default: {
                            throw new Error("Invalid arguments")
                        }
                    }
                }

                RotatedRect.points = function (obj) {
                    return Module.rotatedRectPoints(obj)
                };
                RotatedRect.boundingRect = function (obj) {
                    return Module.rotatedRectBoundingRect(obj)
                };
                RotatedRect.boundingRect2f = function (obj) {
                    return Module.rotatedRectBoundingRect2f(obj)
                };
                Module["RotatedRect"] = RotatedRect;

                function Scalar(v0, v1, v2, v3) {
                    this.push(typeof v0 === "undefined" ? 0 : v0);
                    this.push(typeof v1 === "undefined" ? 0 : v1);
                    this.push(typeof v2 === "undefined" ? 0 : v2);
                    this.push(typeof v3 === "undefined" ? 0 : v3)
                }

                Scalar.prototype = new Array;
                Scalar.all = function (v) {
                    return new Scalar(v, v, v, v)
                };
                Module["Scalar"] = Scalar;

                function MinMaxLoc() {
                    switch (arguments.length) {
                        case 0: {
                            this.minVal = 0;
                            this.maxVal = 0;
                            this.minLoc = new Point;
                            this.maxLoc = new Point;
                            break
                        }
                        case 4: {
                            this.minVal = arguments[0];
                            this.maxVal = arguments[1];
                            this.minLoc = arguments[2];
                            this.maxLoc = arguments[3];
                            break
                        }
                        default: {
                            throw new Error("Invalid arguments")
                        }
                    }
                }

                Module["MinMaxLoc"] = MinMaxLoc;

                function Circle() {
                    switch (arguments.length) {
                        case 0: {
                            this.center = new Point;
                            this.radius = 0;
                            break
                        }
                        case 2: {
                            this.center = arguments[0];
                            this.radius = arguments[1];
                            break
                        }
                        default: {
                            throw new Error("Invalid arguments")
                        }
                    }
                }

                Module["Circle"] = Circle;

                function TermCriteria() {
                    switch (arguments.length) {
                        case 0: {
                            this.type = 0;
                            this.maxCount = 0;
                            this.epsilon = 0;
                            break
                        }
                        case 3: {
                            this.type = arguments[0];
                            this.maxCount = arguments[1];
                            this.epsilon = arguments[2];
                            break
                        }
                        default: {
                            throw new Error("Invalid arguments")
                        }
                    }
                }

                Module["TermCriteria"] = TermCriteria;
                Module["matFromArray"] = function (rows, cols, type, array) {
                    var mat = new cv.Mat(rows, cols, type);
                    switch (type) {
                        case cv.CV_8U:
                        case cv.CV_8UC1:
                        case cv.CV_8UC2:
                        case cv.CV_8UC3:
                        case cv.CV_8UC4: {
                            mat.data.set(array);
                            break
                        }
                        case cv.CV_8S:
                        case cv.CV_8SC1:
                        case cv.CV_8SC2:
                        case cv.CV_8SC3:
                        case cv.CV_8SC4: {
                            mat.data8S.set(array);
                            break
                        }
                        case cv.CV_16U:
                        case cv.CV_16UC1:
                        case cv.CV_16UC2:
                        case cv.CV_16UC3:
                        case cv.CV_16UC4: {
                            mat.data16U.set(array);
                            break
                        }
                        case cv.CV_16S:
                        case cv.CV_16SC1:
                        case cv.CV_16SC2:
                        case cv.CV_16SC3:
                        case cv.CV_16SC4: {
                            mat.data16S.set(array);
                            break
                        }
                        case cv.CV_32S:
                        case cv.CV_32SC1:
                        case cv.CV_32SC2:
                        case cv.CV_32SC3:
                        case cv.CV_32SC4: {
                            mat.data32S.set(array);
                            break
                        }
                        case cv.CV_32F:
                        case cv.CV_32FC1:
                        case cv.CV_32FC2:
                        case cv.CV_32FC3:
                        case cv.CV_32FC4: {
                            mat.data32F.set(array);
                            break
                        }
                        case cv.CV_64F:
                        case cv.CV_64FC1:
                        case cv.CV_64FC2:
                        case cv.CV_64FC3:
                        case cv.CV_64FC4: {
                            mat.data64F.set(array);
                            break
                        }
                        default: {
                            throw new Error("Type is unsupported")
                        }
                    }
                    return mat
                };
                Module["matFromImageData"] = function (imageData) {
                    var mat = new cv.Mat(imageData.height, imageData.width, cv.CV_8UC4);
                    mat.data.set(imageData.data);
                    return mat
                };


                return cv;
            }
        );
    })();
    return cv(global);
}));
