/**
 * @file add deal origin op
 */

import { Transformer, env, interfaces } from '@paddlejs/paddlejs-core';

const IMG_ORIGIN = 'image';
const FINAL_PACK_OP_NAME = 'fetch_pack';
const DEFAULT_WIDTH = 500;
const DEFAULT_HEIGHT = 280;

export default class DealOrigin extends Transformer {
    private width;
    private height;

    constructor(width?: number, height?: number) {
        super('DealOrigin');
        this.width = width || DEFAULT_WIDTH;
        this.height = height || DEFAULT_HEIGHT;
    }

    transform(...args: any) {
        if (!env.get('webgl_gpu_pipeline')) {
            return;
        }
        const [ops, vars] = args;
        const fetchOp = ops.find(item => item.type === 'fetch');
        const [inputName] = fetchOp.inputs.X;

        const segImgOp = {
            attrs: {},
            inputs: {
                X: [inputName],
                Y: [IMG_ORIGIN]
            },
            outputs: {
                Out: [FINAL_PACK_OP_NAME]
            },
            type: 'segImg',
            isPacked: true,
            bufferType: interfaces.BufferType.ColorBuffer,
            uniform: {
                type: {
                    type: '1i',
                    value: 0
                }
            }
        };

        const packOutVar = {
            name: FINAL_PACK_OP_NAME,
            shape: [1, 1, this.height, this.width],
            persistable: false
        };

        fetchOp.inputs.X = [FINAL_PACK_OP_NAME];
        ops.push(segImgOp);
        if (vars instanceof Array) {
            vars.push(...[packOutVar]);
        }
        else {
            vars[FINAL_PACK_OP_NAME] = packOutVar;
        }
    }
}
