export function flatten(arr: number[] | number[][]) {
    return arr.toString().split(',').map(item => +item);
}

export function int(num: number) {
    return num > 0 ? Math.floor(num) : Math.ceil(num);
}

export function clip(data: number, min: number, max: number) {
    return data < min ? min : data > max ? max : data;
}
