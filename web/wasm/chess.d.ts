/* tslint:disable */
/* eslint-disable */

export function initThreadPool(num_threads: number): Promise<any>;

export class wbg_rayon_PoolBuilder {
    private constructor();
    free(): void;
    [Symbol.dispose](): void;
    build(): void;
    mainJS(): string;
    numThreads(): number;
    receiver(): number;
}

export function wbg_rayon_start_worker(receiver: number): void;

export type InitInput = RequestInfo | URL | Response | BufferSource | WebAssembly.Module;

export interface InitOutput {
    readonly main: (a: number, b: number) => number;
    readonly __wbg_wbg_rayon_poolbuilder_free: (a: number, b: number) => void;
    readonly initThreadPool: (a: number) => number;
    readonly wbg_rayon_poolbuilder_build: (a: number) => void;
    readonly wbg_rayon_poolbuilder_mainJS: (a: number) => number;
    readonly wbg_rayon_poolbuilder_numThreads: (a: number) => number;
    readonly wbg_rayon_poolbuilder_receiver: (a: number) => number;
    readonly wbg_rayon_start_worker: (a: number) => void;
    readonly __wasm_bindgen_func_elem_3512: (a: number, b: number) => void;
    readonly __wasm_bindgen_func_elem_221784: (a: number, b: number) => void;
    readonly __wasm_bindgen_func_elem_276274: (a: number, b: number) => void;
    readonly __wasm_bindgen_func_elem_6270: (a: number, b: number) => void;
    readonly __wasm_bindgen_func_elem_130901: (a: number, b: number) => void;
    readonly __wasm_bindgen_func_elem_279079: (a: number, b: number, c: number, d: number) => void;
    readonly __wasm_bindgen_func_elem_7195: (a: number, b: number, c: number, d: number) => void;
    readonly __wasm_bindgen_func_elem_279085: (a: number, b: number, c: number, d: number) => void;
    readonly __wasm_bindgen_func_elem_224929: (a: number, b: number, c: number) => void;
    readonly __wasm_bindgen_func_elem_276214: (a: number, b: number, c: number) => void;
    readonly __wasm_bindgen_func_elem_7193: (a: number, b: number, c: number) => void;
    readonly __wasm_bindgen_func_elem_7193_6: (a: number, b: number, c: number) => void;
    readonly __wasm_bindgen_func_elem_7193_7: (a: number, b: number, c: number) => void;
    readonly __wasm_bindgen_func_elem_7193_8: (a: number, b: number, c: number) => void;
    readonly __wasm_bindgen_func_elem_7193_9: (a: number, b: number, c: number) => void;
    readonly __wasm_bindgen_func_elem_7193_10: (a: number, b: number, c: number) => void;
    readonly __wasm_bindgen_func_elem_7193_11: (a: number, b: number, c: number) => void;
    readonly __wasm_bindgen_func_elem_7193_12: (a: number, b: number, c: number) => void;
    readonly __wasm_bindgen_func_elem_7194: (a: number, b: number, c: number) => void;
    readonly __wasm_bindgen_func_elem_3491: (a: number, b: number) => void;
    readonly __wasm_bindgen_func_elem_7196: (a: number, b: number) => void;
    readonly __wasm_bindgen_func_elem_131004: (a: number, b: number) => void;
    readonly memory: WebAssembly.Memory;
    readonly __wbindgen_export: (a: number, b: number) => number;
    readonly __wbindgen_export2: (a: number, b: number, c: number, d: number) => number;
    readonly __wbindgen_export3: (a: number) => void;
    readonly __wbindgen_export4: (a: number, b: number, c: number) => void;
    readonly __wbindgen_add_to_stack_pointer: (a: number) => number;
    readonly __wbindgen_thread_destroy: (a?: number, b?: number, c?: number) => void;
    readonly __wbindgen_start: (a: number) => void;
}

export type SyncInitInput = BufferSource | WebAssembly.Module;

/**
 * Instantiates the given `module`, which can either be bytes or
 * a precompiled `WebAssembly.Module`.
 *
 * @param {{ module: SyncInitInput, memory?: WebAssembly.Memory, thread_stack_size?: number }} module - Passing `SyncInitInput` directly is deprecated.
 * @param {WebAssembly.Memory} memory - Deprecated.
 *
 * @returns {InitOutput}
 */
export function initSync(module: { module: SyncInitInput, memory?: WebAssembly.Memory, thread_stack_size?: number } | SyncInitInput, memory?: WebAssembly.Memory): InitOutput;

/**
 * If `module_or_path` is {RequestInfo} or {URL}, makes a request and
 * for everything else, calls `WebAssembly.instantiate` directly.
 *
 * @param {{ module_or_path: InitInput | Promise<InitInput>, memory?: WebAssembly.Memory, thread_stack_size?: number }} module_or_path - Passing `InitInput` directly is deprecated.
 * @param {WebAssembly.Memory} memory - Deprecated.
 *
 * @returns {Promise<InitOutput>}
 */
export default function __wbg_init (module_or_path?: { module_or_path: InitInput | Promise<InitInput>, memory?: WebAssembly.Memory, thread_stack_size?: number } | InitInput | Promise<InitInput>, memory?: WebAssembly.Memory): Promise<InitOutput>;
