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
    readonly __wasm_bindgen_func_elem_3738: (a: number, b: number) => void;
    readonly __wasm_bindgen_func_elem_222020: (a: number, b: number) => void;
    readonly __wasm_bindgen_func_elem_276511: (a: number, b: number) => void;
    readonly __wasm_bindgen_func_elem_6506: (a: number, b: number) => void;
    readonly __wasm_bindgen_func_elem_131137: (a: number, b: number) => void;
    readonly __wasm_bindgen_func_elem_279316: (a: number, b: number, c: number, d: number) => void;
    readonly __wasm_bindgen_func_elem_7431: (a: number, b: number, c: number, d: number) => void;
    readonly __wasm_bindgen_func_elem_279322: (a: number, b: number, c: number, d: number) => void;
    readonly __wasm_bindgen_func_elem_225165: (a: number, b: number, c: number) => void;
    readonly __wasm_bindgen_func_elem_276451: (a: number, b: number, c: number) => void;
    readonly __wasm_bindgen_func_elem_7429: (a: number, b: number, c: number) => void;
    readonly __wasm_bindgen_func_elem_7429_6: (a: number, b: number, c: number) => void;
    readonly __wasm_bindgen_func_elem_7429_7: (a: number, b: number, c: number) => void;
    readonly __wasm_bindgen_func_elem_7429_8: (a: number, b: number, c: number) => void;
    readonly __wasm_bindgen_func_elem_7429_9: (a: number, b: number, c: number) => void;
    readonly __wasm_bindgen_func_elem_7429_10: (a: number, b: number, c: number) => void;
    readonly __wasm_bindgen_func_elem_7429_11: (a: number, b: number, c: number) => void;
    readonly __wasm_bindgen_func_elem_7429_12: (a: number, b: number, c: number) => void;
    readonly __wasm_bindgen_func_elem_7430: (a: number, b: number, c: number) => void;
    readonly __wasm_bindgen_func_elem_3717: (a: number, b: number) => void;
    readonly __wasm_bindgen_func_elem_7432: (a: number, b: number) => void;
    readonly __wasm_bindgen_func_elem_131240: (a: number, b: number) => void;
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
