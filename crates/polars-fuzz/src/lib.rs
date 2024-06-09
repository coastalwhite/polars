use polars::prelude::*;

mod select_modulo;
#[macro_use]
mod bitset;
mod exprs;
mod dsl;

type EntropySrc = rand::rngs::ThreadRng;

struct Context {
    input: Node,
    expr: ExprContext,
}

struct ExprContext {
    max_size: u32,
}
