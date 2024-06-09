use polars::prelude::*;
use rand::Rng;

use crate::select_modulo::SelectModulo;
use crate::{Context, EntropySrc};
bitset! {
    pub struct DslSet, pub enum DslVariant: u64 {
        WithColumns,
    }
}

fn arbitrary_dsl(entropy: &mut EntropySrc, ctx: &Context, enabled: DslSet) -> Result<DslPlan, ()> {
    let variant = enabled.select_modulo(entropy.gen()).ok_or(())?;

    match variant {
        DslVariant::WithColumns => {
            let options = ProjectionOptions {
                run_parallel: false,
                duplicate_check: false,
                should_broadcast: false,
            };

            Ok(DslPlan::HStack {
                input: ctx.input,
                exprs: vec![],
                options,
            })
        },
    }
}
