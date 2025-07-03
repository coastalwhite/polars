use polars_core::prelude::*;
use polars_tpg::dot::TpgDot;

use crate::prelude::*;

impl LazyFrame {
    /// Get a dot language representation of the LogicalPlan.
    pub fn to_dot(&self, optimized: bool) -> PolarsResult<String> {
        let lp = if optimized {
            self.clone().to_alp_optimized()
        } else {
            self.clone().to_alp()
        }?;

        let tpg = lp.to_tpg();
        Ok(TpgDot {
            plan: &tpg,
        }
        .to_string())
    }

    /// Get a dot language representation of the streaming physical plan.
    #[cfg(feature = "new_streaming")]
    pub fn to_dot_streaming_phys(&self, optimized: bool) -> PolarsResult<String> {
        let lf = self.clone().with_new_streaming(true);
        let mut lp = if optimized {
            lf.to_alp_optimized()
        } else {
            lf.to_alp()
        }?;
        polars_stream::visualize_physical_plan(lp.lp_top, &mut lp.lp_arena, &mut lp.expr_arena)
    }
}
