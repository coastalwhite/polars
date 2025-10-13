use std::borrow::Cow;
use std::cell::RefCell;
use std::sync::Arc;

use polars_compute::unique::amortized_unique_from_dtype;
use polars_core::POOL;
use polars_core::error::PolarsResult;
use polars_core::frame::DataFrame;
use polars_core::prelude::{CompatLevel, GroupPositions, GroupsType, IntoColumn};
use polars_utils::UnitVec;
use rayon::iter::{IntoParallelIterator, ParallelIterator};
use rayon::slice::ParallelSlice;

use crate::prelude::{AggState, AggregationContext, PhysicalExpr, UpdateGroups};
use crate::state::ExecutionState;

pub fn reverse<'a>(
    inputs: &[Arc<dyn PhysicalExpr>],
    df: &DataFrame,
    groups: &'a GroupPositions,
    state: &ExecutionState,
) -> PolarsResult<AggregationContext<'a>> {
    assert_eq!(inputs.len(), 1);

    let mut ac = inputs[0].evaluate_on_groups(df, groups, state)?;

    // Length preserving operation on scalars keeps scalar.
    if let AggState::AggregatedScalar(_) | AggState::LiteralScalar(_) = &ac.state {
        return Ok(ac);
    }

    POOL.install(|| {
        ac.groups = Cow::Owned(
            GroupsType::Idx(match &**ac.groups().as_ref() {
                GroupsType::Idx(idx) => idx
                    .into_par_iter()
                    .map(|(first, idx)| {
                        (
                            idx.last().copied().unwrap_or(first),
                            idx.iter().copied().rev().collect(),
                        )
                    })
                    .collect(),
                GroupsType::Slice {
                    groups,
                    overlapping: _,
                } => groups
                    .into_par_iter()
                    .map(|[start, len]| {
                        (
                            start + len.saturating_sub(1),
                            (*start..*start + *len).rev().collect(),
                        )
                    })
                    .collect(),
            })
            .into_sliceable(),
        );
    });

    Ok(ac)
}

pub fn unique<'a>(
    inputs: &[Arc<dyn PhysicalExpr>],
    df: &DataFrame,
    groups: &'a GroupPositions,
    state: &ExecutionState,
) -> PolarsResult<AggregationContext<'a>> {
    assert_eq!(inputs.len(), 1);

    let mut ac = inputs[0].evaluate_on_groups(df, groups, state)?;

    if let AggState::LiteralScalar(c) | AggState::AggregatedScalar(c) = &ac.state {
        let mut c = c.as_list().into_column();
        if matches!(ac.state, AggState::LiteralScalar(_)) {
            c = c.new_from_index(0, ac.groups.len());
        }
        ac.state = AggState::AggregatedList(c);
        ac.with_update_groups(UpdateGroups::WithSeriesLen);
        return Ok(ac);
    }

    let mut values = ac.get_values().clone();
    values = values.to_physical_repr();
    if let Some(v) = values.try_str() {
        values = v.as_binary().into_column();
    }

    let array = values.rechunk_to_arrow(CompatLevel::newest());

    POOL.install(|| {
        let state = amortized_unique_from_dtype(array.dtype());
        let mut states = (0..POOL.current_num_threads())
            .map(|_| state.new_empty())
            .collect::<Vec<_>>();

        let states_ptr = states.as_mut_slice();

        ac.groups = Cow::Owned(
            GroupsType::Idx(match &**ac.groups().as_ref() {
                GroupsType::Idx(idx) => idx
                    .clone()
                    .into_par_iter()
                    .map(|(first, mut idxs)| {
                        let thread_idx = POOL.current_thread_index().unwrap();
                        let state = unsafe { &mut *states_ptr.as_mut_ptr().add(thread_idx) };
                        state.retain_unique(array.as_ref(), &mut idxs);
                        (idxs.first().copied().unwrap_or(first), idxs)
                    })
                    .collect(),
                GroupsType::Slice {
                    groups,
                    overlapping: _,
                } => groups
                    .into_par_iter()
                    .map(|[start, len]| {
                        let thread_idx = POOL.current_thread_index().unwrap();
                        let state = unsafe { &mut *states_ptr.as_mut_ptr().add(thread_idx) };
                        let mut idxs = UnitVec::new();
                        state.arg_unique(array.as_ref(), &mut idxs, *start, *len);
                        (idxs.first().copied().unwrap_or(*start), idxs)
                    })
                    .collect(),
            })
            .into_sliceable(),
        );
    });

    Ok(ac)
}
