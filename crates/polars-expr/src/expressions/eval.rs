use std::borrow::Cow;
use std::cell::LazyCell;
use std::sync::Arc;

use arrow::bitmap::{Bitmap, BitmapBuilder};
use polars_core::chunked_array::builder::{AnonymousOwnedListBuilder, get_list_builder};
use polars_core::error::{PolarsResult, feature_gated};
use polars_core::frame::DataFrame;
#[cfg(feature = "dtype-array")]
use polars_core::prelude::ArrayChunked;
use polars_core::prelude::{
    BooleanChunked, ChunkCast, ChunkExplode, Column, Field, GroupPositions, GroupsType, IdxCa,
    IntoColumn, ListBuilderTrait, ListChunked, PlHashMap,
};
use polars_core::schema::Schema;
use polars_core::series::Series;
use polars_plan::constants::PL_ELEMENT_NAME;
use polars_plan::dsl::{EvalVariant, Expr};
use polars_utils::IdxSize;
use polars_utils::pl_str::PlSmallStr;

use super::{AggregationContext, PhysicalExpr};
use crate::prelude::{AggState, UpdateGroups};
use crate::state::ExecutionState;

#[derive(Clone)]
pub struct EvalExpr {
    input: Arc<dyn PhysicalExpr>,
    evaluation: Arc<dyn PhysicalExpr>,
    variant: EvalVariant,
    expr: Expr,
    output_field: Field,
    is_scalar: bool,
    evaluation_is_scalar: bool,
    evaluation_is_elementwise: bool,
    evaluation_is_fallible: bool,
    non_element_columns: Arc<[PlSmallStr]>,
}

impl EvalExpr {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(
        input: Arc<dyn PhysicalExpr>,
        evaluation: Arc<dyn PhysicalExpr>,
        variant: EvalVariant,
        expr: Expr,
        output_field: Field,
        is_scalar: bool,
        evaluation_is_scalar: bool,
        evaluation_is_elementwise: bool,
        evaluation_is_fallible: bool,
        non_element_columns: Arc<[PlSmallStr]>,
    ) -> Self {
        Self {
            input,
            evaluation,
            variant,
            expr,
            output_field,
            is_scalar,
            evaluation_is_scalar,
            evaluation_is_elementwise,
            evaluation_is_fallible,
            non_element_columns,
        }
    }

    fn prepare_state_for_listarr_eval<'a>(
        &self,
        df: &DataFrame,
        state: &'a ExecutionState,
        validity: Option<&Bitmap>,
    ) -> Cow<'a, ExecutionState> {
        let mut state = Cow::Borrowed(state);
        if !self.non_element_columns.is_empty() {
            let validity =
                validity.map(|v| BooleanChunked::from_bitmap(PlSmallStr::EMPTY, v.clone()));
            state.to_mut().ext_named_groups = Arc::new(PlHashMap::from_iter(
                self.non_element_columns.iter().map(|c| {
                    let mut values = df.column(c).unwrap().clone();
                    if let Some(validity) = validity.as_ref() {
                        values = values.filter(validity).unwrap();
                    }
                    (c.clone(), AggState::AggregatedScalar(values))
                }),
            ));
        }
        state
    }

    fn evaluate_on_list_chunked(
        &self,
        ext_df: &DataFrame,
        ca: &ListChunked,
        state: &ExecutionState,
        is_agg: bool,
    ) -> PolarsResult<Column> {
        let df = ca
            .get_inner()
            .with_name(PL_ELEMENT_NAME.clone())
            .into_frame();

        // Fast path: Empty or only nulls.
        if ca.null_count() == ca.len() {
            let name = self.output_field.name.clone();
            return Ok(Column::full_null(name, ca.len(), self.output_field.dtype()));
        }

        let has_masked_out_values = LazyCell::new(|| ca.has_masked_out_values());
        let may_fail_on_masked_out_elements = self.evaluation_is_fallible && *has_masked_out_values;

        // Fast path: fully elementwise expression without masked out values.
        if self.evaluation_is_elementwise
            && self.non_element_columns.is_empty()
            && !may_fail_on_masked_out_elements
        {
            let mut column = self.evaluation.evaluate(&df, state)?;

            // Since `lit` is marked as elementwise, this may lead to problems.
            if column.len() == 1 && df.height() != 1 {
                column = column.new_from_index(0, df.height());
            }

            if !is_agg || !self.evaluation_is_scalar {
                column = ca
                    .with_inner_values(column.as_materialized_series())
                    .into_column();
            }

            return Ok(column);
        }

        let validity = ca.rechunk_validity();
        let offsets = ca.offsets()?;

        // Create groups for all valid array elements.
        let groups = if ca.has_nulls() {
            let validity = validity.as_ref().unwrap();
            offsets
                .offset_and_length_iter()
                .zip(validity.iter())
                .filter_map(|((offset, length), validity)| {
                    validity.then_some([offset as IdxSize, length as IdxSize])
                })
                .collect()
        } else {
            offsets
                .offset_and_length_iter()
                .map(|(offset, length)| [offset as IdxSize, length as IdxSize])
                .collect()
        };
        let groups = GroupsType::Slice {
            groups,
            overlapping: false,
        };
        let groups = Cow::Owned(groups.into_sliceable());

        let state = self.prepare_state_for_listarr_eval(ext_df, state, validity.as_ref());
        let mut ac = self
            .evaluation
            .evaluate_on_groups(&df, &groups, state.as_ref())?;

        ac.groups(); // Update the groups.

        let flat_naive = ac.flat_naive();

        // Fast path. Groups are pointing to the same offsets in the data buffer.
        if flat_naive.len() == df.height()
            && let Some(output_groups) = ac.groups.as_ref().as_unrolled_slice()
            && !(is_agg && self.evaluation_is_scalar)
        {
            let groups_are_unchanged = if let Some(validity) = &validity {
                assert_eq!(validity.set_bits(), output_groups.len());
                validity
                    .true_idx_iter()
                    .zip(output_groups)
                    .all(|(j, [start, len])| {
                        let (original_start, original_end) =
                            unsafe { offsets.start_end_unchecked(j) };
                        (*start == original_start as IdxSize)
                            & (*len == (original_end - original_start) as IdxSize)
                    })
            } else {
                output_groups
                    .iter()
                    .zip(offsets.offset_and_length_iter())
                    .all(|([start, len], (original_start, original_len))| {
                        (*start == original_start as IdxSize) & (*len == original_len as IdxSize)
                    })
            };

            if groups_are_unchanged {
                let values = flat_naive.as_materialized_series();
                return Ok(ca.with_inner_values(values).into_column());
            }
        }

        // Slow path. Groups have changed, so we need to gather data again.
        if is_agg && self.evaluation_is_scalar {
            let mut values = ac.finalize();

            // We didn't have any groups for the `null` values so we have to reinsert them.
            if let Some(validity) = validity {
                values = values.deposit(&validity);
            }

            Ok(values)
        } else {
            let mut ca = ac.aggregated_as_list();

            // We didn't have any groups for the `null` values so we have to reinsert them.
            if let Some(validity) = validity {
                ca = Cow::Owned(ca.deposit(&validity));
            }

            Ok(ca.into_owned().into_column())
        }
    }

    #[cfg(feature = "dtype-array")]
    fn evaluate_on_array_chunked(
        &self,
        ext_df: &DataFrame,
        ca: &ArrayChunked,
        state: &ExecutionState,
        as_list: bool,
        is_agg: bool,
    ) -> PolarsResult<Column> {
        let df = ca
            .get_inner()
            .with_name(PL_ELEMENT_NAME.clone())
            .into_frame();

        // Fast path: Empty or only nulls.
        if ca.null_count() == ca.len() {
            let name = self.output_field.name.clone();
            return Ok(Column::full_null(name, ca.len(), self.output_field.dtype()));
        }

        let validity = ca.rechunk_validity();
        let may_fail_on_masked_out_elements = self.evaluation_is_fallible && ca.has_nulls();

        // Fast path: fully elementwise expression without masked out values.
        if self.evaluation_is_elementwise
            && self.non_element_columns.is_empty()
            && !may_fail_on_masked_out_elements
        {
            assert!(!self.evaluation_is_scalar);

            let mut column = self.evaluation.evaluate(&df, state)?;
            if column.len() == 1 && df.height() != 1 {
                column = column.new_from_index(0, df.height());
            }
            assert_eq!(column.len(), ca.len() * ca.width());

            let dtype = column.dtype().clone();
            let mut out = ArrayChunked::from_aligned_values(
                self.output_field.name.clone(),
                &dtype,
                ca.width(),
                column.take_materialized_series().into_chunks(),
                ca.len(),
            );

            if let Some(validity) = validity {
                out.set_validity(&validity);
            }

            return Ok(if as_list {
                out.to_list().into_column()
            } else {
                out.clone().into_column()
            });
        }

        // Create groups for all valid array elements.
        let groups = if ca.has_nulls() {
            let validity = validity.as_ref().unwrap();
            (0..ca.len())
                .filter(|i| unsafe { validity.get_bit_unchecked(*i) })
                .map(|i| [(i * ca.width()) as IdxSize, ca.width() as IdxSize])
                .collect()
        } else {
            (0..ca.len())
                .map(|i| [(i * ca.width()) as IdxSize, ca.width() as IdxSize])
                .collect()
        };
        let groups = GroupsType::Slice {
            groups,
            overlapping: false,
        };
        let groups = Cow::Owned(groups.into_sliceable());

        let state = self.prepare_state_for_listarr_eval(ext_df, state, validity.as_ref());
        let mut ac = self
            .evaluation
            .evaluate_on_groups(&df, &groups, state.as_ref())?;

        ac.groups(); // Update the groups.

        let flat_naive = ac.flat_naive();

        // Fast path. Groups are pointing to the same offsets in the data buffer.
        if flat_naive.len() == ca.len() * ca.width()
            && let Some(output_groups) = ac.groups.as_ref().as_unrolled_slice()
            && !(is_agg && self.evaluation_is_scalar)
        {
            let ca_width = ca.width() as IdxSize;
            let groups_are_unchanged = if let Some(validity) = &validity {
                assert_eq!(validity.set_bits(), output_groups.len());
                validity
                    .true_idx_iter()
                    .zip(output_groups)
                    .all(|(j, [start, len])| {
                        (*start == j as IdxSize * ca_width) & (*len == ca_width)
                    })
            } else {
                use polars_utils::itertools::Itertools;

                output_groups
                    .iter()
                    .enumerate_idx()
                    .all(|(i, [start, len])| (*start == i * ca_width) & (*len == ca_width))
            };

            if groups_are_unchanged {
                let values = flat_naive;
                let dtype = values.dtype().clone();
                let mut out = ArrayChunked::from_aligned_values(
                    self.output_field.name.clone(),
                    &dtype,
                    ca.width(),
                    values.as_materialized_series().chunks().clone(),
                    ca.len(),
                );

                if let Some(validity) = validity {
                    out.set_validity(&validity);
                }

                return Ok(if as_list {
                    out.to_list().into_column()
                } else {
                    out.into_column()
                });
            }
        }

        // Slow path. Groups have changed, so we need to gather data again.
        if is_agg && self.evaluation_is_scalar {
            let mut values = ac.finalize();

            // We didn't have any groups for the `null` values so we have to reinsert them.
            if let Some(validity) = validity {
                values = values.deposit(&validity);
            }

            Ok(values)
        } else {
            let mut ca = ac.aggregated_as_list();

            // We didn't have any groups for the `null` values so we have to reinsert them.
            if let Some(validity) = validity {
                ca = Cow::Owned(ca.deposit(&validity));
            }

            Ok(if as_list {
                ca.into_owned().into_column()
            } else {
                ca.cast(self.output_field.dtype()).unwrap().into_column()
            })
        }
    }

    fn evaluate_cumulative_eval(
        &self,
        df: &DataFrame,
        input: &Series,
        min_samples: usize,
        state: &ExecutionState,
    ) -> PolarsResult<Column> {
        if input.is_empty() {
            return Ok(Column::new_empty(
                self.output_field.name().clone(),
                self.output_field.dtype(),
            ));
        }

        let mut deposit: Option<Bitmap> = None;
        let groups = if min_samples == 0 {
            (1..input.len() as IdxSize).map(|i| [0, i]).collect()
        } else {
            let validity = input
                .rechunk_validity()
                .unwrap_or_else(|| Bitmap::new_with_value(true, input.len()));
            let mut count = 0;
            let mut deposit_builder = BitmapBuilder::with_capacity(input.len());
            let out = (0..input.len() as IdxSize)
                .filter(|i| {
                    count += usize::from(unsafe { validity.get_bit_unchecked(*i as usize) });
                    let is_selected = count >= min_samples;
                    unsafe { deposit_builder.push_unchecked(is_selected) };
                    is_selected
                })
                .map(|i| [0, i + 1])
                .collect();
            deposit = Some(deposit_builder.freeze());
            out
        };

        let groups = GroupsType::Slice {
            groups,
            overlapping: true,
        };

        let groups = groups.into_sliceable();

        let mut df = df.clone();
        df.with_column(
            input
                .clone()
                .with_name(PL_ELEMENT_NAME.clone())
                .into_column(),
        )?;
        let agg = self.evaluation.evaluate_on_groups(&df, &groups, state)?;
        let (mut out, _) = agg.get_final_aggregation();

        // Since we only evaluated the expressions on the items that satisfied the min samples, we
        // need to fix it up here again.
        if let Some(deposit) = deposit {
            let mut i = 0;
            let gather_idxs = deposit
                .iter()
                .map(|v| {
                    let out = i;
                    i += IdxSize::from(v);
                    out
                })
                .collect::<Vec<IdxSize>>();
            let gather_idxs =
                IdxCa::from_vec_validity(PlSmallStr::EMPTY, gather_idxs, Some(deposit));
            out = unsafe { out.take_unchecked(&gather_idxs) };
        }

        Ok(out)
    }

    fn evaluate_one(
        &self,
        df: &DataFrame,
        input: &Series,
        state: &ExecutionState,
    ) -> PolarsResult<Column> {
        match self.variant {
            EvalVariant::List => {
                let lst = input.list()?;
                self.evaluate_on_list_chunked(df, lst, state, false)
            },
            EvalVariant::ListAgg => {
                let lst = input.list()?;
                self.evaluate_on_list_chunked(df, lst, state, true)
            },
            EvalVariant::Array { as_list } => feature_gated!("dtype-array", {
                self.evaluate_on_array_chunked(df, input.array()?, state, as_list, false)
            }),
            EvalVariant::ArrayAgg => feature_gated!("dtype-array", {
                self.evaluate_on_array_chunked(df, input.array()?, state, true, true)
            }),
            EvalVariant::Cumulative { min_samples } => {
                self.evaluate_cumulative_eval(df, input, min_samples, state)
            },
        }
    }

    fn evaluate_on_groups_with_non_element_columns<'a>(
        &self,
        df: &DataFrame,
        groups: &'a GroupPositions,
        state: &ExecutionState,
    ) -> PolarsResult<AggregationContext<'a>> {
        // @NOTE: This is a bit of a fallback implementation and is probably quite slow.

        let mut ac = self.input.evaluate_on_groups(df, groups, state)?;

        let groups = ac.groups();
        if groups.len() == 0 {
            return Ok(AggregationContext {
                state: AggState::AggregatedList(Column::new_empty(
                    self.output_field.name().clone(),
                    &self.output_field.dtype().clone().implode(),
                )),
                groups: groups.clone(),
                update_groups: UpdateGroups::No,
                original_len: false,
            });
        }

        let mut containers = Vec::with_capacity(1 + self.non_element_columns.len());

        let eval = |s: Option<AmortSeries>, length: usize, containers: &mut Vec<Column>| {
            let df = unsafe { DataFrame::new_no_checks(length, std::mem::take(containers)) };
            let out = self
                .evaluate_one(&df, s.as_ref().unwrap().as_ref(), state)?
                .take_materialized_series();
            *containers = df.take_columns();
            PolarsResult::Ok(out)
        };

        let mut builder = get_list_builder(
            self.output_field.dtype(),
            ac.groups.total_num_elements(),
            ac.groups.len(),
            self.output_field.name().clone(),
        );
        let ca = ac.aggregated_as_list().into_owned();
        match &**ac.groups.as_ref() {
            GroupsType::Idx(idxs) => {
                for (s, (_, i)) in ca.amortized_iter().zip(idxs.iter()) {
                    containers.clear();
                    containers.extend(self.non_element_columns.iter().map(|c| unsafe {
                        df.column(c.as_str()).unwrap().take_slice_unchecked(&i)
                    }));
                    builder.append_series(&eval(s, i.len(), &mut containers)?)?;
                }
            },
            GroupsType::Slice {
                groups,
                overlapping: _,
            } => {
                for (s, [start, length]) in ca.amortized_iter().zip(groups.iter()) {
                    containers.clear();
                    containers.extend(self.non_element_columns.iter().map(|c| {
                        df.column(c.as_str())
                            .unwrap()
                            .slice(*start as i64, *length as usize)
                    }));
                    builder.append_series(&eval(s, *length as usize, &mut containers)?)?;
                }
            },
        }

        ac.state = AggState::AggregatedList(builder.finish().into_column());
        return Ok(ac);
    }
}

impl PhysicalExpr for EvalExpr {
    fn as_expression(&self) -> Option<&Expr> {
        Some(&self.expr)
    }

    fn evaluate(&self, df: &DataFrame, state: &ExecutionState) -> PolarsResult<Column> {
        let input = self.input.evaluate(df, state)?;
        self.evaluate_one(df, input.as_materialized_series(), state)
    }

    fn evaluate_on_groups<'a>(
        &self,
        df: &DataFrame,
        groups: &'a GroupPositions,
        state: &ExecutionState,
    ) -> PolarsResult<AggregationContext<'a>> {
        if self.non_element_columns.is_empty() {
            let mut input = self.input.evaluate_on_groups(df, groups, state)?;
            match self.variant {
                EvalVariant::List => {
                    let out = self.evaluate_on_list_chunked(
                        df,
                        input.get_values().list()?,
                        state,
                        false,
                    )?;
                    input.with_values(out, false, Some(&self.expr))?;
                },
                EvalVariant::ListAgg => {
                    let out =
                        self.evaluate_on_list_chunked(df, input.get_values().list()?, state, true)?;
                    input.with_values(out, false, Some(&self.expr))?;
                },
                EvalVariant::Array { as_list } => feature_gated!("dtype-array", {
                    let out = self.evaluate_on_array_chunked(
                        df,
                        input.aggregated().array()?,
                        state,
                        as_list,
                        false,
                    )?;
                    input.with_values(out, true, Some(&self.expr))?;
                }),
                EvalVariant::ArrayAgg => feature_gated!("dtype-array", {
                    let out = self.evaluate_on_array_chunked(
                        df,
                        input.aggregated().array()?,
                        state,
                        true,
                        true,
                    )?;
                    input.with_values(out, true, Some(&self.expr))?;
                }),
                EvalVariant::Cumulative { min_samples } => {
                    let mut builder = AnonymousOwnedListBuilder::new(
                        self.output_field.name().clone(),
                        input.groups().len(),
                        Some(self.output_field.dtype.clone()),
                    );
                    for group in input.iter_groups(false) {
                        match group {
                            None => {},
                            Some(group) => {
                                let out = self.evaluate_cumulative_eval(
                                    df,
                                    group.as_ref(),
                                    min_samples,
                                    state,
                                )?;
                                builder.append_series(out.as_materialized_series())?;
                            },
                        }
                    }

                    input.with_values(builder.finish().into_column(), true, Some(&self.expr))?;
                },
            }
            Ok(input)
        } else {
            self.evaluate_on_groups_with_non_element_columns(df, groups, state)
        }
    }

    fn to_field(&self, _input_schema: &Schema) -> PolarsResult<Field> {
        Ok(self.output_field.clone())
    }

    fn is_scalar(&self) -> bool {
        self.is_scalar
    }
}
