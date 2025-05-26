use super::*;
use crate::plans::conversion::functions::convert_functions;

pub enum ConversionOutput<T> {
    Normal(T),
    Selector(Vec<T>),
}

impl<T> ConversionOutput<T> {
    pub fn map<U>(self, f: impl Fn(T) -> U) -> ConversionOutput<U> {
        match self {
            Self::Normal(v) => ConversionOutput::Normal(f(v)),
            Self::Selector(v) => ConversionOutput::Selector(v.into_iter().map(f).collect()),
        }
    }

    pub fn try_map<U>(self, f: impl Fn(T) -> PolarsResult<U>) -> PolarsResult<ConversionOutput<U>> {
        match self {
            Self::Normal(v) => Ok(ConversionOutput::Normal(f(v)?)),
            Self::Selector(v) => Ok(ConversionOutput::Selector(v.into_iter().map(f).collect()?)),
        }
    }

    pub fn combine<U>(self, other: Self, f: impl Fn(T, T) -> U) -> ConversionOutput<U> {
        Self::combine_vec(vec![self, other], |vs| f(vs[0], vs[1]))
    }

    pub fn combine_vec<U>(items: Vec<Self>, f: impl Fn(Vec<T>) -> U) -> ConversionOutput<U> {
        todo!()
    }
}

impl ConversionOutput<Node> {
    pub fn into_expr_ir(self, output_name: OutputName) -> ConversionOutput<ExprIR> {
        match self {
            Self::Normal(v) => ConversionOutput::Normal(ExprIR::new(v, output_name)),
            Self::Selector(items) => ConversionOutput::Selector(
                items
                    .into_iter()
                    .map(|v| ExprIR::new(v, output_name.clone()))
                    .collect(),
            ),
        }
    }
}

pub fn to_expr_ir(expr: Expr, arena: &mut Arena<AExpr>) -> PolarsResult<ConversionOutput<ExprIR>> {
    let mut state = ConversionContext::new();
    let node = to_aexpr_impl(expr, arena, &mut state)?;
    Ok(node.into_expr_ir(state.output_name))
}

pub(super) fn to_expr_irs(
    input: Vec<Expr>,
    arena: &mut Arena<AExpr>,
) -> PolarsResult<Vec<ConversionOutput<ExprIR>>> {
    input.into_iter().map(|e| to_expr_ir(e, arena)).collect()
}

pub fn to_expr_ir_ignore_alias(
    expr: Expr,
    arena: &mut Arena<AExpr>,
) -> PolarsResult<ConversionOutput<ExprIR>> {
    let mut state = ConversionContext::new();
    state.ignore_alias = true;
    let node = to_aexpr_impl_materialized_lit(expr, arena, &mut state)?;
    Ok(node.into_expr_ir(state.output_name))
}

pub(super) fn to_expr_irs_ignore_alias(
    input: Vec<Expr>,
    arena: &mut Arena<AExpr>,
) -> PolarsResult<Vec<ConversionOutput<ExprIR>>> {
    input
        .into_iter()
        .map(|e| to_expr_ir_ignore_alias(e, arena))
        .collect()
}

/// converts expression to AExpr and adds it to the arena, which uses an arena (Vec) for allocation
pub fn to_aexpr(expr: Expr, arena: &mut Arena<AExpr>) -> PolarsResult<ConversionOutput<Node>> {
    to_aexpr_impl_materialized_lit(
        expr,
        arena,
        &mut ConversionContext {
            prune_alias: false,
            ..Default::default()
        },
    )
}

#[derive(Default)]
pub(super) struct ConversionContext {
    pub(super) output_name: OutputName,
    /// Remove alias from the expressions and set as [`OutputName`].
    pub(super) prune_alias: bool,
    /// If an `alias` is encountered prune and ignore it.
    pub(super) ignore_alias: bool,
}

impl ConversionContext {
    fn new() -> Self {
        Self {
            prune_alias: true,
            ..Default::default()
        }
    }
}

fn to_aexprs(
    input: Vec<Expr>,
    arena: &mut Arena<AExpr>,
    state: &mut ConversionContext,
) -> PolarsResult<Vec<ConversionOutput<Node>>> {
    input
        .into_iter()
        .map(|e| to_aexpr_impl_materialized_lit(e, arena, state))
        .collect()
}

pub(super) fn set_function_output_name<F>(
    e: &[ExprIR],
    state: &mut ConversionContext,
    function_fmt: F,
) where
    F: FnOnce() -> PlSmallStr,
{
    if state.output_name.is_none() {
        if e.is_empty() {
            let s = function_fmt();
            state.output_name = OutputName::LiteralLhs(s);
        } else {
            state.output_name = e[0].output_name_inner().clone();
        }
    }
}

fn to_aexpr_impl_materialized_lit(
    expr: Expr,
    arena: &mut Arena<AExpr>,
    state: &mut ConversionContext,
) -> PolarsResult<ConversionOutput<Node>> {
    // Already convert `Lit Float and Lit Int` expressions that are not used in a binary / function expression.
    // This means they can be materialized immediately
    let e = match expr {
        Expr::Literal(lv @ LiteralValue::Dyn(_)) => Expr::Literal(lv.materialize()),
        Expr::Alias(inner, name) if matches!(&*inner, Expr::Literal(LiteralValue::Dyn(_))) => {
            let Expr::Literal(lv) = &*inner else {
                unreachable!()
            };
            Expr::Alias(Arc::new(Expr::Literal(lv.clone().materialize())), name)
        },
        e => e,
    };
    to_aexpr_impl(e, arena, state)
}

/// Converts expression to AExpr and adds it to the arena, which uses an arena (Vec) for allocation.
#[recursive]
pub(super) fn to_aexpr_impl(
    expr: Expr,
    arena: &mut Arena<AExpr>,
    state: &mut ConversionContext,
) -> PolarsResult<ConversionOutput<Node>> {
    let owned = Arc::unwrap_or_clone;
    Ok(match expr {
        Expr::Explode { input, skip_empty } => to_aexpr_impl(owned(input), arena, state)?
            .map(|expr| arena.add(AExpr::Explode { expr, skip_empty })),
        Expr::Alias(e, name) => {
            if state.prune_alias {
                if state.output_name.is_none() && !state.ignore_alias {
                    state.output_name = OutputName::Alias(name);
                }
                let _ = to_aexpr_impl(owned(e), arena, state)?;
                arena.pop().unwrap()
            } else {
                to_aexpr_impl(owned(e), arena, state)?.map(|e| arena.add(AExpr::Alias(e, name)))
            }
        },
        Expr::Literal(lv) => {
            if state.output_name.is_none() {
                state.output_name = OutputName::LiteralLhs(lv.output_column_name().clone());
            }
            ConversionOutput::Normal(arena.add(AExpr::Literal(lv)))
        },
        Expr::Column(name) => {
            if state.output_name.is_none() {
                state.output_name = OutputName::ColumnLhs(name.clone())
            }
            ConversionOutput::Normal(arena.add(AExpr::Column(name)))
        },
        Expr::BinaryExpr { left, op, right } => {
            let l = to_aexpr_impl(owned(left), arena, state)?;
            let r = to_aexpr_impl(owned(right), arena, state)?;

            l.combine(r, |left, right| {
                arena.add(AExpr::BinaryExpr { left, op, right })
            })
        },
        Expr::Cast {
            expr,
            dtype,
            options,
        } => to_aexpr_impl(owned(expr), arena, state)?.map(|expr| {
            arena.add(AExpr::Cast {
                expr,
                dtype,
                options,
            })
        }),
        Expr::Gather {
            expr,
            idx,
            returns_scalar,
        } => {
            let expr = to_aexpr_impl(owned(expr), arena, state)?;
            let idx = to_aexpr_impl_materialized_lit(owned(idx), arena, state)?;

            expr.combine(idx, |expr, idx| {
                arena.add(AExpr::Gather {
                    expr,
                    idx,
                    returns_scalar,
                })
            })
        },
        Expr::Sort { expr, options } => to_aexpr_impl(owned(expr), arena, state)?
            .map(|expr| arena.add(AExpr::Sort { expr, options })),
        Expr::SortBy {
            expr,
            by,
            sort_options,
        } => {
            let expr = to_aexpr_impl(owned(expr), arena, state)?;
            let mut by = by
                .into_iter()
                .map(|e| to_aexpr_impl(e, arena, state))
                .collect::<PolarsResult<_>>()?;
            by.push(expr);

            ConversionOutput::combine_vec(by, |mut by| {
                let expr = by.pop();
                arena.add(AExpr::SortBy {
                    expr,
                    by,
                    sort_options,
                })
            })
        },
        Expr::Filter { input, by } => {
            let input = to_aexpr_impl(owned(input), arena, state)?;
            let by = to_aexpr_impl(owned(by), arena, state)?;
            input.combine(by, |input, by| arena.add(AExpr::Filter { input, by }))
        },
        Expr::Agg(agg) => {
            let a_agg = match agg {
                AggExpr::Min {
                    input,
                    propagate_nans,
                } => to_aexpr_impl_materialized_lit(owned(input), arena, state)?.map(|input| {
                    IRAggExpr::Min {
                        input,
                        propagate_nans,
                    }
                }),
                AggExpr::Max {
                    input,
                    propagate_nans,
                } => to_aexpr_impl_materialized_lit(owned(input), arena, state)?.map(|input| {
                    IRAggExpr::Max {
                        input,
                        propagate_nans,
                    }
                }),
                AggExpr::Median(expr) => to_aexpr_impl_materialized_lit(owned(expr), arena, state)?
                    .map(IRAggExpr::Median),
                AggExpr::NUnique(expr) => {
                    to_aexpr_impl_materialized_lit(owned(expr), arena, state)?
                        .map(IRAggExpr::NUnique)
                },
                AggExpr::First(expr) => {
                    to_aexpr_impl_materialized_lit(owned(expr), arena, state)?.map(IRAggExpr::First)
                },
                AggExpr::Last(expr) => {
                    to_aexpr_impl_materialized_lit(owned(expr), arena, state)?.map(IRAggExpr::Last)
                },
                AggExpr::Mean(expr) => {
                    to_aexpr_impl_materialized_lit(owned(expr), arena, state)?.map(IRAggExpr::Mean)
                },
                AggExpr::Implode(expr) => {
                    to_aexpr_impl_materialized_lit(owned(expr), arena, state)?
                        .map(IRAggExpr::Implode)
                },
                AggExpr::Count(expr, include_nulls) => {
                    to_aexpr_impl_materialized_lit(owned(expr), arena, state)?
                        .map(|input| IRAggExpr::Count(input, include_nulls))
                },
                AggExpr::Quantile {
                    expr,
                    quantile,
                    method,
                } => {
                    let expr = to_aexpr_impl_materialized_lit(owned(expr), arena, state)?;
                    let quantile = to_aexpr_impl_materialized_lit(owned(quantile), arena, state)?;
                    expr.combine(quantile, |expr, quantile| IRAggExpr::Quantile {
                        expr,
                        quantile,
                        method,
                    })
                },
                AggExpr::Sum(expr) => {
                    to_aexpr_impl_materialized_lit(owned(expr), arena, state)?.map(IRAggExpr::Sum)
                },
                AggExpr::Std(expr, ddof) => {
                    to_aexpr_impl_materialized_lit(owned(expr), arena, state)?
                        .map(|expr| IRAggExpr::Std(expr, ddof))
                },
                AggExpr::Var(expr, ddof) => {
                    to_aexpr_impl_materialized_lit(owned(expr), arena, state)?
                        .map(|expr| IRAggExpr::Var(expr, ddof))
                },
                AggExpr::AggGroups(expr) => {
                    to_aexpr_impl_materialized_lit(owned(expr), arena, state)?
                        .map(IRAggExpr::AggGroups)
                },
            };
            a_agg.map(|v| arena.add(AExpr::Agg(v)))
        },
        Expr::Ternary {
            predicate,
            truthy,
            falsy,
        } => {
            // Truthy must be resolved first to get the lhs name first set.
            let t = to_aexpr_impl(owned(truthy), arena, state)?;
            let p = to_aexpr_impl_materialized_lit(owned(predicate), arena, state)?;
            let f = to_aexpr_impl(owned(falsy), arena, state)?;

            ConversionOutput::combine_vec(vec![t, p, f], |vs| {
                arena.add(AExpr::Ternary {
                    predicate: vs[0],
                    truthy: vs[1],
                    falsy: vs[2],
                })
            })
        },
        Expr::AnonymousFunction {
            input,
            function,
            output_type,
            options,
        } => {
            let e = to_expr_irs(input, arena)?;
            ConversionOutput::combine_vec(e, |e| {
                set_function_output_name(&e, state, || PlSmallStr::from_static(options.fmt_str));
                arena.add(AExpr::AnonymousFunction {
                    input: e,
                    function,
                    output_type,
                    options,
                })
            })
        },
        Expr::Function {
            input,
            function,
            options,
        } => return convert_functions(input, function, options, arena, state),
        Expr::Window {
            function,
            partition_by,
            order_by,
            options,
        } => {
            // Process function first so name is correct.
            let function = to_aexpr_impl(owned(function), arena, state)?;
            let order_by = if let Some((e, options)) = order_by {
                Some((to_aexpr_impl(owned(e.clone()), arena, state)?, options))
            } else {
                None
            };

            AExpr::Window {
                function,
                partition_by: to_aexprs(partition_by, arena, state)?,
                order_by,
                options,
            }
        },
        Expr::Slice {
            input,
            offset,
            length,
        } => {
            let input = to_aexpr_impl(owned(input), arena, state)?;
            let offset = to_aexpr_impl_materialized_lit(owned(offset), arena, state)?;
            let length = to_aexpr_impl_materialized_lit(owned(length), arena, state)?;

            ConversionOutput::combine_vec(vec![input, offset, length], |vs| {
                arena.add(AExpr::Slice {
                    input: vs[0],
                    offset: vs[1],
                    length: vs[2],
                })
            })
        },
        Expr::Len => {
            if state.output_name.is_none() {
                state.output_name = OutputName::LiteralLhs(get_len_name())
            }
            ConversionOutput::Normal(arena.add(AExpr::Len))
        },
        #[cfg(feature = "dtype-struct")]
        e @ Expr::Field(_) => {
            polars_bail!(InvalidOperation: "'Expr: {}' not allowed in this context/location", e)
        },
        e @ Expr::IndexColumn(_)
        | e @ Expr::Wildcard
        | e @ Expr::Nth(_)
        | e @ Expr::SubPlan { .. }
        | e @ Expr::KeepName(_)
        | e @ Expr::Exclude(_, _)
        | e @ Expr::RenameAlias { .. }
        | e @ Expr::Columns { .. }
        | e @ Expr::DtypeColumn { .. }
        | e @ Expr::Selector(_) => {
            polars_bail!(InvalidOperation: "'Expr: {}' not allowed in this context/location", e)
        },
    })
}
