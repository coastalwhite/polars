mod builder;
mod equality;
mod evaluate;
mod function_expr;
#[cfg(feature = "cse")]
mod hash;
mod minterm_iter;
pub mod predicates;
mod scalar;
mod schema;
mod traverse;

use std::hash::{Hash, Hasher};

pub use function_expr::*;
#[cfg(feature = "cse")]
pub(super) use hash::traverse_and_hash_aexpr;
pub use minterm_iter::MintermIter;
use polars_compute::rolling::QuantileMethod;
use polars_core::chunked_array::cast::CastOptions;
use polars_core::prelude::*;
use polars_core::utils::{get_time_units, try_get_supertype};
use polars_utils::arena::{Arena, Node};
pub use scalar::{
    is_elementwise_ae, is_elementwise_with_ctx_ae, is_length_preserving_ae,
    is_length_preserving_with_ctx_ae, is_scalar_ae, is_scalar_with_ctx_ae,
};
use strum_macros::IntoStaticStr;
pub use traverse::*;
mod properties;
pub use aexpr::function_expr::schema::FieldsMapper;
pub use builder::AExprBuilder;
pub use properties::*;
pub use schema::ToFieldContext;

use crate::constants::LEN;
use crate::prelude::*;

#[derive(Clone, Debug, IntoStaticStr)]
#[cfg_attr(feature = "ir_serde", derive(serde::Serialize, serde::Deserialize))]
pub enum IRAggExpr {
    Min {
        input: Node,
        propagate_nans: bool,
    },
    Max {
        input: Node,
        propagate_nans: bool,
    },
    Median(Node),
    NUnique(Node),
    First(Node),
    Last(Node),
    Mean(Node),
    Implode(Node),
    Quantile {
        expr: Node,
        quantile: Node,
        method: QuantileMethod,
    },
    Sum(Node),
    Count {
        input: Node,
        include_nulls: bool,
    },
    Std(Node, u8),
    Var(Node, u8),
    AggGroups(Node),
}

impl Hash for IRAggExpr {
    fn hash<H: Hasher>(&self, state: &mut H) {
        std::mem::discriminant(self).hash(state);
        match self {
            Self::Min {
                input: _,
                propagate_nans,
            }
            | Self::Max {
                input: _,
                propagate_nans,
            } => propagate_nans.hash(state),
            Self::Quantile {
                method: interpol, ..
            } => interpol.hash(state),
            Self::Std(_, v) | Self::Var(_, v) => v.hash(state),
            Self::Count {
                input: _,
                include_nulls,
            } => include_nulls.hash(state),
            _ => {},
        }
    }
}

impl IRAggExpr {
    pub(super) fn equal_nodes(&self, other: &IRAggExpr) -> bool {
        use IRAggExpr::*;
        match (self, other) {
            (
                Min {
                    propagate_nans: l, ..
                },
                Min {
                    propagate_nans: r, ..
                },
            ) => l == r,
            (
                Max {
                    propagate_nans: l, ..
                },
                Max {
                    propagate_nans: r, ..
                },
            ) => l == r,
            (Quantile { method: l, .. }, Quantile { method: r, .. }) => l == r,
            (Std(_, l), Std(_, r)) => l == r,
            (Var(_, l), Var(_, r)) => l == r,
            _ => std::mem::discriminant(self) == std::mem::discriminant(other),
        }
    }
}

impl From<IRAggExpr> for GroupByMethod {
    fn from(value: IRAggExpr) -> Self {
        use IRAggExpr::*;
        match value {
            Min {
                input: _,
                propagate_nans,
            } => {
                if propagate_nans {
                    GroupByMethod::NanMin
                } else {
                    GroupByMethod::Min
                }
            },
            Max {
                input: _,
                propagate_nans,
            } => {
                if propagate_nans {
                    GroupByMethod::NanMax
                } else {
                    GroupByMethod::Max
                }
            },
            Median(_) => GroupByMethod::Median,
            NUnique(_) => GroupByMethod::NUnique,
            First(_) => GroupByMethod::First,
            Last(_) => GroupByMethod::Last,
            Mean(_) => GroupByMethod::Mean,
            Implode(_) => GroupByMethod::Implode,
            Sum(_) => GroupByMethod::Sum,
            Count {
                input: _,
                include_nulls,
            } => GroupByMethod::Count { include_nulls },
            Std(_, ddof) => GroupByMethod::Std(ddof),
            Var(_, ddof) => GroupByMethod::Var(ddof),
            AggGroups(_) => GroupByMethod::Groups,
            Quantile { .. } => unreachable!(),
        }
    }
}

/// IR expression node that is allocated in an [`Arena`][polars_utils::arena::Arena].
#[derive(Clone, Debug, Default)]
#[cfg_attr(feature = "ir_serde", derive(serde::Serialize, serde::Deserialize))]
pub enum AExpr {
    /// Values in a `eval` context.
    ///
    /// Equivalent of `pl.element()`.
    Element,
    Explode {
        expr: Node,
        skip_empty: bool,
    },
    Column(PlSmallStr),
    Literal(LiteralValue),
    BinaryExpr {
        left: Node,
        op: Operator,
        right: Node,
    },
    Cast {
        expr: Node,
        dtype: DataType,
        options: CastOptions,
    },
    Sort {
        expr: Node,
        options: SortOptions,
    },
    Gather {
        expr: Node,
        idx: Node,
        returns_scalar: bool,
    },
    SortBy {
        expr: Node,
        by: Vec<Node>,
        sort_options: SortMultipleOptions,
    },
    Filter {
        input: Node,
        by: Node,
    },
    Agg(IRAggExpr),
    Ternary {
        predicate: Node,
        truthy: Node,
        falsy: Node,
    },
    AnonymousFunction {
        input: Vec<ExprIR>,
        function: OpaqueColumnUdf,
        options: FunctionOptions,
        fmt_str: Box<PlSmallStr>,
    },
    /// Evaluates the `evaluation` expression on the output of the `expr`.
    ///
    /// Consequently, `expr` is an input and `evaluation` is not and needs a different schema.
    Eval {
        expr: Node,

        /// An expression that is guaranteed to not contain any column reference beyond
        /// `pl.element()` which refers to `pl.col("")`.
        evaluation: Node,

        variant: EvalVariant,
    },
    Function {
        /// Function arguments
        /// Some functions rely on aliases,
        /// for instance assignment of struct fields.
        /// Therefor we need [`ExprIr`].
        input: Vec<ExprIR>,
        /// function to apply
        function: IRFunctionExpr,
        options: FunctionOptions,
    },
    Window {
        function: Node,
        partition_by: Vec<Node>,
        order_by: Option<(Node, SortOptions)>,
        options: WindowType,
    },
    Slice {
        input: Node,
        offset: Node,
        length: Node,
    },
    #[default]
    Len,
}

pub struct ExprTraversalContext {
    pub columns_are_scalars: bool,
}

impl Default for ExprTraversalContext {
    fn default() -> Self {
        Self::DEFAULT
    }
}

impl ExprTraversalContext {
    pub const DEFAULT: ExprTraversalContext = Self {
        columns_are_scalars: false,
    };
}

impl AExpr {
    #[cfg(feature = "cse")]
    pub(crate) fn col(name: PlSmallStr) -> Self {
        AExpr::Column(name)
    }

    pub fn is_scalar(&self, arena: &Arena<AExpr>) -> bool {
        self.is_scalar_with_ctx(
            arena,
            &ExprTraversalContext {
                columns_are_scalars: false,
            },
        )
    }

    pub fn is_length_preserving(&self, arena: &Arena<AExpr>) -> bool {
        self.is_length_preserving_with_ctx(
            arena,
            &ExprTraversalContext {
                columns_are_scalars: false,
            },
        )
    }

    #[recursive::recursive]
    pub fn is_scalar_with_ctx(&self, arena: &Arena<AExpr>, ctx: &ExprTraversalContext) -> bool {
        match self {
            AExpr::Element => false,
            AExpr::Literal(lv) => lv.is_scalar(),
            AExpr::Function { options, input, .. }
            | AExpr::AnonymousFunction { options, input, .. } => {
                if options.flags.contains(FunctionFlags::RETURNS_SCALAR) {
                    true
                } else if options.is_elementwise()
                    || options.flags.contains(FunctionFlags::LENGTH_PRESERVING)
                {
                    input.iter().all(|e| e.is_scalar_with_ctx(arena, ctx))
                } else {
                    false
                }
            },
            AExpr::BinaryExpr { left, right, .. } => {
                is_scalar_with_ctx_ae(*left, arena, ctx)
                    && is_scalar_with_ctx_ae(*right, arena, ctx)
            },
            AExpr::Ternary {
                predicate,
                truthy,
                falsy,
            } => {
                is_scalar_with_ctx_ae(*predicate, arena, ctx)
                    && is_scalar_with_ctx_ae(*truthy, arena, ctx)
                    && is_scalar_with_ctx_ae(*falsy, arena, ctx)
            },
            AExpr::Agg(_) | AExpr::Len => true,
            AExpr::Cast { expr, .. } => is_scalar_with_ctx_ae(*expr, arena, ctx),
            AExpr::Eval { expr, variant, .. } => {
                variant.is_length_preserving() && is_scalar_with_ctx_ae(*expr, arena, ctx)
            },
            AExpr::Sort { expr, .. } => is_scalar_with_ctx_ae(*expr, arena, ctx),
            AExpr::Gather { returns_scalar, .. } => *returns_scalar,
            AExpr::SortBy { expr, .. } => is_scalar_with_ctx_ae(*expr, arena, ctx),
            AExpr::Window { function, .. } => is_scalar_with_ctx_ae(*function, arena, ctx),
            AExpr::Column(_) => ctx.columns_are_scalars,
            AExpr::Explode { .. } | AExpr::Filter { .. } | AExpr::Slice { .. } => false,
        }
    }

    #[recursive::recursive]
    pub fn is_length_preserving_with_ctx(
        &self,
        arena: &Arena<AExpr>,
        ctx: &ExprTraversalContext,
    ) -> bool {
        fn broadcasting_input_length_preserving(
            n: impl IntoIterator<Item = Node>,
            arena: &Arena<AExpr>,
            ctx: &ExprTraversalContext,
        ) -> bool {
            let mut num_items = 0;
            let mut num_length_preserving = 0;
            let mut num_scalar_or_length_preserving = 0;

            for n in n {
                num_items += 1;

                if is_length_preserving_with_ctx_ae(n, arena, ctx) {
                    num_length_preserving += 1;
                    num_scalar_or_length_preserving += 1;
                } else if is_scalar_with_ctx_ae(n, arena, ctx) {
                    num_scalar_or_length_preserving += 1;
                }
            }

            num_length_preserving > 0 && num_scalar_or_length_preserving == num_items
        }

        match self {
            AExpr::Element => true,
            AExpr::Column(_) => !ctx.columns_are_scalars,

            AExpr::Literal(_) | AExpr::Agg(_) | AExpr::Len => false,
            AExpr::Function { options, input, .. }
            | AExpr::AnonymousFunction { options, input, .. } => {
                if options.flags.is_elementwise() {
                    broadcasting_input_length_preserving(input.iter().map(|e| e.node()), arena, ctx)
                } else if options.flags.is_length_preserving() {
                    input
                        .iter()
                        .all(|e| e.is_length_preserving_with_ctx(arena, ctx))
                } else {
                    false
                }
            },
            AExpr::BinaryExpr { left, right, .. } => {
                broadcasting_input_length_preserving([*left, *right], arena, ctx)
            },
            AExpr::Ternary {
                predicate,
                truthy,
                falsy,
            } => broadcasting_input_length_preserving([*predicate, *truthy, *falsy], arena, ctx),
            AExpr::Cast { expr, .. } => is_length_preserving_with_ctx_ae(*expr, arena, ctx),
            AExpr::Eval { expr, variant, .. } => {
                variant.is_length_preserving()
                    && is_length_preserving_with_ctx_ae(*expr, arena, ctx)
            },
            AExpr::Sort { expr, .. } => is_length_preserving_with_ctx_ae(*expr, arena, ctx),
            AExpr::Gather {
                expr: _,
                idx,
                returns_scalar,
            } => !returns_scalar && is_length_preserving_with_ctx_ae(*idx, arena, ctx),
            AExpr::SortBy { expr, by, .. } => broadcasting_input_length_preserving(
                std::iter::once(*expr).chain(by.iter().copied()),
                arena,
                ctx,
            ),
            AExpr::Window { function, .. } => {
                is_length_preserving_with_ctx_ae(*function, arena, ctx)
            },

            AExpr::Explode { .. } | AExpr::Filter { .. } | AExpr::Slice { .. } => false,
        }
    }

    #[recursive::recursive]
    pub fn is_elementwise_with_ctx(
        &self,
        arena: &Arena<AExpr>,
        ctx: &ExprTraversalContext,
    ) -> bool {
        fn broadcasting_input_elementwise(
            n: impl IntoIterator<Item = Node>,
            arena: &Arena<AExpr>,
            ctx: &ExprTraversalContext,
        ) -> bool {
            let mut num_items = 0;
            let mut num_elementwise = 0;
            let mut num_scalar_or_elementwise = 0;

            for n in n {
                num_items += 1;

                if is_elementwise_with_ctx_ae(n, arena, ctx) {
                    num_elementwise += 1;
                    num_scalar_or_elementwise += 1;
                } else if is_scalar_with_ctx_ae(n, arena, ctx) {
                    num_scalar_or_elementwise += 1;
                }
            }

            num_elementwise > 0 && num_scalar_or_elementwise == num_items
        }

        match self {
            AExpr::Column(_) => !ctx.columns_are_scalars,

            AExpr::Function { options, input, .. }
            | AExpr::AnonymousFunction { options, input, .. } => {
                options.flags.is_elementwise()
                    && broadcasting_input_elementwise(input.iter().map(|e| e.node()), arena, ctx)
            },
            AExpr::BinaryExpr { left, right, .. } => {
                broadcasting_input_elementwise([*left, *right], arena, ctx)
            },
            AExpr::Ternary {
                predicate,
                truthy,
                falsy,
            } => broadcasting_input_elementwise([*predicate, *truthy, *falsy], arena, ctx),
            AExpr::Cast { expr, .. } => is_elementwise_with_ctx_ae(*expr, arena, ctx),
            AExpr::Eval { expr, variant, .. } => {
                variant.is_elementwise() && is_elementwise_with_ctx_ae(*expr, arena, ctx)
            },

            AExpr::Sort { .. }
            | AExpr::Gather { .. }
            | AExpr::SortBy { .. }
            | AExpr::Literal(_)
            | AExpr::Agg(_)
            | AExpr::Len
            | AExpr::Window { .. }
            | AExpr::Explode { .. }
            | AExpr::Filter { .. }
            | AExpr::Slice { .. } => false,
        }
    }

    /// Is the top-level expression fallible based on the data values.
    pub fn is_fallible_top_level(&self, arena: &Arena<AExpr>) -> bool {
        #[allow(clippy::collapsible_match, clippy::match_like_matches_macro)]
        match self {
            AExpr::Function {
                input, function, ..
            } => match function {
                IRFunctionExpr::ListExpr(f) => match f {
                    IRListFunction::Get(false) => true,
                    #[cfg(feature = "list_gather")]
                    IRListFunction::Gather(false) => true,
                    _ => false,
                },
                #[cfg(feature = "dtype-array")]
                IRFunctionExpr::ArrayExpr(f) => match f {
                    IRArrayFunction::Get(false) => true,
                    _ => false,
                },
                #[cfg(all(feature = "strings", feature = "temporal"))]
                IRFunctionExpr::StringExpr(f) => match f {
                    IRStringFunction::Strptime(_, strptime_options) => {
                        debug_assert!(input.len() <= 2);

                        let ambiguous_arg_is_infallible_scalar = input
                            .get(1)
                            .map(|x| arena.get(x.node()))
                            .is_some_and(|ae| match ae {
                                AExpr::Literal(lv) => {
                                    lv.extract_str().is_some_and(|ambiguous| match ambiguous {
                                        "earliest" | "latest" | "null" => true,
                                        "raise" => false,
                                        v => {
                                            if cfg!(debug_assertions) {
                                                panic!("unhandled parameter to ambiguous: {v}")
                                            }
                                            false
                                        },
                                    })
                                },
                                _ => false,
                            });

                        let ambiguous_is_fallible = !ambiguous_arg_is_infallible_scalar;

                        !matches!(arena.get(input[0].node()), AExpr::Literal(_))
                            && (strptime_options.strict || ambiguous_is_fallible)
                    },
                    _ => false,
                },
                _ => false,
            },
            AExpr::Cast {
                expr,
                dtype: _,
                options: CastOptions::Strict,
            } => !matches!(arena.get(*expr), AExpr::Literal(_)),
            _ => false,
        }
    }
}
