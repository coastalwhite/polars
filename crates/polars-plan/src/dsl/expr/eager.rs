use polars_core::frame::DataFrame;
use polars_error::PolarsResult;
use polars_utils::idx_vec::UnitVec;
use polars_utils::pl_str::PlSmallStr;

use super::Expr;
use crate::dsl::{DslPlan, FunctionExpr};

pub enum EagerExpr {
    ShrinkType,
}

pub fn contains_eager_expression_amortized<'a>(
    expr: &'a Expr,
    stack: &mut UnitVec<&'a Expr>,
) -> bool {
    stack.clear();
    expr.nodes(stack);
    while let Some(expr) = stack.pop() {
        if to_eager_expression_top_level(expr).is_some() {
            return true;
        }
        expr.nodes(stack);
    }
    false
}

pub fn to_eager_expression_top_level(expr: &Expr) -> Option<EagerExpr> {
    match expr {
        Expr::Alias(..)
        | Expr::Column(..)
        | Expr::Selector(..)
        | Expr::Literal(..)
        | Expr::DataTypeFunction(..)
        | Expr::BinaryExpr { .. }
        | Expr::Cast { .. }
        | Expr::Sort { .. }
        | Expr::Gather { .. }
        | Expr::SortBy { .. }
        | Expr::Agg(_)
        | Expr::Ternary { .. }
        | Expr::Explode { .. }
        | Expr::Filter { .. }
        | Expr::Window { .. }
        | Expr::Slice { .. }
        | Expr::KeepName(..)
        | Expr::Len
        | Expr::Field(..)
        | Expr::AnonymousFunction { .. }
        | Expr::Eval { .. }
        | Expr::SubPlan(..)
        | Expr::RenameAlias { .. } => None,

        Expr::Function { input, function } => todo!(),
    }
}

pub fn to_eager_function_top_level(f: &FunctionExpr) -> Option<EagerExpr> {
    match f {
        FunctionExpr::ShrinkType => Some(EagerExpr::ShrinkType),

        _ => None,
    }
}

pub enum ResolvedEager {
    Lazy,
    Resolved(DataFrame),
}

pub fn select_expr_and_resolve_eager(
    exprs: &mut Vec<Expr>,
    input: &DslPlan,
    evaluate: &dyn Fn(&DslPlan) -> PolarsResult<DataFrame>,
    hstack: bool,
) -> PolarsResult<ResolvedEager> {
    let mut stack = UnitVec::new();
    if exprs
        .iter()
        .all(|expr| !contains_eager_expression_amortized(expr, &mut stack))
    {
        return Ok(ResolvedEager::Lazy);
    }

    let input = evaluate(input)?;

    let mut output = DataFrame::empty();
    if hstack {
        output.extend(&input)?;
    }

    for e in exprs {
        let mut current = input.clone();

        while e.contains_eager_expression_amortized(stack)
        
    }
}
