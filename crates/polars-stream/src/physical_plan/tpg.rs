use polars_plan::plans::AExpr;
use polars_tpg::{TextPlanGraph, TpgNode, TpgValue, TpgVerbosity};
use polars_utils::arena::Arena;
use slotmap::SlotMap;

use super::{PhysNode, PhysNodeKey};

fn visualize_plan_to_tpg(
    roots: &[PhysNodeKey],
    phys_sm: &SlotMap<PhysNodeKey, PhysNode>,
    expr_arena: &Arena<AExpr>,
) -> TextPlanGraph {
    let mut next_key = 0;
    let mut nodes = Vec::with_capacity(16);
    let mut stack: Vec<(TpgKey, PhysNodeKey)> = Vec::with_capacity(16);
    let mut visited: SecondaryMap<PhysNodeKey, TpgKey> = Default::default();

    macro_rules! new_child {
        ($child:expr) => {{
            let child: Node = $child;
            if matches!(phys_sm[$child], PhysNode::Multiplexer { .. })
                && let Some(graph_key) = visited.get(&child)
            {
                *graph_key
            } else {
                let graph_key = next_key;
                next_key += 1;
                let graph_key = TpgKey(graph_key);
                visited.insert(child, graph_key);
                stack.push((graph_key, child));
                graph_key
            }
        }};
    }

    let mut roots = node_keys.len();
    for &n in roots {
        new_child!(*n);
    }

    let mut inputs_scratch = Vec::with_capacity(4);
    while let Some((graph_key, node)) = stack.pop() {
        let phys_node = &phys_sm[node];

        inputs_scratch.clear();
        visit_node_inputs(phys_node, &mut inputs_scratch);
        let children: Vec<TpgKey> = inputs_scratch
            .iter()
            .copied()
            .map(|n| new_child!(n))
            .collect();

        use {PhysNode as P, TpgNode as C, TpgVerbosity as V};

        match phys_node {
    P::InMemorySource {
        df
    } => C::new("in_memory_source"),

    Select {
        input: _,
        selectors,
        extend_original,
    } => if extend_original {
        C::new("with_columns")
    } else {
        C::new("select")
    },

    WithRowIndex {
        input: _,
        name,
        offset,
    } => {
        C::new("with_row_index").prop(V::Default, "name", name.as_str()).prop(V::Default, "offset", offset)
    },

    InputIndependentSelect {
        selectors,
    } => C::new("input_dependent_select"),

    Reduce {
        input: _,
        exprs,
    } => C::new("reduce"),

    StreamingSlice {
        input: _,
        offset,
        length,
    } => C::new("slice").prop(V::Default, "offset", offset).prop(V::Default, "length", length),

    NegativeSlice {
        input: _,
        offset,
        length,
    } => {
C::new("negative_slice").prop(V::Default, "offset", offset).prop(V::Default, "length", length)
    },

    DynamicSlice {
        input,
        offset,
        length,
    } => C::new("dynamic_slice"),

    Filter {
        input: _,
        predicate,
    } => C::new("filter"),

    SimpleProjection {
        input: _,
        columns,
    } => C::new("simple_projection"),

    InMemorySink {
        input: _,
    } => C::new("sink"),

    FileSink {
        target,
        sink_options,
        file_type,
        input,
        cloud_options,
    } => C::new("file_sink"),

    PartitionSink {
        input: _,
        base_path,
        file_path_cb,
        sink_options,
        variant,
        file_type,
        cloud_options,
        per_partition_sort_by,
        finish_callback,
    } => C::new("partition_sink"),

    SinkMultiple {
        sinks: _,
    } => continue,

    /// Generic fallback for (as-of-yet) unsupported streaming mappings.
    /// Fully sinks all data to an in-memory data frame and uses the in-memory
    /// engine to perform the map.
    InMemoryMap {
        input: _,
        map,

        /// A formatted explain of what the in-memory map. This usually calls format on the IR.
        format_str,
    } => ,

    Map {
        input: PhysStream,
        map: Arc<dyn DataFrameUdf>,
    },

    Sort {
        input: PhysStream,
        by_column: Vec<ExprIR>,
        slice: Option<(i64, usize)>,
        sort_options: SortMultipleOptions,
    },

    TopK {
        input: PhysStream,
        k: PhysStream,
        by_column: Vec<ExprIR>,
        reverse: Vec<bool>,
        nulls_last: Vec<bool>,
    },

    Repeat {
        value: PhysStream,
        repeats: PhysStream,
    },

    #[cfg(feature = "cum_agg")]
    CumAgg {
        input: PhysStream,
        kind: crate::nodes::cum_agg::CumAggKind,
    },

    // Parameter is the input stream
    Rle(PhysStream),
    RleId(PhysStream),
    PeakMinMax {
        input: PhysStream,
        is_peak_max: bool,
    },

    OrderedUnion {
        inputs: Vec<PhysStream>,
    },

    Zip {
        inputs: Vec<PhysStream>,
        /// If true shorter inputs are extended with nulls to the longest input,
        /// if false all inputs must be the same length, or have length 1 in
        /// which case they are broadcast.
        null_extend: bool,
    },

    #[allow(unused)]
    Multiplexer {
        input: PhysStream,
    },

    MultiScan {
        scan_sources: ScanSources,

        file_reader_builder: Arc<dyn FileReaderBuilder>,
        cloud_options: Option<Arc<CloudOptions>>,

        /// Columns to project from the file.
        file_projection_builder: ProjectionBuilder,
        /// Final output schema of morsels being sent out of MultiScan.
        output_schema: SchemaRef,

        row_index: Option<RowIndex>,
        pre_slice: Option<Slice>,
        predicate: Option<ExprIR>,

        hive_parts: Option<HivePartitionsDf>,
        include_file_paths: Option<PlSmallStr>,
        cast_columns_policy: CastColumnsPolicy,
        missing_columns_policy: MissingColumnsPolicy,
        forbid_extra_columns: Option<ForbidExtraColumns>,

        deletion_files: Option<DeletionFilesList>,

        /// Schema of columns contained in the file. Does not contain external columns (e.g. hive / row_index).
        file_schema: SchemaRef,
    },

    #[cfg(feature = "python")]
    PythonScan {
        options: polars_plan::plans::python::PythonOptions,
    },

    GroupBy {
        input: PhysStream,
        key: Vec<ExprIR>,
        // Must be a 'simple' expression, a singular column feeding into a single aggregate, or Len.
        aggs: Vec<ExprIR>,
    },

    EquiJoin {
        input_left: PhysStream,
        input_right: PhysStream,
        left_on: Vec<ExprIR>,
        right_on: Vec<ExprIR>,
        args: JoinArgs,
    },

    SemiAntiJoin {
        input_left: PhysStream,
        input_right: PhysStream,
        left_on: Vec<ExprIR>,
        right_on: Vec<ExprIR>,
        args: JoinArgs,
        output_bool: bool,
    },

    CrossJoin {
        input_left: PhysStream,
        input_right: PhysStream,
        args: JoinArgs,
    },

    /// Generic fallback for (as-of-yet) unsupported streaming joins.
    /// Fully sinks all data to in-memory data frames and uses the in-memory
    /// engine to perform the join.
    InMemoryJoin {
        input_left: PhysStream,
        input_right: PhysStream,
        left_on: Vec<ExprIR>,
        right_on: Vec<ExprIR>,
        args: JoinArgs,
        options: Option<JoinTypeOptionsIR>,
    },

    #[cfg(feature = "merge_sorted")]
    MergeSorted {
        input_left: PhysStream,
        input_right: PhysStream,
    },
        }
    }

    TextPlanGraph {
        title: "Streaming Physical".into(),
        roots,
        nodes,
        legend: Default::default(),
    }
}
