{
    logdir: "/apdcephfs/share_1316500/nlphuang/huangrongjie1/Speech2SQL/logdir/task_012_single_gpu",
    model_config: "/apdcephfs/share_1316500/nlphuang/huangrongjie1/Speech2SQL/configs/spider/nl2code-glove.jsonnet",
    model_config_args: {
        att: 0,
        cv_link: true,
        clause_order: null, # strings like "SWGOIF"
        level_traversal: false,
        enumerate_order: false,
    },

    eval_name: "glove_run_%s_%d" % [self.eval_use_heuristic, self.eval_beam_size],
    eval_output: "/apdcephfs/share_1316500/nlphuang/huangrongjie1/Speech2SQL/logdir/task_012_single_gpu/ie_dirs",
    eval_beam_size: 1,
    eval_use_heuristic: true,   
    # eval_steps: [ 1000 * x + 100 for x in std.range(30, 39)] + [40000],
    # eval_steps: [ 1000 * x + 100 for x in std.range(10, 11)],
    eval_steps: [1000 * x + 100 for x in std.range(5,12)]+[12900],
    eval_section: "val",
}