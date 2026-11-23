"""Start the current six-epoch curriculum in a new directory, without replacing the approved model."""
def run_training(args):
    if args.epochs != 6:
        raise ValueError('The approved curriculum has two stages of three epochs; use --epochs 6')
    if args.output.exists():
        raise FileExistsError('Use a new output directory')
    from tools.training import refine_four_metrics
    refine_four_metrics.OUT=args.output.resolve()
    refine_four_metrics.run()
