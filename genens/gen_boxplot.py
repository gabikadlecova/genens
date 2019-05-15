import os
import sys

from tests import process_results as pr

if __name__ == "__main__":
    x_axis = sys.argv[1]
    eval_metric = sys.argv[2]
    out_dir = sys.argv[3]

    dirs = sys.argv[4:]

    if not len(dirs):
        sys.exit('Must provide at least one directory')

    dirs = [os.path.normpath(d) for d in dirs]
    columns = [os.path.basename(d) for d in dirs]
    
    df = pr.get_score_stats(columns, dirs)
    
    pr.boxplot_compare_columns(df, out_dir, x_axis=x_axis, eval_metric=eval_metric)
    stats = pr.boxplot_get_stats(df)
    print(stats.to_latex())

