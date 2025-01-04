import autorootcwd  # noqa
import sweetviz as sv

from scripts.utils import load_train_dataset

df = load_train_dataset()
report = sv.analyze(df, target_feat="reordered")
report.show_html('sweetviz_report.html')
