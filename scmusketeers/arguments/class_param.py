from scmusketeers.tools.utils import str2bool


class CLASS_PARAM:

    class_hidden_size = [64]
    class_hidden_dropout = None
    class_batchnorm = True
    class_activation = "relu"
    class_output_activation = "softmax"

    def __init__(self, run_file):
        self.class_hidden_size = run_file.class_hidden_size
        self.class_hidden_dropout = run_file.class_hidden_dropout
        self.class_batchnorm = run_file.class_batchnorm
        self.class_activation = run_file.class_activation
        self.class_output_activation = run_file.class_output_activation
    
    def init_arguments(self, class_group):
        class_group.add_argument(
            "--class_hidden_size",
            type=int,
            nargs="+",
            default=[64],
            help="Hidden sizes of the successive classification layers",
        )
        class_group.add_argument(
            "--class_hidden_dropout", type=float, nargs="?", default=None, help=""
        )
        class_group.add_argument(
            "--class_batchnorm",
            type=str2bool.str2bool,
            nargs="?",
            const=True,
            default=True,
            help="",
        )
        class_group.add_argument(
            "--class_activation", type=str, nargs="?", default="relu", help=""
        )
        class_group.add_argument(
            "--class_output_activation",
            type=str,
            nargs="?",
            default="softmax",
            help="",
        )

