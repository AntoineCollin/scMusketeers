class DANN_PARAM:
    def __init__(self, run_file):
        self.dann_hidden_size = run_file.dann_hidden_size
        self.dann_hidden_dropout = run_file.dann_hidden_dropout
        self.dann_batchnorm = run_file.dann_batchnorm
        self.dann_activation = run_file.dann_activation
        self.dann_output_activation = run_file.dann_output_activation
