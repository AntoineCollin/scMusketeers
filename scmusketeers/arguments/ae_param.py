class AE_PARAM:
    def __init__(self, run_file):
        self.ae_activation = run_file.ae_activation
        self.ae_bottleneck_activation = run_file.ae_bottleneck_activation
        self.ae_output_activation = run_file.ae_output_activation
        self.ae_init = run_file.ae_init
        self.ae_batchnorm = run_file.ae_batchnorm
        self.ae_l1_enc_coef = run_file.ae_l1_enc_coef
        self.ae_l2_enc_coef = run_file.ae_l2_enc_coef
        self.ae_hidden_size = run_file.ae_hidden_size
        self.ae_hidden_dropout = run_file.ae_hidden_dropout
