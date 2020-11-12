
class BertInitConfig(object):
    def __init__(self):
        self.task_name = None
        self.model_name_or_path = None
        self.data_dir = None
        self.output_dir = None
        self.n_gpu = None
        self.device = None

        self.model_type = 'bert'
        self.cache_dir = ''
        self.gradient_accumulation_steps = 1
        self.learning_rate = 2e-5
        self.weight_decay = 0.0
        self.adam_epsilon = 1e-8
        self.max_grad_norm = 1.0
        self.num_train_epochs = 1.0
        self.max_steps = -1
        self.warmup_steps = 0
        self.logging_steps = 50
        self.save_steps = 50
        self.seed = 42
        self.fp16_opt_level = 'O1'
        self.local_rank = -1
        self.server_ip = ''
        self.server_port = ''
        self.config_name = ''
        self.tokenizer_name = ''
        self.do_train = True
        self.do_eval = True
        self.evaluate_during_training = False
        self.max_seq_length = 128
        self.do_lower_case = True
        self.per_gpu_eval_batch_size=32
        self.per_gpu_train_batch_size=4
        self.no_cuda = False
        self.overwrite_output_dir = True
        self.fp16 = False
        self.overwrite_cache = False
        self.eval_all_checkpoints = False
