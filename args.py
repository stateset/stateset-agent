
self.args = {
    'output_dir': 'outputs/',
    'cache_dir': 'cache_dir/',

    'fp16': True,
    'fp16_opt_level': 'O1',
    'max_seq_length': 512,
    'train_batch_size': 8,
    'gradient_accumulation_steps': 1,
    'eval_batch_size': 8,
    'num_train_epochs': 1,
    'weight_decay': 0,
    'learning_rate': 4e-5,
    'adam_epsilon': 1e-8,
    'warmup_ratio': 0.06,
    'warmup_steps': 0,
    'max_grad_norm': 1.0,

    'logging_steps': 50,
    'save_steps': 2000,
    'evaluate_during_training': False,

    'overwrite_output_dir': False,
    'reprocess_input_data': False,

    'process_count': cpu_count() - 2 if cpu_count() > 2 else 1,
    'n_gpu': 1,
    'silent': False,

    'doc_stride': 384,
    'max_query_length': 64,
    'n_best_size': 20,
    'max_answer_length': 100,
    'null_score_diff_threshold': 0.0
}