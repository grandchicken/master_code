def add_modelparameter(parser):
    parser.add_argument('--num_attention_heads', type=int, default=12)
    parser.add_argument('--hidden_size', type=int, default=768)
    parser.add_argument('--attention_probs_dropout_prob', type=float, default=0.1)
    return parser