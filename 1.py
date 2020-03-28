
from simpletransformers.conv_ai import ConvAIModel


train_args = {
    "overwrite_output_dir": True,
    "reprocess_input_data": True
}

# Create a ConvAIModel
model = ConvAIModel("gpt", "gpt_personachat_cache", use_cuda=True, args=train_args)