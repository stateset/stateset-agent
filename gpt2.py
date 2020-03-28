import gpt_2_simple as gpt2

gpt2.download_gpt2(model_name="117M")

sess = gpt2.start_tf_sess()
gpt2.finetune(sess, 
                "stateset.txt",model_name="117M",
                steps=1000)