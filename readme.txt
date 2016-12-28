The guide to run Sentiment Scopes experiments

1. Run "make" to compile the source code
2. Run "make init" to initialize files and folders
3. Run "make <experiment>" for experiments

<experiment> includes
sentimentscope_latent_english
sentimentscope_latent_nohiddeninfo_english
sentimentscope_latent_withpostag_english
sentimentscope_latent_wordembedding_english
sentimentscope_nonlatent_english
sentimentscope_semimarkov_nonlatent_english
sentimentscope_semimarkov_latent_english
baseline_collapse_english
baseline_pipeline_english

sentimentscope_latent_spanish
sentimentscope_latent_nohiddeninfo_spanish
sentimentscope_latent_withpostag_spanish
sentimentscope_latent_wordembedding_spanish
sentimentscope_nonlatent_spanish
sentimentscope_semimarkov_nonlatent_spanish
sentimentscope_semimarkov_latent_spanish
baseline_collapse_spanish
baseline_pipeline_spanish

After executing one command, the program will create a unique folder for storing experiment results. You can type "make check" to see the progress logs and evaluation results in the end. The experiment results store in experiments/sentiment/model/<modelname>/<lang>.

Notice that pipeline will involve two steps of training and evaluation automatically.
