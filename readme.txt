The guide to run Sentiment Span experiments

1. Run "make init" to initialize files and folders
2. Run "make <experiment>" for experiments

<experiment> includes
sentimentspan_latent_english
sentimentspan_latent_nohiddeninfo_english
sentimentspan_latent_withpostag_english
sentimentspan_latent_wordembedding_english
sentimentspan_nonlatent_english
sentimentspan_semimarkov_nonlatent_english
sentimentspan_semimarkov_latent_english
baseline_collapse_english
baseline_pipeline_english

sentimentspan_latent_spanish
sentimentspan_latent_nohiddeninfo_spanish
sentimentspan_latent_withpostag_spanish
sentimentspan_latent_wordembedding_spanish
sentimentspan_nonlatent_spanish
sentimentspan_semimarkov_nonlatent_spanish
sentimentspan_semimarkov_latent_spanish
baseline_collapse_spanish
baseline_pipeline_spanish

After executing one command, the program will create a unique folder for storing experiment results. You can type "make check" to show the folder. The experiment results store in experiments/sentiment/model/<modelname>/<lang>.

Notice that pipeline will involve two steps of training and evaluation automatically.

3. Run "make eval" to evaluate the results and obtain the statistics for current experiment. In addition, statistics will be stored in the folder mentioned above.