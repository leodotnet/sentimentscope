build:
	ant clean
	ant build
init:
	python scripts/init.py
	chmod +x *.sh
	mkdir logs
check:
	./check.sh
	more check.sh
movelog:
	mv *.log logs/.
sentimentspan_latent_english:
	./exp_sentimentspanlatent.sh en full
sentimentspan_latent_nohiddeninfo_english:
	./exp_sentimentspanlatent_nohiddeninfo.sh en full
sentimentspan_latent_withpostag_english:
	./exp_sentimentspanlatent_withpostag.sh en full
sentimentspan_latent_wordembedding_english:
	./exp_sentimentspanlatent_wordembedding.sh en full
sentimentspan_nonlatent_english:
	./exp_sentimentspan_nonlatent.sh en full
sentimentspan_semimarkov_nonlatent_english:
	./exp_semimarkov.sh en full
sentimentspan_semimarkov_latent_wordembedding_english:
	./exp_semimarkovlatent_wordembedding.sh en full
sentimentspan_semimarkov_latent_english:
	./exp_semimarkovlatent.sh en full
baseline_collapse_english:
	./exp_baseline_collapse.sh en full
baseline_pipeline_english:
	./exp_baseline_pipeline.sh en full


sentimentspan_latent_extend_english:
	./exp_sentimentspanlatent_extend.sh en full


sentimentspan_latent_spanish:
	./exp_sentimentspanlatent.sh es full
sentimentspan_latent_nohiddeninfo_spanish:
	./exp_sentimentspanlatent_nohiddeninfo.sh es full
sentimentspan_latent_withpostag_spanish:
	./exp_sentimentspanlatent_withpostag.sh es full
sentimentspan_latent_wordembedding_spanish:
	./exp_sentimentspanlatent_wordembedding.sh es full
sentimentspan_nonlatent_spanish:
	./exp_sentimentspan_nonlatent.sh es full
sentimentspan_semimarkov_nonlatent_spanish:
	./exp_semimarkov.sh es full
sentimentspan_semimarkov_latent_spanish:
	./exp_semimarkovlatent.sh es full
sentimentspan_semimarkov_latent_wordembedding_spanish:
	./exp_semimarkovlatent_wordembedding.sh es full
baseline_collapse_spanish:
	./exp_baseline_collapse.sh es full
baseline_pipeline_spanish:
	./exp_baseline_pipeline.sh es full


sentimentspan_latent_extend_spanish:
	./exp_sentimentspanlatent_extend.sh es full


