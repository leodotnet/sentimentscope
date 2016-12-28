build:
	ant clean
	ant build
init:
	python scripts/init.py
	chmod +x *.sh
check:
	./check.sh
movelog:
	mv *.log logs/.



baseline_joint_english:
	nohup ./exp_baseline_jointexact.sh en full &
sentimentspan_latent_english_dev:
	nohup ./exp_sentimentspanlatent.sh en dev &
sentimentspan_latent_wordembedding_english_dev:
	nohup ./exp_sentimentspanlatent_wordembedding.sh en dev &

sentimentspan_latent_english:
	nohup ./exp_sentimentspanlatent.sh en full &
sentimentspan_latent_nosp_english:
	nohup ./exp_sentimentspanlatent_nosp.sh en full &
sentimentspan_latent_withpostag_english:
	nohup ./exp_sentimentspanlatent_withpostag.sh en full &
sentimentspan_latent_wordembedding_english:
	nohup ./exp_sentimentspanlatent_wordembedding.sh en full &
sentimentspan_nonlatent_english:
	nohup ./exp_sentimentspan_nonlatent.sh en full &
sentimentspan_nonlatent_nosp_english:
	nohup ./exp_sentimentspan_nonlatent_nosp.sh en full &


sentimentspan_semimarkov_nonlatent_english:
	nohup ./exp_semimarkov.sh en full &
sentimentspan_semimarkov_latent_wordembedding_english:
	nohup ./exp_semimarkovlatent_wordembedding.sh en full &
sentimentspan_semimarkov_latent_english:
	nohup ./exp_semimarkovlatent.sh en full &
baseline_collapse_english:
	nohup ./exp_baseline_collapse.sh en full &
baseline_pipeline_english:
	nohup ./exp_baseline_pipeline.sh en full &


sentimentspan_latent_extend_english:
	nohup ./exp_sentimentspanlatent_extend.sh en full &


sentimentspan_latent_spanish:
	nohup ./exp_sentimentspanlatent.sh es full &
sentimentspan_latent_nosp_spanish:
	nohup ./exp_sentimentspanlatent_nosp.sh es full &
sentimentspan_latent_withpostag_spanish:
	nohup ./exp_sentimentspanlatent_withpostag.sh es full &
sentimentspan_latent_wordembedding_spanish:
	nohup ./exp_sentimentspanlatent_wordembedding.sh es full &
sentimentspan_nonlatent_spanish:
	nohup ./exp_sentimentspan_nonlatent.sh es full &
sentimentspan_nonlatent_nosp_spanish:
	nohup ./exp_sentimentspan_nonlatent_nosp.sh es full &

sentimentspan_semimarkov_nonlatent_spanish:
	nohup ./exp_semimarkov.sh es full &
sentimentspan_semimarkov_latent_spanish:
	nohup ./exp_semimarkovlatent.sh es full &
sentimentspan_semimarkov_latent_wordembedding_spanish:
	nohup ./exp_semimarkovlatent_wordembedding.sh es full &
baseline_collapse_spanish:
	nohup ./exp_baseline_collapse.sh es full &
baseline_pipeline_spanish:
	nohup ./exp_baseline_pipeline.sh es full &
baseline_joint_spanish:
	nohup ./exp_baseline_jointexact.sh es full &


sentimentspan_additional_restaurant_english:
	nohup ./exp_sentimentspanlatent_additional.sh en restaurant

sentimentspan_additional_restaurant_spanish:
	nohup ./exp_sentimentspanlatent_additional.sh es restaurant

sentimentspan_additional_social_spanish:
	nohup ./exp_sentimentspanlatent_additional.sh es social

sentimentspan_additional_stompl_spanish:
	nohup ./exp_sentimentspanlatent_additional.sh es stompl






